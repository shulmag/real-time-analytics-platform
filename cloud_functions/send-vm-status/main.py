'''
Description: Send email with the status of every VM. Helps identify VMs that are running and should not be.
'''
from datetime import datetime
from typing import Dict, Iterable
from pytz import timezone

import smtplib
from email.mime.text import MIMEText

from google.cloud import compute_v1, secretmanager


EASTERN = timezone('US/Eastern')

PROJECT_ID = 'eng-reactor-287421'

ALWAYS_RUNNING_INSTANCES = {'ftp', 'redis-forwarder', 'windows-server-mbs'}


def list_all_instances(project_id: str) -> Dict[str, Iterable[compute_v1.Instance]]:
    '''Returns a dictionary of all instances present in a project, grouped by their zone.

    Args:
        project_id: project ID or project number of the Cloud project you want to use.
    Returns:
        A dictionary with zone names as keys (in form of "zones/{zone_name}") and
        iterable collections of Instance objects as values.
    '''
    instance_client = compute_v1.InstancesClient()
    request = compute_v1.AggregatedListInstancesRequest()
    request.project = project_id
    # Use the `max_results` parameter to limit the number of results that the API returns per response page.
    request.max_results = 50

    agg_list = instance_client.aggregated_list(request=request)
    instances = {}
    # Despite using the `max_results` parameter, you don't need to handle the pagination yourself. The returned `AggregatedListPager` object handles pagination automatically, returning separated pages as you iterate over the results.
    for zone, response in agg_list:
        if response.instances:
            for instance in response.instances:
                instances[instance.name] = {
                    'status': instance.status,
                    'machine_type': instance.machine_type,
                    'last_start_timestamp': instance.last_start_timestamp,
                    'last_stop_timestamp': instance.last_stop_timestamp,
                    'zone': zone,
                }
    return instances


def access_secret_version(secret_id: str, project_id: str = 'eng-reactor-287421', version_id='latest'):
    name = f'projects/{project_id}/secrets/{secret_id}/versions/{version_id}'
    response = secretmanager.SecretManagerServiceClient().access_secret_version(request={'name': name})
    payload = response.payload.data.decode('UTF-8')
    return payload


def send_email(subject, error_message):
    receiver_email = 'ficc-eng@ficc.ai'
    sender_email = access_secret_version('notifications_username')
    sender_password = access_secret_version('notifications_password')

    msg = MIMEText(error_message)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email

    smtp_server = 'smtp.gmail.com'
    port = 587

    with smtplib.SMTP(smtp_server, port) as server:
        try:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        except Exception as e:
            print(e)
        finally:
            server.quit()


def truncate_machine_type(machine_type: str) -> str:
    '''`machine_type` comes in as a long hyperlink, but the useful part is just the zone and the machine type.
    
    >>> truncate_machine_type('https://www.googleapis.com/compute/v1/projects/eng-reactor-287421/zones/us-west1-b/machineTypes/custom-32-307200-ext')
    'us-west1-b/custom-32-307200-ext'
    '''
    machine_type_split = machine_type.split('/')
    zones_idx = machine_type_split.index('zones')
    return f'{machine_type_split[zones_idx + 1]}/{machine_type_split[-1]}'


def main(cloudevent):
    instances = list_all_instances(PROJECT_ID)
    running, not_running = ['\nRUNNING Instances'], ['\nOther Instances']
    running_instances = set()
    for instance_name in sorted(instances.keys()):    # alphabetize for easier reading
        instance = instances[instance_name]
        info = f"{instance_name} ({truncate_machine_type(instance['machine_type'])}). last start time: {instance['last_start_timestamp'][:19]}"
        if instance['status'] == 'RUNNING':
            running_instances.add(instance_name)
            running.append(info)
        else:
            not_running.append(f"{info}, last stop time: {instance['last_stop_timestamp'][:19]}")
    
    if running_instances != ALWAYS_RUNNING_INSTANCES:    # only send email if instances that are not supposed to be running are running
        current_datetime_minute = datetime.now(EASTERN).strftime('%Y-%m-%d %H:%M')
        vm_info_all = f'The following VMs should always be running: {", ".join(ALWAYS_RUNNING_INSTANCES)}. The other ones should not.\n'
        vm_index = 1
        for vm_info in running + not_running:
            prefix = ''
            if 'last start time:' in vm_info:    # if 'last_start_time' in the print output line, then the line refers to a VM instance
                prefix = f'{vm_index}. '
                vm_index += 1
            vm_info_all += f'{prefix}{vm_info}\n'
        send_email(f'Extra VMs are running as of {current_datetime_minute} ET', vm_info_all)
    return f'The following instances were running: {running_instances}'
