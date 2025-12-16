'''
Description: Upload CSV to SFTP server. The particular SFTP server is dedicated to InvestorTools.
'''
import pysftp
import paramiko

from auxiliary_variables import SFTP_HOST, SFTP_USERNAME, SFTP_PASSWORD, GOOGLE_CLOUD_BUCKET, STORAGE_CLIENT
from auxiliary_functions import run_multiple_times_before_failing, download_file


SFTP_PORT = 22    # default SFTP port is 22


@run_multiple_times_before_failing((paramiko.ssh_exception.SSHException,), 10)    # catches transient connectivity issue: `paramiko.ssh_exception.SSHException: Error reading SSH protocol banner`
def upload_file_to_sftp(filepath: str, sftp_host: str = SFTP_HOST, sftp_username: str = SFTP_USERNAME, sftp_password: str = SFTP_PASSWORD) -> str:
    '''Upload a file located at `filepath` to a SFTP server specified by `sftp_host`.'''
    # add all host keys to the options which can be generated with `ssh-keyscan sftp.ficc.ai` on the terminal and is needed for security purposes in production
    cnopts = pysftp.CnOpts()
    # cnopts.hostkeys.add(sftp_host, 'ssh-ed25519', 'AAAAC3NzaC1lZDI1NTE5AAAAIAT3Yeyze9vIcmbUxrtzS+Zt5XU6GeThwIN/Bm/iDNK5')
    # cnopts.hostkeys.add(sftp_host, 'ssh-rsa', 'AAAAB3NzaC1yc2EAAAADAQABAAABgQDLJnt97T2KXmO+/hTHGT2waxDn7c2cYEUWj2ru9D49xNbbjzD1tCloLyBvdBiV5psynu5RQBO3mTfhduwH+CI7zXoCNsydHxn7PpK8kARrnjqCC6v+1iMSG5AvmY0F/w0yQ4Km3tfm0AcWv0s6i7OvKABtvnsBkCgo4AD6uaxxFmZz6Xc+6xcyaLDa2uzRAmtLOSQdtdjUlovV45nw4J/p4ggbeCbhzn7SuP/mpbXmsSKC21T1Vwqt0C9doPMLfDFWyjIIr4CMocNOBZ5Ie7m1zXMgIi9KWMN76+6JYq+JBk67E/aohGeMgBDtF1YCjIRAw9vsiBAWI14R+M/jSzmVMgySJWAi+7l+M2lMROxyywfCPAeze1438AgJWS9UQ35Rcvrj3r8NyRvfjasa2EZcxa4QstoRsYv41lPwkfCRzjf6qwlwAHOmDDo6Jpvo4DMNzA+2npTbWz61bH2hVtvi3cf5VjipiB5uBLpD0dXg2863CKiC6lMkaExVBvUT3a8')
    # cnopts.hostkeys.add(sftp_host, 'ecdsa-sha2-nistp256', 'AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBHmefwm3sCi7VK0iS91dq/PtQgrpmlc3Mnx6gGJoMLhjRDFAM0+kKButa8hFTBi4UkItKuhhqsvpCwkPUVIUqtY=')
    local_sftp_known_hosts_filepath = '/tmp/sftp_known_hosts'
    download_file(STORAGE_CLIENT, GOOGLE_CLOUD_BUCKET, 'sftp_known_hosts', local_sftp_known_hosts_filepath)
    cnopts.hostkeys.load(local_sftp_known_hosts_filepath)    # stored in a local file called `sftp_known_hosts`

    with pysftp.Connection(host=sftp_host, username=sftp_username, password=sftp_password, port=SFTP_PORT, cnopts=cnopts) as sftp:
        print(f'Connection successfully established to {sftp_host}')

        # upload the file to the SFTP server
        remote_filepath = f'uploads/{filepath}'
        sftp.put(filepath, remote_filepath)
        print(f'File {filepath} successfully uploaded to {remote_filepath} in SFTP host: {sftp_host}')
