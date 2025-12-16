When adding a single job into the repository, use the following command:
```
gcloud scheduler jobs describe <job_name> --location=<location> --format=json > <job_name>.json
```
Note that JSON files are ignored due to the `.gitignore`, and so to add this file to the repo, use `-f`, i.e, `git add -f <job_name>.json`.

---

To get all of the cloud scheduler jobs in JSON format into a file, run the following command:
```
gcloud scheduler jobs list --location=us-east4 --format=json >> all_jobs.json
```
Note that this will only get the jobs for location `us-east4`. We have jobs in two other regions: `us-central1` and `us-west1`, so run the above command twice more for each of these regions.

Now, open `all_jobs.json` in a text editor and search for brackets, that indicate the end of one the list of the jobs for one of the regions and the start of the list of the jobs for another region. Remove these brackets and replace it with a comma so that the JSON file is just one continuous list of items.

Run the following Python script to export each job in `all_jobs.json` into a separate file:
```
import json
import os

# Load all Cloud Scheduler jobs
with open('all_jobs.json', 'r') as f:
    jobs = json.load(f)

# Create a directory to store individual job files
output_dir = 'cloud_scheduler_jobs'
os.makedirs(output_dir, exist_ok=True)

# Loop through each job and save it as a separate file
for job in jobs:
    job_id = job.get('name', 'unknown_job').split('/')[-1]  # Extract job name
    job_file = os.path.join(output_dir, f'{job_id}.json')
    
    with open(job_file, 'w') as f:
        json.dump(job, f, indent=4)

print(f'Saved {len(jobs)} jobs in {output_dir} directory.')
```
Now, delete `all_jobs.json` and the above script from the directory.
