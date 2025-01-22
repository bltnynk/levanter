import re
import subprocess


def cancel_and_delete_ray_jobs():
    # Run `ray job list` and capture the output
    try:
        result = subprocess.run(["ray", "job", "list"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError:
        print("Ray CLI is not installed or not in PATH.")
        return

    if result.returncode != 0:
        print(f"Error listing jobs: {result.stderr.strip()}")
        return

    # Extract job IDs and statuses using regex
    job_list_output = result.stdout
    jobs = re.findall(r"job_id='([a-fA-F0-9]+)'.*?status=<JobStatus\.([A-Z_]+):", job_list_output)

    if not jobs:
        print("No jobs found.")
        return

    running_jobs = [job_id for job_id, status in jobs if status in ["RUNNING", "PENDING"]]
    stopped_jobs = [job_id for job_id, status in jobs if status not in ["RUNNING", "PENDING"]]

    # Cancel running jobs
    if running_jobs:
        print(f"Found {len(running_jobs)} running job(s) to cancel: {', '.join(running_jobs)}")
        for job_id in running_jobs:
            try:
                stop_result = subprocess.run(
                    ["ray", "job", "stop", job_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                if stop_result.returncode == 0:
                    print(f"Successfully canceled job {job_id}.")
                else:
                    print(f"Failed to cancel job {job_id}: {stop_result.stderr.strip()}")
            except Exception as e:
                print(f"Error while stopping job {job_id}: {e}")
    else:
        print("No running jobs to cancel.")

    # Delete stopped jobs
    if stopped_jobs:
        print(f"Found {len(stopped_jobs)} stopped job(s) to delete: {', '.join(stopped_jobs)}")
        for job_id in stopped_jobs:
            try:
                delete_result = subprocess.run(
                    ["ray", "job", "delete", job_id], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                )
                if delete_result.returncode == 0:
                    print(f"Successfully deleted job {job_id}.")
                else:
                    print(f"Failed to delete job {job_id}: {delete_result.stderr.strip()}")
            except Exception as e:
                print(f"Error while deleting job {job_id}: {e}")
    else:
        print("No stopped jobs to delete.")


if __name__ == "__main__":
    cancel_and_delete_ray_jobs()
