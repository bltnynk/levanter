import re
import subprocess


def cancel_all_ray_jobs():
    # Run `ray job list` and capture the output
    try:
        result = subprocess.run(["ray", "job", "list"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError:
        print("Ray CLI is not installed or not in PATH.")
        return

    if result.returncode != 0:
        print(f"Error listing jobs: {result.stderr.strip()}")
        return

    # Extract job IDs using regex
    job_list_output = result.stdout
    job_ids = re.findall(r"job_id='([a-fA-F0-9]+)'", job_list_output)

    if not job_ids:
        print("No jobs found to cancel.")
        return

    print(f"Found {len(job_ids)} job(s) to cancel: {', '.join(job_ids)}")

    # Cancel each job
    for job_id in job_ids:
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


if __name__ == "__main__":
    cancel_all_ray_jobs()
