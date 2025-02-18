import asyncio
import re
import subprocess


async def run_command(ctx, command):
    async with ctx:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        retcode = await process.wait()
        if retcode != 0:
            print(f"Error running command: {command}")
        else:
            print(f"Successfully ran command: {command}")


async def cancel_and_delete_ray_jobs():
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
    # stopped_jobs = [job_id for job_id, status in jobs if status not in ["RUNNING", "PENDING"]]
    stopped_jobs = []

    # Cancel running jobs
    semaphore = asyncio.Semaphore(8)
    if running_jobs:
        stop_cmds = [run_command(semaphore, f"ray job stop {job_id}") for job_id in running_jobs]
        await asyncio.gather(*stop_cmds)
    else:
        print("No running jobs to cancel.")

    # Delete stopped jobs
    if stopped_jobs:
        delete_cmds = [run_command(semaphore, f"ray job delete {job_id}") for job_id in stopped_jobs]
        await asyncio.gather(*delete_cmds)
    else:
        print("No stopped jobs to delete.")


if __name__ == "__main__":
    asyncio.run(cancel_and_delete_ray_jobs())
