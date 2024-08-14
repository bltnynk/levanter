import argparse
import base64
import getpass
import os
import subprocess
import tempfile
import textwrap
import time

import push_docker
from helpers import cli


def _default_run_id():
    """Generate a run ID for wandb and continuation.

    Wandb expects a base36 encoded ID of exactly 8 lowercase characters
    or it won't generate a display name."""
    rng_bytes = os.urandom(16)
    run_id = base64.b32encode(rng_bytes)[:8].lower()
    run_id = run_id.decode("utf-8")
    assert len(run_id) == 8
    for char in run_id:
        assert char in "abcdefghijklmnopqrstuvwxyz0123456789"
    return run_id


def main():
    parser = argparse.ArgumentParser()
    config = cli.load_config()

    cli.add_arg(parser, config, ["--dry-run"], action="store_true", default=False)
    cli.add_arg(parser, config, ["--yes"], action="store_true", default=False)
    cli.add_arg(parser, config, ["--detach"], action="store_true", default=False)

    cli.add_arg(parser, config, ["--zone"], type=str, default="us-central2-b")
    cli.add_arg(parser, config, ["--use_spot"], action="store_true", default=False)
    cli.add_arg(parser, config, ["--tpu_type"], type=str, required=True)

    cli.add_arg(parser, config, ["--image_name"], default=f"levanter-{getpass.getuser()}")
    cli.add_arg(parser, config, ["--github_user"], required=True, type=str)
    cli.add_arg(parser, config, ["--github_token"], required=True, type=str)

    cli.add_arg(parser, config, ["--run_id"], default=_default_run_id(), type=str)

    parser.add_argument(
        "-e", "--env", action="append", nargs=2, metavar=("KEY", "VALUE"), default=config.get("env", {}).items()
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    dry_run = args.dry_run
    yes = args.yes
    detach = args.detach

    zone = args.zone
    region = "-".join(zone.split("-")[:-1])
    use_spot = args.use_spot
    tpu_type = args.tpu_type

    image_id = args.image_name
    github_user = args.github_user
    github_token = args.github_token

    run_id = args.run_id
    env = {k: v for k, v in args.env}

    command = args.command

    if "WANDB_PROJECT" not in env:
        env["WANDB_PROJECT"] = "levanter"

    if command[0] == "--":
        command = command[1:]

    tag = int(time.time())

    full_image_id = push_docker.push_to_github(
        local_image=image_id,
        tag=tag,
        github_user=github_user,
        github_token=github_token,
        docker_file="docker/tpu/Dockerfile.incremental",
    )

    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()

    print(github_token)
    sky_spec = textwrap.dedent(
        f"""
        name: {run_id}
        resources:
          cloud: gcp
          region: {region}
          zone: {zone}
          use_spot: {use_spot}
          accelerators: tpu-{tpu_type}:1
          accelerator_args:
            tpu_vm: True
            runtime_version: tpu-ubuntu2204-base
        run: |
          sudo docker login -u {github_user} -p {github_token} ghcr.io
          command="sudo docker pull {full_image_id}"

          # Maximum number of attempts
          max_attempts=3

          # Attempt counter
          attempt=1
          # Loop until the command succeeds or we reach the maximum number of attempts
          while [ $attempt -le $max_attempts ]; do
              echo "Attempt $attempt of $max_attempts..."
              # Run the command
              $command

              # Check if the command succeeded
              if [ $? -eq 0 ]; then
              echo "Command succeeded."
              break
              else
              echo "Command failed."
              # Increment the attempt counter
              ((attempt++))

              # If we've reached the maximum number of attempts, exit with an error
              if [ $attempt -gt $max_attempts ]; then
                  echo "Command failed after $max_attempts attempts."
                  exit 1
              fi
              fi
          done
          sudo docker run -t --rm --name=levanter --privileged \\
              --shm-size=32gb --net=host --init -v /home/{getpass.getuser()}:/home/levanter \\
              {' '.join(f"-e {k}='{v}'" for k, v in env.items())} \\
              -e WANDB_DOCKER={image_id} -e RUN_ID={run_id} -e GIT_COMMIT={git_commit} \\
              -v /tmp:/tmp --workdir /home/levanter \\
              {full_image_id} \\
              {' '.join(command)}
    """
    )

    with tempfile.NamedTemporaryFile("w", prefix=f"sky_launch_{run_id}", suffix=".yaml", delete=False) as f:
        f.write(sky_spec)
        f.flush()
        sky_spec_filename = f.name
        print(f"Saved sky spec to {sky_spec_filename}")
        if dry_run:
            return
        sky_cmd = ["sky", "jobs", "launch"]
        if yes:
            sky_cmd.append("--yes")
        if detach:
            sky_cmd.append("--detach-run")
        sky_cmd.append(sky_spec_filename)
        subprocess.check_call(sky_cmd)
        print("Launched job with run ID", run_id)


if __name__ == "__main__":
    main()
