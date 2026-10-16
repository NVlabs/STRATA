#!/usr/bin/env python
# ruff: noqa: S603 S607 S103
import argparse
import glob
import inspect
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

import dotenv

dotenv.load_dotenv()

BUCKET = "s3://screamcast-runs/"

project_root = os.environ["PROJECT_ROOT"]

root_parser = argparse.ArgumentParser()

subparsers = root_parser.add_subparsers(dest="command", required=True)
parser = subparsers.add_parser("submit")
parser.add_argument("--name", type=str, default="")
parser.add_argument("--account", type=str, default="coreai_climate_earth2")
parser.add_argument("--stage-only", action="store_true")
parser.add_argument("--nodes", type=int, default=2)
parser.add_argument(
    "--queue",
    type=str,
    default=None,
    help="Override Slurm queue/QoS (e.g. debug, regular). Overrides profile.",
)
parser.add_argument(
    "--constraint",
    "-C",
    dest="constraint",
    type=str,
    default=None,
    help="Override Slurm constraint (e.g. gpu or gpu&hbm80g). Overrides profile.",
)
parser.add_argument(
    "--time",
    "-t",
    dest="time",
    type=str,
    default=None,
    help="Override walltime (HH:MM:SS). Overrides profile.",
)
parser.add_argument(
    "-f",
    "--force",
    action="store_true",
    help="If passed, overwrite any existing directory.",
)

parser = subparsers.add_parser("resubmit")
parser.add_argument("name")
parser.add_argument("--nodes", type=int, default=2)
parser.add_argument("--account", type=str, default="coreai_climate_earth2")
parser.add_argument(
    "--queue",
    type=str,
    default=None,
    help="Override Slurm queue/QoS (e.g. debug, regular). Overrides profile.",
)
parser.add_argument(
    "--constraint",
    "-C",
    dest="constraint",
    type=str,
    default=None,
    help="Override Slurm constraint (e.g. gpu or gpu&hbm80g). Overrides profile.",
)
parser.add_argument(
    "--time",
    "-t",
    dest="time",
    type=str,
    default=None,
    help="Override walltime (HH:MM:SS). Overrides profile.",
)

parser = subparsers.add_parser("list")
parser = subparsers.add_parser("path")
parser.add_argument("name")

parser = subparsers.add_parser("logs")
parser.add_argument("name")
parser.add_argument("--follow", "-f", action="store_true")

parser = subparsers.add_parser("cd")
parser.add_argument("name")

parser = subparsers.add_parser("push")
parser.add_argument("name")

_funcs = {}


def command(help=""):
    def myfunc(func):
        parser = subparsers.add_parser(func.__name__, help=help)
        parser.set_defaults(func=func)
        signature = inspect.signature(func)
        names = []
        for name, param in signature.parameters.items():
            if param.annotation == param.empty:
                raise ValueError("No annotation found for {name}.")
            if param.default == param.empty:
                parser.add_argument(name)
            else:
                parser.add_argument(
                    "--" + name, default=param.default, type=param.annotation
                )
            names.append(name)
        _funcs[func] = names
        return func

    return myfunc


def get_submit_script():
    return os.path.abspath(os.environ["SUBMIT_SCRIPT"])


def submit(
    stage_dir,
    name,
    nodes: int,
    account: str = "coreai_climate_earth2",
    queue: str = None,
    constraint: str = None,
    time: str = None,
):
    gpus = nodes * 8
    job_name = f"{account}-research.screamcast-{name}"
    os.environ["RUN_CMD"] = "./run.sh"
    gpu_flags = []
    if os.environ.get("SBATCH_GPU_FLAG", "Y") == "Y":
        gpu_flags = ["--gpus", str(gpus)]
    metrics_flag = []
    if os.environ.get("LOG_EOS_NODE_METRICS", "N") == "Y":
        metrics_flag = ["--comment=metrics"]
    sbatch_overrides = []
    if queue:
        sbatch_overrides += ["-q", queue]
    if constraint:
        sbatch_overrides += ["-C", constraint]
    if time:
        sbatch_overrides += ["-t", time]

    subprocess.check_call(
        ["sbatch"]
        + gpu_flags
        + [
            "--nodes",
            str(nodes),
            "--job-name",
            job_name,
            "--account",
            account,
        ]
        + metrics_flag
        + sbatch_overrides
        + [
            get_submit_script(),
        ],
        cwd=stage_dir,
    )


def stage(name, cli_args, overwrite=False):
    # Ensure PROJECT_ROOT is set
    if not project_root:
        raise EnvironmentError(
            "The PROJECT_ROOT environment variable is not set. Please set it to a valid directory."
        )

    # Get current Git commit hash
    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except subprocess.SubprocessError:
        git_commit = "Not a git repository"

    # Create a timestamped folder for staging within PROJECT_ROOT
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not name:
        stage_dir = os.path.join(project_root, f"staged_experiment_{timestamp}")
    else:
        stage_dir = os.path.join(project_root, name)

    if overwrite:
        print(f"Removing {stage_dir}")
        shutil.rmtree(stage_dir)

    try:
        os.makedirs(stage_dir)
    except FileExistsError:
        print(f"{stage_dir} exists. Add -f to overwrite.")
        sys.exit(1)

    # Copy the current working directory to the stage folder, excluding .git
    for item in os.listdir("."):
        if item == ".git":
            continue
        source = os.path.join(".", item)
        dest = os.path.join(stage_dir, item)
        if os.path.isdir(source):
            shutil.copytree(source, dest)
        else:
            shutil.copy2(source, dest)

    # Write a run.sh script with the CLI command
    run_script_path = os.path.join(stage_dir, "run.sh")
    with open(run_script_path, "w") as run_script:
        run_script.write("#!/bin/bash\n")
        run_script.write(f"python {' '.join(cli_args)}\n")
    os.chmod(run_script_path, 0o755)  # Make the script executable

    # Log metadata
    metadata = {"timestamp": timestamp, "cli_args": cli_args, "git_commit": git_commit}
    with open(os.path.join(stage_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Experiment staged in: {stage_dir}")
    print(f"Metadata: {metadata}")
    return stage_dir


def push(path):
    exclude = ["--exclude", "events.out.tfevents*"]
    subprocess.check_call(["pdx", "cp"] + exclude + [path, "s3://screamcast-runs/"])


@command()
def pull(name: str):
    local_path = os.path.join(project_root, name)
    remote_path = f"s3://screamcast-runs/{name}/*"
    subprocess.check_call(["pdx", "cp"] + [remote_path, local_path])


@command(help="list the experiments in the remote bucket")
def listremote():
    files = subprocess.check_output(["pdx", "ls", "--show-fullpath", BUCKET])
    for line in files.decode().splitlines():
        print(line[len(BUCKET) :].rstrip("/"))


if __name__ == "__main__":
    args, cli_args = root_parser.parse_known_args()
    if args.command == "submit":
        stage_dir = stage(args.name, cli_args, overwrite=args.force)
        if not args.stage_only:
            submit(
                stage_dir,
                name=args.name,
                nodes=args.nodes,
                account=args.account,
                queue=getattr(args, "queue", None),
                constraint=getattr(args, "constraint", None),
                time=getattr(args, "time", None),
            )
    elif args.command == "resubmit":
        stage_dir = os.path.join(project_root, args.name)
        submit(
            stage_dir,
            name=args.name,
            nodes=args.nodes,
            account=args.account,
            queue=getattr(args, "queue", None),
            constraint=getattr(args, "constraint", None),
            time=getattr(args, "time", None),
        )
    elif args.command == "list":
        for p in os.listdir(project_root):
            print(p)
    elif args.command == "path":
        print(os.path.join(project_root, args.name))
    elif args.command == "logs":
        paths = glob.glob(os.path.join(project_root, args.name, "*.out"))
        command = ["tail", "-f"] if args.follow else ["cat"]
        subprocess.check_call(command + paths)
    elif args.command == "cd":
        path = os.path.join(project_root, args.name)
        subprocess.check_call(["bash"], cwd=path)
    elif args.command == "push":
        path = os.path.join(project_root, args.name)
        push(path)
    else:
        args.func(**{name: getattr(args, name) for name in _funcs[args.func]})
