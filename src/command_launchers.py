import os
import subprocess
import time
import torch

def local_launcher(commands):
    """Launch commands serially on the local machine."""
    for cmd in commands:
        subprocess.call(cmd, shell=True)


def dummy_launcher(commands):
    """
    Doesn't run anything; instead, prints each command.
    Useful for testing.
    """
    for cmd in commands:
        print(f'Dummy launcher: {cmd}')


def multi_gpu_launcher(commands):
    """
    Launch commands on the local machine, using all GPUs in parallel.
    """
    print('WARNING: using multi_gpu_launcher.')
    n_gpus = torch.cuda.device_count()
    procs_by_gpu = [None] * n_gpus

    while len(commands) > 0:
        for gpu_idx in range(n_gpus):
            proc = procs_by_gpu[gpu_idx]
            if (proc is None) or (proc.poll() is not None):
                cmd = commands.pop(0)
                new_proc = subprocess.Popen(f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}', shell=True)
                procs_by_gpu[gpu_idx] = new_proc
                break
        time.sleep(1)

    for p in procs_by_gpu:
        if p is not None:
            p.wait()


def get_slurm_jobs(user=None):
    """Get the status of SLURM jobs for a user."""
    if user is None:
        user = os.getenv("USER")
    cmd = f"squeue -u {user} -o '%i %t %r %j'"
    try:
        output = subprocess.check_output(cmd, shell=True).decode()
        if output.strip() == "":
            return []
        jobs = []
        for line in output.split("\n")[1:]:  # Skip header
            if line.strip():
                job_id, state, reason, name = line.split()
                jobs.append({
                    "job_id": job_id,
                    "state": state,
                    "reason": reason,
                    "name": name
                })
        return jobs
    except subprocess.CalledProcessError:
        return []

def block_until_running(max_jobs=12):
    """Block until the number of queued and running jobs is below max_jobs."""
    while True:
        jobs = get_slurm_jobs()
        n_jobs = len(jobs)
        if n_jobs < max_jobs:
            break
        time.sleep(60)  # Check every minute

def slurm_launcher(commands, output_dirs, max_jobs=12):
    """Submit commands to SLURM using --wrap."""
    if not commands:
        return

    print(f"Submitting {len(commands)} jobs to SLURM with max {max_jobs} GPUs")
    # Default SLURM parameters
    slurm_params = [
        "--partition=",
        "--qos=",
        "--time=3-00:00:00",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=8",
        "--mem=32G",
        "--gres=gpu:1",
        "--exclude="
    ]

    # Submit jobs
    job_ids = []
    for i, (cmd, output_dir) in enumerate(zip(commands, output_dirs)):
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        # Create SLURM command with output/error files
        slurm_cmd = [
            "sbatch",
            f"--job-name=job_{i}/{len(commands)}_{max_jobs}",
            f"--output={output_dir}/job_{i}_%j.out",
            f"--error={output_dir}/job_{i}_%j.err",
            *slurm_params,
            "--wrap",
            f'"{cmd}"'
        ]

        # Submit job
        try:
            os.makedirs(output_dir, exist_ok=True)
            output = subprocess.check_output(
                " ".join(slurm_cmd), shell=True
            ).decode()
            job_id = output.split()[-1]
            job_ids.append(job_id)
        except subprocess.CalledProcessError as e:
            print(f"Failed to submit job {i}: {e}")
            continue

        print(f"Submitted job {i+1}/{len(commands)} with ID {job_id}")
        # Block if too many jobs
        block_until_running(max_jobs)

    return job_ids

REGISTRY = {
    'local': local_launcher,
    'dummy': dummy_launcher,
    'multi_gpu': multi_gpu_launcher,
    'slurm': slurm_launcher
}
