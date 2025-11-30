#!/usr/bin/env python3

import sys
sys.path.insert(0, '../src/')
import argparse
import os
import json
import hashlib
import copy
import shlex
import subprocess
import time
from project_paths import get_python_path, get_results_dir, get_selection_dir, get_output_dir, get_partition_name

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
    mem = 32  # Memory for figure generation
    # Default SLURM parameters
    partition = get_partition_name() or "<PARTITION_NAME>"
    slurm_params = [
        f"--partition={partition}",
        "--time=5-00:00:00",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=6",
        f"--mem={mem}G",
        "--gres=gpu:1",
    ]

    # Submit jobs
    job_ids = []
    for i, (cmd, output_dir) in enumerate(zip(commands, output_dirs)):
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        # Create SLURM command with output/error files
        slurm_cmd = [
            "sbatch",
            f"--job-name=figures_{i}/{len(commands)}_{max_jobs}",
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

def local_launcher(commands):
    """Launch commands serially on the local machine."""
    for cmd in commands:
        subprocess.call(cmd, shell=True)

REGISTRY = {
    'local': local_launcher,
    'slurm': slurm_launcher
}

class FigureJob:
    """Class to manage individual figure generation jobs."""
    NOT_LAUNCHED = 'Not launched'
    INCOMPLETE = 'Incomplete'
    DONE = 'Done'

    def __init__(self, figure_args, command_launcher='local'):
        # Define the keys that uniquely identify a figure generation job
        keys = ['dataset', 'metric', 'seed']
        # Add optional keys if they exist
        if 'train_idxs' in figure_args:
            keys.append('train_idxs')
        if 'test_idx' in figure_args:
            keys.append('test_idx')

        python_path = get_python_path()
        args_str = json.dumps({k: figure_args[k] for k in keys}, sort_keys=True)
        train_idxs_str = figure_args.get('train_idxs', 'default')
        test_idx_str = figure_args.get('test_idx', 'default')
        args_hash = f"{figure_args['dataset']}_{train_idxs_str}_{test_idx_str}_{figure_args['metric']}"

        self.output_dir = os.path.join(
            figure_args['output_dir'], 'figure_jobs', args_hash
        )
        figure_args['output_dir'] = self.output_dir

        self.figure_args = copy.deepcopy(figure_args)
        self.command_launcher = command_launcher

        # Build the command string
        # Change to repository root, then to scripts directory
        command = ["cd ..; cd scripts;"]
        command += [python_path, 'generate_correlation_plots.py']

        for k, v in sorted(self.figure_args.items()):
            if isinstance(v, list):
                v = ' '.join([str(v_) for v_ in v])
            elif isinstance(v, str):
                v = shlex.quote(v)
            elif isinstance(v, bool):
                if v:  # Only add flag if True
                    command.append(f'--{k}')
                continue  # Skip adding this argument if False
            command.append(f'--{k} {v}')
        self.command_str = ' '.join(command)

        # Check job state
        if os.path.exists(os.path.join(self.output_dir, 'done')):
            self.state = FigureJob.DONE
        elif os.path.exists(self.output_dir):
            self.state = FigureJob.INCOMPLETE
        else:
            self.state = FigureJob.NOT_LAUNCHED

    @staticmethod
    def launch(jobs, launcher_fn, max_slurm_jobs=1):
        """Launch the jobs."""
        print('Launching figure generation jobs...')
        jobs = jobs.copy()
        import numpy as np
        np.random.shuffle(jobs)
        print('Making job directories:')

        commands = [job.command_str for job in jobs]
        output_dirs = [job.output_dir for job in jobs]

        if launcher_fn == slurm_launcher:
            launcher_fn(commands, output_dirs, max_slurm_jobs)
        else:
            launcher_fn(commands)
        print(f'Launched {len(jobs)} figure generation jobs!')

    def __str__(self):
        job_info = (
            self.figure_args['dataset'],
            self.figure_args.get('train_idxs', 'default'),
            self.figure_args.get('test_idx', 'default'),
            self.figure_args['metric'],
            self.figure_args['seed']
        )
        return '{}: {} {}'.format(
            self.state,
            self.output_dir,
            job_info)

    def is_done(self):
        """Check if the job is complete."""
        return os.path.exists(os.path.join(self.output_dir, 'done'))

    def mark_done(self):
        """Mark the job as complete."""
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, 'done'), 'w') as f:
            f.write('')
        self.state = FigureJob.DONE

    @staticmethod
    def delete(jobs):
        """Delete incomplete jobs."""
        for job in jobs:
            if os.path.exists(job.output_dir):
                import shutil
                shutil.rmtree(job.output_dir)
            job.state = FigureJob.NOT_LAUNCHED

class ScatterPlotJob:
    """Class to manage individual scatter plot generation jobs."""
    NOT_LAUNCHED = 'Not launched'
    INCOMPLETE = 'Incomplete'
    DONE = 'Done'

    def __init__(self, scatter_args, command_launcher='local'):
        # Define the keys that uniquely identify a scatter plot job
        keys = ['dataset', 'train_idxs', 'test_idx']
        # Add optional keys if they exist
        if 'num_oods' in scatter_args:
            keys.append('num_oods')

        python_path = get_python_path()
        args_str = json.dumps({k: scatter_args[k] for k in keys}, sort_keys=True)
        train_idxs_str = scatter_args.get('train_idxs', 'default')
        test_idx_str = scatter_args.get('test_idx', 'default')
        args_hash = f"scatter_{scatter_args['dataset']}_{train_idxs_str}_{test_idx_str}"

        self.output_dir = os.path.join(
            scatter_args['save_dir'], 'scatter_jobs', args_hash
        )
        scatter_args['save_dir'] = self.output_dir

        self.scatter_args = copy.deepcopy(scatter_args)
        self.command_launcher = command_launcher

        # Build the command string
        # Change to repository root, then to scripts directory
        command = ["cd ..; cd scripts;"]
        command += [python_path, 'generate_scatter_plots.py']

        for k, v in sorted(self.scatter_args.items()):
            if isinstance(v, list):
                v = ' '.join([str(v_) for v_ in v])
            elif isinstance(v, str):
                v = shlex.quote(v)
            elif isinstance(v, bool):
                if v:  # Only add flag if True
                    command.append(f'--{k}')
                continue  # Skip adding this argument if False
            command.append(f'--{k} {v}')
        self.command_str = ' '.join(command)

        # Check job state
        if os.path.exists(os.path.join(self.output_dir, 'done')):
            self.state = ScatterPlotJob.DONE
        elif os.path.exists(self.output_dir):
            self.state = ScatterPlotJob.INCOMPLETE
        else:
            self.state = ScatterPlotJob.NOT_LAUNCHED

    @staticmethod
    def launch(jobs, launcher_fn, max_slurm_jobs=1):
        """Launch the jobs."""
        print('Launching scatter plot generation jobs...')
        jobs = jobs.copy()
        import numpy as np
        np.random.shuffle(jobs)
        print('Making job directories:')

        commands = [job.command_str for job in jobs]
        output_dirs = [job.output_dir for job in jobs]

        if launcher_fn == slurm_launcher:
            launcher_fn(commands, output_dirs, max_slurm_jobs)
        else:
            launcher_fn(commands)
        print(f'Launched {len(jobs)} scatter plot generation jobs!')

    def __str__(self):
        job_info = (
            self.scatter_args['dataset'],
            self.scatter_args.get('train_idxs', 'default'),
            self.scatter_args.get('test_idx', 'default'),
            self.scatter_args.get('num_oods', [250])
        )
        return '{}: {} {}'.format(
            self.state,
            self.output_dir,
            job_info)

    def is_done(self):
        """Check if the job is complete."""
        return os.path.exists(os.path.join(self.output_dir, 'done'))

    def mark_done(self):
        """Mark the job as complete."""
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, 'done'), 'w') as f:
            f.write('')
        self.state = ScatterPlotJob.DONE

    @staticmethod
    def delete(jobs):
        """Delete incomplete jobs."""
        for job in jobs:
            if os.path.exists(job.output_dir):
                import shutil
                shutil.rmtree(job.output_dir)
            job.state = ScatterPlotJob.NOT_LAUNCHED

def create_scatter_plot_jobs(force=False, metric='r', split_by_architecture=False, seed=0):
    """Create scatter plot generation jobs based on the run_figures.sh script."""
    # Set fixed directories (matching run_figures.sh)
    results_dir = get_results_dir() or "<RESULTS_DIR>"
    selection_dir = get_selection_dir() or "<SELECTION_DIR>"
    output_base = get_output_dir() or "<OUTPUT_DIR>"
    save_dir = f"{output_base}/scatter_plots_selection_camera_ready"

    def get_sample_sizes_for_dataset(dataset_name):
        """Get the sample sizes by scanning the Test Experiments directory for actual results"""
        import glob
        import re
        import os

        # Define the selection directory
        selection_dir_local = get_selection_dir() or "<SELECTION_DIR>"

        # Map dataset names to their directory patterns
        dataset_patterns = {
            'TerraIncognita': 'Test_TerraIncognita_*',
            'PACS': 'Test_PACS_*',
            'VLCS': 'Test_VLCS_*',
            'WILDSCamelyon': 'Test_WILDSCamelyon_*',
            'WILDSFMoW': 'Test_WILDSFMoW_*',
            'CXR_No_Finding': 'Test_CXR_No_Finding_*',
            'WILDSCivilComments': 'Test_WILDSCivilComments_*'
        }

        if dataset_name not in dataset_patterns:
            # Fallback to default sample sizes
            return [100, 250, 500, 1000, 2500, 5000, 10000]

        # Find all directories matching the pattern
        pattern = os.path.join(selection_dir_local, dataset_patterns[dataset_name])
        directories = glob.glob(pattern)

        # Extract sample sizes from directory names
        sample_sizes = []
        for directory in directories:
            # Extract the last number before '_r' in the directory name
            match = re.search(r'_(\d+)_r$', os.path.basename(directory))
            if match:
                sample_size = int(match.group(1))
                sample_sizes.append(sample_size)

        if sample_sizes:
            return sorted(list(set(sample_sizes)))  # Remove duplicates and sort

        # Fallback to default sample sizes
        return [100, 250, 500, 1000, 2500, 5000, 10000]

    jobs = []

    # Job 1: PACS and TerraIncognita datasets
    datasets = ["PACS", "TerraIncognita"]
    for dataset in datasets:
        ood_samples_list = get_sample_sizes_for_dataset(dataset)
        job_args = {
            'dataset': dataset,
            'train_idxs': '0-1-2',
            'test_idx': 3,
            'num_oods': ood_samples_list,
            'results_dir': results_dir,
            'selection_base': selection_dir,
            'save_dir': save_dir,
            'force': force,
            'metric': metric,
            'split_by_architecture': split_by_architecture,
            'seed': seed
        }
        jobs.append(ScatterPlotJob(job_args))

    # Job 2: WILDSCivilComments dataset
    # Note: WILDSCivilComments is not in OODSelect_experiments.py, so use default sample sizes
    ood_samples_list = [100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000]
    job_args = {
        'dataset': 'WILDSCivilComments',
        'train_idxs': '0-1-2-3-5-6',
        'test_idx': 4,
        'num_oods': ood_samples_list,
        'results_dir': results_dir,
        'selection_base': selection_dir,
        'save_dir': save_dir,
        'force': force,
        'metric': metric,
        'split_by_architecture': split_by_architecture,
        'seed': seed
    }
    jobs.append(ScatterPlotJob(job_args))

    # Job 3: CXR_No_Finding dataset
    ood_samples_list = get_sample_sizes_for_dataset('CXR_No_Finding')
    job_args = {
        'dataset': 'CXR_No_Finding',
        'train_idxs': '1-2-3-4',
        'test_idx': 0,
        'num_oods': ood_samples_list,
        'results_dir': results_dir,
        'selection_base': selection_dir,
        'save_dir': save_dir,
        'force': force,
        'metric': metric,
        'split_by_architecture': split_by_architecture,
        'seed': seed
    }
    jobs.append(ScatterPlotJob(job_args))

    # Job 4: WILDSCamelyon dataset (environments 3 and 4)
    ood_samples_list = get_sample_sizes_for_dataset('WILDSCamelyon')
    for env in [3, 4]:
        job_args = {
            'dataset': 'WILDSCamelyon',
            'train_idxs': '0-1-2',
            'test_idx': env,
            'num_oods': ood_samples_list,
            'results_dir': results_dir,
            'selection_base': selection_dir,
            'save_dir': save_dir,
            'force': force,
            'metric': metric,
            'split_by_architecture': split_by_architecture,
            'seed': seed
        }
        jobs.append(ScatterPlotJob(job_args))

    # Job 5: VLCS dataset
    ood_samples_list = get_sample_sizes_for_dataset('VLCS')
    job_args = {
        'dataset': 'VLCS',
        'train_idxs': '0-2-3',
        'test_idx': 1,
        'num_oods': ood_samples_list,
        'results_dir': results_dir,
        'selection_base': selection_dir,
        'save_dir': save_dir,
        'force': force,
        'metric': metric,
        'split_by_architecture': split_by_architecture,
        'seed': seed
    }
    jobs.append(ScatterPlotJob(job_args))

    return jobs

def create_figure_jobs(force=False, metric='r', split_by_architecture=False):
    """Create figure generation jobs based on the run_figures.sh script."""
    # Set fixed directories (matching run_figures.sh)
    results_dir = get_results_dir() or "<RESULTS_DIR>"
    selection_dir = get_selection_dir() or "<SELECTION_DIR>"
    output_base = get_output_dir() or "<OUTPUT_DIR>"

    if split_by_architecture:
        output_dir = f"{output_base}/figures_selection_camera_ready_arch"
        table_output_dir = f"{output_base}/tables_selection_camera_ready_arch"
    else:
        output_dir = f"{output_base}/figures_selection_camera_ready"
        table_output_dir = f"{output_base}/tables_selection_camera_ready"

    jobs = []

    # Job 1: PACS and TerraIncognita datasets
    datasets = ["PACS", "TerraIncognita"]
    for dataset in datasets:
        job_args = {
            'output_dir': output_dir,
            'dataset': dataset,
            'selection_dir': selection_dir,
            'results_dir': results_dir,
            'metric': metric,
            'table_output_dir': table_output_dir,
            'seed': 0,
            'force': force,
            'split_by_architecture': split_by_architecture
        }
        jobs.append(FigureJob(job_args))

    # Job 2: WILDSCivilComments dataset
    job_args = {
        'output_dir': output_dir,
        'dataset': 'WILDSCivilComments',
        'train_idxs': '0-1-2-3-5-6',
        'test_idx': '4',
        'metric': metric,
        'selection_dir': selection_dir,
        'results_dir': results_dir,
        'table_output_dir': table_output_dir,
        'seed': 0,
        'force': force,
        'split_by_architecture': split_by_architecture
    }
    jobs.append(FigureJob(job_args))

    # Job 3: CXR_No_Finding dataset
    job_args = {
        'output_dir': output_dir,
        'dataset': 'CXR_No_Finding',
        'train_idxs': '1-2-3-4',
        'test_idx': '0',
        'metric': metric,
        'selection_dir': selection_dir,
        'results_dir': results_dir,
        'table_output_dir': table_output_dir,
        'seed': 0,
        'force': force,
        'split_by_architecture': split_by_architecture
    }
    jobs.append(FigureJob(job_args))

    # Job 4: WILDSCamelyon dataset (environments 3 and 4)
    for env in [3, 4]:
        job_args = {
            'output_dir': output_dir,
            'dataset': 'WILDSCamelyon',
            'train_idxs': '0-1-2',
            'test_idx': str(env),
            'metric': metric,
            'selection_dir': selection_dir,
            'results_dir': results_dir,
            'table_output_dir': table_output_dir,
            'seed': 0,
            'force': force,
            'split_by_architecture': split_by_architecture
        }
        jobs.append(FigureJob(job_args))

    # Job 5: VLCS dataset
    job_args = {
        'output_dir': output_dir,
        'dataset': 'VLCS',
        'train_idxs': '0-2-3',
        'test_idx': '1',
        'metric': metric,
        'selection_dir': selection_dir,
        'results_dir': results_dir,
        'table_output_dir': table_output_dir,
        'seed': 0,
        'force': force,
        'split_by_architecture': split_by_architecture
    }
    jobs.append(FigureJob(job_args))

    return jobs

def ask_for_confirmation():
    response = input('Are you sure? (y/n) ')
    if not response.lower().strip()[:1] == "y":
        print('Nevermind!')
        exit(0)

def main():
    parser = argparse.ArgumentParser(description='Launch figure and scatter plot generation jobs using SLURM')
    parser.add_argument('command', choices=['launch', 'delete_incomplete', 'delete_all'])
    parser.add_argument('--skip_confirmation', action='store_true')
    parser.add_argument('--command_launcher', type=str, required=True, choices=['slurm', 'local'])
    parser.add_argument('--max_slurm_jobs', type=int, default=1)
    parser.add_argument('--redo_incomplete', action='store_true')
    parser.add_argument('--job_type', type=str, choices=['figures', 'scatter', 'both'], default='both',
                       help='Type of jobs to launch: figures, scatter plots, or both')
    parser.add_argument('--force', action='store_true',
                       help='Force regeneration of figures even if they already exist')
    parser.add_argument('--metric', type=str, default='r',
                       choices=['r', 'r2', 'spearman'],
                       help='Correlation metric type (default: r)')
    parser.add_argument('--split_by_architecture', action='store_true',
                       help='Split models by architecture families instead of randomly')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed for splitting (default: 0)')

    args = parser.parse_args()

    # Create jobs based on job_type
    all_jobs = []

    if args.job_type in ['figures', 'both']:
        figure_jobs = create_figure_jobs(force=args.force, metric=args.metric, split_by_architecture=args.split_by_architecture)
        all_jobs.extend(figure_jobs)
        print(f"Created {len(figure_jobs)} figure generation jobs")

    if args.job_type in ['scatter', 'both']:
        scatter_jobs = create_scatter_plot_jobs(force=args.force, metric=args.metric, split_by_architecture=args.split_by_architecture, seed=args.seed)
        all_jobs.extend(scatter_jobs)
        print(f"Created {len(scatter_jobs)} scatter plot generation jobs")

    # Print job status
    print("\nFIGURE JOBS DONE:")
    for job in all_jobs:
        if isinstance(job, FigureJob) and job.state == FigureJob.DONE:
            print(job)

    print("\nFIGURE JOBS INCOMPLETE:")
    for job in all_jobs:
        if isinstance(job, FigureJob) and job.state == FigureJob.INCOMPLETE:
            print(job)

    print("\nFIGURE JOBS NOT LAUNCHED:")
    for job in all_jobs:
        if isinstance(job, FigureJob) and job.state == FigureJob.NOT_LAUNCHED:
            print(job)

    print("\nSCATTER PLOT JOBS DONE:")
    for job in all_jobs:
        if isinstance(job, ScatterPlotJob) and job.state == ScatterPlotJob.DONE:
            print(job)

    print("\nSCATTER PLOT JOBS INCOMPLETE:")
    for job in all_jobs:
        if isinstance(job, ScatterPlotJob) and job.state == ScatterPlotJob.INCOMPLETE:
            print(job)

    print("\nSCATTER PLOT JOBS NOT LAUNCHED:")
    for job in all_jobs:
        if isinstance(job, ScatterPlotJob) and job.state == ScatterPlotJob.NOT_LAUNCHED:
            print(job)

    figure_jobs = [j for j in all_jobs if isinstance(j, FigureJob)]
    scatter_jobs = [j for j in all_jobs if isinstance(j, ScatterPlotJob)]

    print(f"\nTotal: {len(all_jobs)} jobs")
    print(f"Figure jobs: {len(figure_jobs)} total, {len([j for j in figure_jobs if j.state == FigureJob.DONE])} done, {len([j for j in figure_jobs if j.state == FigureJob.INCOMPLETE])} incomplete, {len([j for j in figure_jobs if j.state == FigureJob.NOT_LAUNCHED])} not launched")
    print(f"Scatter plot jobs: {len(scatter_jobs)} total, {len([j for j in scatter_jobs if j.state == ScatterPlotJob.DONE])} done, {len([j for j in scatter_jobs if j.state == ScatterPlotJob.INCOMPLETE])} incomplete, {len([j for j in scatter_jobs if j.state == ScatterPlotJob.NOT_LAUNCHED])} not launched")

    if args.command == 'launch':
        to_launch = []

        if args.job_type in ['figures', 'both']:
            figure_to_launch = [j for j in figure_jobs if j.state == FigureJob.NOT_LAUNCHED]
            if args.redo_incomplete:
                figure_to_launch = [j for j in figure_jobs if j.state == FigureJob.INCOMPLETE] + figure_to_launch
            to_launch.extend(figure_to_launch)

        if args.job_type in ['scatter', 'both']:
            scatter_to_launch = [j for j in scatter_jobs if j.state == ScatterPlotJob.NOT_LAUNCHED]
            if args.redo_incomplete:
                scatter_to_launch = [j for j in scatter_jobs if j.state == ScatterPlotJob.INCOMPLETE] + scatter_to_launch
            to_launch.extend(scatter_to_launch)

        print(f'About to launch {len(to_launch)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        launcher_fn = REGISTRY[args.command_launcher]

        # Launch figure jobs
        figure_jobs_to_launch = [j for j in to_launch if isinstance(j, FigureJob)]
        if figure_jobs_to_launch:
            FigureJob.launch(figure_jobs_to_launch, launcher_fn, args.max_slurm_jobs)

        # Launch scatter plot jobs
        scatter_jobs_to_launch = [j for j in to_launch if isinstance(j, ScatterPlotJob)]
        if scatter_jobs_to_launch:
            ScatterPlotJob.launch(scatter_jobs_to_launch, launcher_fn, args.max_slurm_jobs)

    elif args.command == 'delete_incomplete':
        to_delete = []

        if args.job_type in ['figures', 'both']:
            figure_to_delete = [j for j in figure_jobs if j.state == FigureJob.INCOMPLETE]
            to_delete.extend(figure_to_delete)

        if args.job_type in ['scatter', 'both']:
            scatter_to_delete = [j for j in scatter_jobs if j.state == ScatterPlotJob.INCOMPLETE]
            to_delete.extend(scatter_to_delete)

        print(f'About to delete {len(to_delete)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()

        # Delete figure jobs
        figure_jobs_to_delete = [j for j in to_delete if isinstance(j, FigureJob)]
        if figure_jobs_to_delete:
            FigureJob.delete(figure_jobs_to_delete)

        # Delete scatter plot jobs
        scatter_jobs_to_delete = [j for j in to_delete if isinstance(j, ScatterPlotJob)]
        if scatter_jobs_to_delete:
            ScatterPlotJob.delete(scatter_jobs_to_delete)

if __name__ == "__main__":
    main()
