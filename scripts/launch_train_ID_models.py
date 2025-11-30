#!/usr/bin/env python
"""Launch batch training jobs."""

import sys
sys.path.insert(0, '../src/')
import argparse
import os
import shutil
from utils import Job
import domainbed_experiments
import command_launchers

def construct_train_args(config, dataset, algorithm, test_env, model_arch, transfer, weights, trial, hparams_idx):
    """Construct training arguments for a single job."""
    train_args = {
        # Data arguments
        'data_dir': config['data_dir'],
        'output_dir': config['output_dir'],
        'dataset': dataset,
        'test_envs': test_env,
        'holdout_fraction': config['holdout_fraction'],
        'uda_holdout_fraction': config['uda_holdout_fraction'],

        # Model arguments
        'algorithm': algorithm,
        'model_arch': model_arch,
        'transfer': transfer,
        'weights': weights,

        # Training arguments
        'trial_seed': trial,
        'hparams_seed': hparams_idx,

        # Logging arguments
        'log_backend': config['log_backend'],
    }
    return train_args

def make_args_list(experiment):
    return domainbed_experiments.get_hparams(experiment)

def ask_for_confirmation():
    response = input('Are you sure? (y/n) ')
    if not response.lower().strip()[:1] == "y":
        print('Nevermind!')
        exit(0)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('command', choices=['launch', 'delete_incomplete', 'delete_all'])
    parser.add_argument('--experiment', type=str, required = True)
    parser.add_argument('--skip_confirmation', action='store_true')
    parser.add_argument('--command_launcher', type=str, required=True, choices=['slurm', 'local'])
    parser.add_argument('--max_slurm_jobs', type=int, default = 1)
    parser.add_argument('--redo_incomplete', action='store_true')
    parser.add_argument('--overwrite_existing', action='store_true')
    parser.add_argument('--log_backend', type=str, choices=['wandb', 'csv', 'none'],
                        default='none')
    args = parser.parse_args()



    jobs = []

    for experiment in make_args_list(args.experiment):
        experiment['log_backend'] = args.log_backend

        # Construct training arguments
        train_args = construct_train_args(
            experiment,
            experiment['dataset'],
            experiment['algorithm'],
            experiment['test_envs'],
            experiment['model_arch'],
            experiment['transfer'],
            experiment['weights'],
            experiment['trial_seed'],
            experiment['hparams_seed']
        )

        # Create job
        job = Job(train_args,
                args.command_launcher)
        jobs.append(job)

    # Launch jobs
    print("JOBS DONE:")
    for job in jobs:
        if job.state == Job.DONE:
            print(job)

    print("\nJOBS INCOMPLETE:")
    for job in jobs:
        if job.state == Job.INCOMPLETE:
            print(job)

    print("\nJOBS NOT LAUNCHED:")
    for job in jobs:
        if job.state == Job.NOT_LAUNCHED:
            print(job)

    print("{} jobs: {} done, {} incomplete, {} not launched.".format(
        len(jobs),
        len([j for j in jobs if j.state == Job.DONE]),
        len([j for j in jobs if j.state == Job.INCOMPLETE]),
        len([j for j in jobs if j.state == Job.NOT_LAUNCHED]))
    )

    if args.command == 'launch':
        to_launch = [j for j in jobs if j.state == Job.NOT_LAUNCHED]
        if args.redo_incomplete:
            to_launch = [j for j in jobs if j.state == Job.INCOMPLETE] + to_launch
        if args.overwrite_existing:
            to_launch = [j for j in jobs if j.state == Job.DONE] + to_launch
        print(f'About to launch {len(to_launch)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        launcher_fn = command_launchers.REGISTRY[args.command_launcher]
        Job.launch(to_launch, launcher_fn, args.max_slurm_jobs)

    elif args.command == 'delete_incomplete':
        to_delete = [j for j in jobs if j.state == Job.INCOMPLETE]
        print(f'About to delete {len(to_delete)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        for job in to_delete:
            if os.path.exists(job.output_dir):
                print(f"Deleting {job.output_dir}")
                shutil.rmtree(job.output_dir)

    elif args.command == 'delete_all':
        print(f'About to delete all {len(jobs)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        for job in jobs:
            if os.path.exists(job.output_dir):
                print(f"Deleting {job.output_dir}")
                shutil.rmtree(job.output_dir)

if __name__ == "__main__":
    main()
