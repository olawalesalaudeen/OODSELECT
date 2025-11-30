#!/usr/bin/env python
"""Launch batch OOD selection jobs."""

import sys
sys.path.insert(0, '../src/')
import argparse
import os
import shutil
import OODSelect_experiments
import command_launchers
from OODSelect_utils import TestJob
from project_paths import get_python_path

def construct_test_args(config, split_by_architecture=False, seed=0):
    test_args = {
        'dataset': config['dataset'],
        'results_dir': config['results_dir'],
        'num_OOD_samples': config['num_OOD_samples'],
        'loss_type': config['loss_type'],
        'num_epochs': config['num_epochs'],
        'num_trials': config['num_trials'],
        'train_idxs': config['train_idxs'],
        'test_idx': config['test_idx'],
        'output_dir': config['output_dir'],
        'split_by_architecture': split_by_architecture,
        'seed': seed,
    }
    if 'max_total_samples' in config:
        test_args['max_total_samples'] = config['max_total_samples']
    return test_args

def make_args_list(experiment):
    return OODSelect_experiments.get_hparams(experiment)

def ask_for_confirmation():
    response = input('Are you sure? (y/n) ')
    if not response.lower().strip()[:1] == "y":
        print('Nevermind!')
        exit(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['launch', 'delete_incomplete', 'delete_all'])
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--skip_confirmation', action='store_true')
    parser.add_argument('--command_launcher', type=str, required=True, choices=['slurm', 'local'])
    parser.add_argument('--max_slurm_jobs', type=int, default = 1)
    parser.add_argument('--redo_incomplete', action='store_true')
    parser.add_argument('--log_backend', type=str, choices=['wandb', 'csv', 'none'],
        default='none')
    parser.add_argument('--split_by_architecture', action='store_true',
        help='Split models by architecture families instead of randomly')
    parser.add_argument('--seed', type=int, default=0,
        help='Random seed for splitting (default: 0)')

    args = parser.parse_args()

    test_args_list = make_args_list(args.experiment)
    for test_args in test_args_list:
        print(test_args)

    jobs = []
    python_path = get_python_path()  # Already returns 'python' as default if not configured

    for experiment in test_args_list:
        test_args = construct_test_args(experiment,
                                      split_by_architecture=args.split_by_architecture,
                                      seed=args.seed)

        # Use python_path from project_paths if not specified in experiment config
        job_python_path = experiment.get('python_path') or python_path
        job = TestJob(test_args,
                job_python_path,
                args.command_launcher)
        jobs.append(job)

    for job in jobs:
        print(job)

    print("{} jobs: {} done, {} incomplete, {} not launched.".format(
        len(jobs),
        len([j for j in jobs if j.state == TestJob.DONE]),
        len([j for j in jobs if j.state == TestJob.INCOMPLETE]),
        len([j for j in jobs if j.state == TestJob.NOT_LAUNCHED])
    ))

    if args.command == 'launch':
        to_launch = [j for j in jobs if j.state == TestJob.NOT_LAUNCHED]
        if args.redo_incomplete:
            to_launch = [j for j in jobs if j.state == TestJob.INCOMPLETE] + to_launch
        print(f'About to launch {len(to_launch)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        launcher_fn = command_launchers.REGISTRY[args.command_launcher]
        TestJob.launch(to_launch, launcher_fn, args.max_slurm_jobs)

    elif args.command == 'delete_incomplete':
        to_delete = [j for j in jobs if j.state == TestJob.INCOMPLETE]
        print(f'About to delete {len(to_delete)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        for job in to_delete:
            if os.path.exists(job.test_args['output_dir']):
                print(f"Deleting {job.test_args['output_dir']}")
                shutil.rmtree(job.test_args['output_dir'])

    elif args.command == 'delete_all':
        print(f'About to delete all {len(jobs)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        for job in jobs:
            if os.path.exists(job.test_args['output_dir']):
                print(f"Deleting {job.test_args['output_dir']}")
                shutil.rmtree(job.test_args['output_dir'])

if __name__ == "__main__":
    main()
