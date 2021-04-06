import argparse
import json
import os.path as osp

import wandb


def parse_args():
    parser = argparse.ArgumentParser(description='Upload Json Log to wandb')
    parser.add_argument('log_dir', help='train log file path')
    args = parser.parse_args()
    return args

def upload(project, log_dir):

    init_kwargs = dict(
        project=project,
        name=log_dir,
        dir=osp.join(log_dir, 'wandb'))
    wandb.init(**init_kwargs, resume=True)
    with open(osp.join(log_dir, 'log.txt'), 'r') as f:
        for line in f.readlines():
            stats = json.loads(line)
            wandb.log(stats)


def main():
    args = parse_args()
    log_dir = args.log_dir
    upload('deit', log_dir)


if __name__ == '__main__':
    main()
