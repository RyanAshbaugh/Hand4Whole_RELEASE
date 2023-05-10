import pdb
import os
import sys
import yaml
import wandb
from copy import copy
from pathlib import Path
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).absolute().parent.parent))

from data.dataset_dronesurf import DataDroneSurfSMPL
from common.utils.logging import get_git_commit
from common.utils.DroneSurfTest import DroneSurfTest

torch.random.manual_seed(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='configs/dronesurf-shape-test.yaml')
    parser.add_argument(
        '--dataset_root', type=str,
        default='/research/iprobe-ashbau12/datasets/iiit-dronesurf/'
        'pose_extraction/pretrained/labeled_parameters/'
    )
    parser.add_argument('--model_chekpoint_root', type=str,
                        default='/research/iprobe-ashbau12/repos/'
                        f'Pose2Pose/output/{{}}/checkpoints')
    parser.add_argument('--results_root', type=str,
                        default='/research/iprobe-ashbau12/repos/'
                        f'Pose2Pose/output/{{}}/results')
    args = parser.parse_args()
    configs = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    results_dir = args.results_root.format(configs['model']['model_type'])

    wandb.init(project='Pose2Pose', config=configs)
    wandb.config.update({'git_commit': get_git_commit()})

    device = torch.device('cuda:' + str(configs['gpu']))

    leaf_node = []
    if configs['test']['pose']:
        pass
    elif configs['test']['shape']:
        shape_dataset = DataDroneSurfSMPL(
            args.dataset_root,
            data_keys=['smpl_shape'],
            data_indices=[[0, 10]],
        )

        shape_dataset.df['features'] = shape_dataset.df['smpl_shape']
        test_dataset = copy(shape_dataset)
        leaf_node.append('shape')
    results_dir = os.path.join(results_dir, configs['tag'], '_'.join(leaf_node))

    dronesurf_test = DroneSurfTest(test_dataset, results_dir)

    dronesurf_test.perform_tests(
        configs['test']['probe_gallery_conditions'],
        configs['test']['probe_conditions'],
        configs['test']['gallery_conditions'],
    )

    pdb.set_trace()
