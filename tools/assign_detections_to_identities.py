import pdb
import os
import sys
import cv2
import json
import copy
import argparse
import traceback
import numpy as np
from tqdm import tqdm
from glob import glob
from pathlib import Path

import torch
import torchvision.transforms as transforms

sys.path.insert(0, 'main')
sys.path.insert(0, 'data')
sys.path.insert(0, 'common')
from config import cfg
from model import get_model
from utils.preprocessing import assign_labels
from utils.preparation import (
    get_common_videos, keyword_filter, remove_substrings_and_strip
)
from utils.human_models import smpl, smpl_x, mano, flame
from utils.vis import render_mesh, save_obj
from collections import OrderedDict

import pandas as pd
import re

transform = transforms.ToTensor()


def run_pose_inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str,
                        default='/research/iprobe/datastore/datasets/face/iiit-dronesurf/',
                        help='Path to video or folder of videos')
    parser.add_argument('--detection_folder', default=None, required=True, type=str)
    parser.add_argument('--output_folder', default=None, required=True, type=str)
    parser.add_argument('--filter_keyword', default='', type=str)
    parser.add_argument('--save_video', action='store_true', default=False)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch', type=int, default=64)
    args = parser.parse_args()

    labeled_parameter_output_folder = os.path.join(args.output_folder,
                                                   'labeled_parameters')
    if not os.path.exists(args.output_folder):
        os.makedirs(labeled_parameter_output_folder, exist_ok=True)

    device = torch.device('cuda:' + args.gpu if torch.cuda.is_available() else 'cpu')

    videos = []
    if args.video.endswith('.mp4'):
        videos.append(args.video)
    else:
        glob_path = os.path.join(args.video, '**/*.mp4')
        videos = glob(glob_path, recursive=True)

    detection_files = glob(os.path.join(args.detection_folder, '**/*.json'),
                           recursive=True)

    videos = keyword_filter(videos, args.filter_keyword)

    common_videos, common_detections = get_common_videos(videos, detection_files)

    assert len(common_videos) == len(common_detections), \
        "Number of videos and detection files should be same"

    substrings_to_remove = [',', '(', ')', 'Frame', 'frame']
    pattern = '|'.join(map(re.escape, substrings_to_remove))

    for ii, (video, detection) in enumerate(tqdm(zip(common_videos, common_detections),
                                                 desc="Processing videos",
                                                 total=len(common_videos))):

        assert video.split('/')[-1].split('.')[0] == \
            detection.split('/')[-1].split('.')[0], \
            "Video and detection file names should be same"

        identities = video.split('/')[-2].split(',')

        column_names = ['frame', 'x1', 'y1', 'x2', 'y2', 'identity']
        labels = pd.DataFrame(columns=column_names)
        for identity in identities:

            label_file = os.path.join(
                video.replace('.mp4', ''),
                f'Person{identity}',
                'coordinates.txt'
            )
            identity_labels = pd.read_csv(label_file, sep=' ', header=None)
            identity_labels['identity'] = identity
            identity_labels.columns = column_names

            labels = pd.concat([labels, identity_labels], ignore_index=True)

        labels = labels.applymap(
            lambda x: remove_substrings_and_strip(x, pattern))

        output_file_path = os.path.join(
            labeled_parameter_output_folder,
            '/'.join(video.split('/')[-5:]).replace('.mp4', '.json')
        )

        pdb.set_trace()

        try:
            if not os.path.exists(output_file_path):
                os.system(f"touch {output_file_path}")
                assign_labels(detection, output_file_path)
            else:
                print(f"\nSkipping {output_file_path}")

        except Exception as e:
            print(traceback.format_exc())
            print("Error processing video {}: {}".format(video, e))


if __name__ == '__main__':
    run_pose_inference()
