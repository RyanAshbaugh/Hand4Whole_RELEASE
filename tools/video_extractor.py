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
from utils.preprocessing import (
    process_bbox, generate_patch_image, process_video
)
from utils.human_models import smpl, smpl_x, mano, flame
from utils.vis import render_mesh, save_obj
from utils.preparation import (
    remove_data_parallel, keyword_filter, get_common_videos
)
from collections import OrderedDict

transform = transforms.ToTensor()


def run_pose_inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str,
                        default='/research/iprobe/datastore/datasets/face/iiit-dronesurf/',
                        help='Path to video or folder of videos')
    parser.add_argument('--model', default='demo/body/snapshot_6_body.pth.tar',
                        type=str, help='Path to model weights')
    parser.add_argument('--detection_folder', default=None, required=True, type=str)
    parser.add_argument('--output_folder', default=None, required=True, type=str)
    parser.add_argument('--filter_keyword', default='', type=str)
    parser.add_argument('--save_video', action='store_true', default=False)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch', type=int, default=64)
    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder, exist_ok=True)

    device = torch.device('cuda:' + args.gpu if torch.cuda.is_available() else 'cpu')

    cfg.set_args(args.gpu, 'body')
    model = get_model('test', device)
    model = model.to(device)
    state_dict = torch.load(args.model, map_location=torch.device('cpu'))['network']
    state_dict = remove_data_parallel(state_dict)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

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

    for ii, (video, detection) in enumerate(tqdm(zip(common_videos, common_detections),
                                                 desc="Processing videos",
                                                 total=len(common_videos))):

        assert video.split('/')[-1].split('.')[0] == \
            detection.split('/')[-1].split('.')[0], \
            "Video and detection file names should be same"

        parameter_folder = os.path.join(args.output_folder, 'parameters')
        output_file_path = os.path.join(
            parameter_folder,
            '/'.join(video.split('/')[-5:]).replace('.mp4', '.json')
        )
        if not os.path.exists('/'.join(output_file_path.split('/')[:-1])):
            os.makedirs('/'.join(output_file_path.split('/')[:-1]), exist_ok=True)

        try:
            if not os.path.exists(output_file_path):
                os.system(f"touch {output_file_path}")
                process_video(video, detection, model, output_file_path, device,
                              args.batch, args.save_video)
            else:
                print(f"\nSkipping {output_file_path}")

        except Exception as e:
            print(traceback.format_exc())
            print("Error processing video {}: {}".format(video, e))


if __name__ == '__main__':
    run_pose_inference()
