import os
import argparse
import traceback
from tqdm import tqdm
from glob import glob
import copy

import sys
sys.path.insert(0, 'main')
sys.path.insert(0, 'data')
sys.path.insert(0, 'common')
from utils.preprocessing import visualize_labeled_video
from utils.preparation import keyword_filter, get_common_videos
from config import cfg

import torch
from utils.human_models import smpl


def run_parameter_visualization():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str,
                        default='/research/iprobe/datastore/datasets/face/iiit-dronesurf/',
                        help='Path to video or folder of videos')
    parser.add_argument('--parameter_folder', default=None, required=True, type=str)
    parser.add_argument('--output_folder', default=None, required=True, type=str)
    parser.add_argument('--filter_keyword', default='', type=str)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--batch', default=64, type=int)
    args = parser.parse_args()

    device = torch.device('cuda:' + args.gpu)

    cfg.set_args('0', 'body')

    smpl_layer = copy.deepcopy(smpl.layer['neutral']).to(device)

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder, exist_ok=True)

    videos = []
    if args.video.endswith('.mp4'):
        videos.append(args.video)
    else:
        glob_path = os.path.join(args.video, '**/*.mp4')
        videos = glob(glob_path, recursive=True)

    parameter_files = glob(os.path.join(args.parameter_folder, '**/*.json'),
                           recursive=True)

    videos = keyword_filter(videos, args.filter_keyword)

    common_videos, common_parameters = get_common_videos(videos, parameter_files)


    for ii, (video, parameter_file_path) in enumerate(
        tqdm(zip(common_videos, common_parameters),
             desc="Processing videos",
             total=len(common_videos))):

        assert video.split('/')[-1].split('.')[0] == \
            parameter_file_path.split('/')[-1].split('.')[0], \
            "Video and detection file names should be same"

        labeled_video_folder = os.path.join(args.output_folder, 'labeled_video')
        labeled_video_path = os.path.join(
            labeled_video_folder,
            '/'.join(video.split('/')[-5:])
        )
        if not os.path.exists('/'.join(labeled_video_path.split('/')[:-1])):
            os.makedirs('/'.join(labeled_video_path.split('/')[:-1]), exist_ok=True)

        try:
            if not os.path.exists(labeled_video_path):
                visualize_labeled_video(
                    video,
                    parameter_file_path,
                    labeled_video_path,
                    smpl_layer,
                    cfg,
                    device,
                    args.batch
                )
            else:
                print(f"\nSkipping {labeled_video_path}")

        except Exception as e:
            print(traceback.format_exc())
            print("Error processing video {}: {}".format(video, e))


if __name__ == '__main__':
    run_parameter_visualization()
