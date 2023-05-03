import pdb
import os
import sys
import json
import argparse
import traceback
from tqdm import tqdm
from glob import glob

sys.path.insert(0, 'common')
from utils.preprocessing import assign_labels
from utils.preparation import (
    get_common_videos, keyword_filter, remove_substrings_and_strip
)

import pandas as pd
import re


def run_detection_identity_assignment():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str,
                        default='/research/iprobe/datastore/datasets/face/iiit-dronesurf/',
                        help='Path to video or folder of videos')
    parser.add_argument('--parameters_folder', default=None, required=True, type=str)
    parser.add_argument('--output_folder', default=None, required=True, type=str)
    parser.add_argument('--filter_keyword', default='', type=str)
    args = parser.parse_args()

    labeled_parameter_output_folder = os.path.join(args.output_folder,
                                                   'labeled_parameters')
    if not os.path.exists(args.output_folder):
        os.makedirs(labeled_parameter_output_folder, exist_ok=True)

    videos = []
    if args.video.endswith('.mp4'):
        videos.append(args.video)
    else:
        glob_path = os.path.join(args.video, '**/*.mp4')
        videos = glob(glob_path, recursive=True)

    parameters_files = glob(os.path.join(args.parameters_folder, '**/*.json'),
                            recursive=True)

    videos = keyword_filter(videos, args.filter_keyword)

    common_videos, common_parameters = get_common_videos(
        videos, parameters_files)

    assert len(common_videos) == len(common_parameters), \
        "Number of videos and detection files should be same"

    substrings_to_remove = [',', '(', ')', '[', ']', 'Frame', 'frame']

    pattern = '|'.join(map(re.escape, substrings_to_remove))

    for ii, (video, parameter_file_path) in enumerate(
        tqdm(zip(common_videos, common_parameters),
             desc="Processing videos",
             total=len(common_videos))):

        assert video.split('/')[-1].split('.')[0] == \
            parameter_file_path.split('/')[-1].split('.')[0], \
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

        try:
            if not os.path.exists(output_file_path):
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                os.system(f"touch {output_file_path}")
                labeled_parameters = assign_labels(parameter_file_path, labels)
                with open(output_file_path, 'w') as f:
                    json.dump(labeled_parameters, f)
            else:
                print(f"\nSkipping {output_file_path}")

        except Exception as e:
            print(traceback.format_exc())
            print("Error processing parameters {}: {}".format(
                parameter_file_path, e))


if __name__ == '__main__':
    run_detection_identity_assignment()
