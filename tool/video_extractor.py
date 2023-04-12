import pdb
import os
import sys
import cv2
import json
import argparse
import traceback
import numpy as np
from tqdm import tqdm
from glob import glob
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel

sys.path.insert(0, 'main')
sys.path.insert(0, 'data')
sys.path.insert(0, 'common')
from config import cfg
from model import get_model
from utils.preprocessing import process_bbox, generate_patch_image
from utils.human_models import smpl, smpl_x, mano, flame
from utils.vis import render_mesh, save_obj

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.ToTensor()


def get_common_videos(videos, jsons):
    video_names = [Path(video).stem for video in videos]
    json_names = [Path(json).stem for json in jsons]

    common_video_files = []
    common_json_files = []

    for video_name, video_file in zip(video_names, videos):
        for json_name, json_file in zip(json_names, jsons):
            if video_name == json_name:
                common_video_files.append(video_file)
                common_json_files.append(json_file)

    return common_video_files, common_json_files


def process_video(video, detection, model, output_folder, save_video=False):

    result = {}

    cap = cv2.VideoCapture(video)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    detections = json.load(open(detection, 'r'))

    assert len(detections) == num_frames, \
        'Number of frames in video and detection file do not match'

    if save_video:
        video_folder = os.path.join(output_folder, 'video')
        if not os.path.exists(video_folder):
            os.makedirs(video_folder, exist_ok=True)
        output_file_path = os.path.join(
            video_folder,
            '/'.join(video.split('/')[-5:])
        )
        video_writer_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            output_file_path,
            video_writer_fourcc,
            cap_fps,
            (int(width), int(height))
        )

    pbar = tqdm(range(int(300)),
                desc=f'Processing {video.split("/")[-1]}')
    # pbar = tqdm(range(int(num_frames)),
    #             desc=f'Processing {video.split("/")[-1]}')
    for ii in pbar:
        try:

            if detections[str(ii)] == []:
                result.update({ii: []})
                continue

            success, image = cap.read()

            if not success:
                result.update({ii: []})
                print('Error reading frame {} from video {}'.format(ii, video))
                continue

            bboxes = [box for box in detections[str(ii)] if box[-1] == 1.0]
            for bbox in bboxes:

                bbox = [int(xx) for xx in bbox[:4]]
                bbox = process_bbox(bbox, width, height)
                img, img2bb_trans, bb2img_trans = generate_patch_image(
                    image, bbox, 1.0, 0.0, False, cfg.input_img_shape)
                img = transform(img.astype(np.float32)) / 255.
                img = img.to(device).unsqueeze(0)

                inputs = {'img': img}
                targets = {}
                meta_info = {}
                with torch.no_grad():
                    outputs = model(inputs, targets, meta_info, 'test')

                if save_video:
                    mesh = outputs['smpl_mesh_cam'].detach().cpu().numpy()[0]

                    focal = [cfg.focal[0] / cfg.input_img_shape[1] * bbox[2],
                            cfg.focal[1] / cfg.input_img_shape[0] * bbox[3]]
                    principal_pt = [
                        cfg.princpt[0] / cfg.input_img_shape[1] * bbox[2] + bbox[0],
                        cfg.princpt[1] / cfg.input_img_shape[0] * bbox[3] + bbox[1]]
                    rendered_img = render_mesh(
                        image.copy(),
                        mesh,
                        smpl.face,
                        {'focal': focal, 'princpt': principal_pt}).astype(np.uint8)
                    video_writer.write(rendered_img)

        except Exception as e:
            print(traceback.format_exc())
            print('Error processing frame {} from video {}'.format(ii, video))
            continue

    cap.release()


def run_pose_inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str,
                        default='/research/iprobe/datastore/datasets/face/iiit-dronesurf/',
                        help='Path to video or folder of videos')
    parser.add_argument('--model', default='demo/body/snapshot_6_body.pth.tar',
                        type=str, help='Path to model weights')
    parser.add_argument('--detection_folder', default=None, required=True, type=str)
    parser.add_argument('--output_folder', default=None, required=True, type=str)
    parser.add_argument('--save_video', action='store_true', default=False)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder, exist_ok=True)

    cfg.set_args(args.gpu, 'body')
    model = get_model('test')
    model = DataParallel(model).cuda()
    model.load_state_dict(torch.load(args.model)['network'], strict=False)
    model.eval()

    videos = []
    if args.video.endswith('.mp4'):
        videos.append(args.video)
    else:
        glob_path = os.path.join(args.video, '**/*.mp4')
        videos = glob(glob_path, recursive=True)

    detection_files = glob(os.path.join(args.detection_folder, '**/*.json'),
                           recursive=True)

    common_videos, common_detections = get_common_videos(videos, detection_files)

    assert len(common_videos) == len(common_detections), \
        "Number of videos and detection files should be same"

    for ii, (video, detection) in enumerate(tqdm(zip(common_videos, common_detections),
                                                 desc="Processing videos")):

        assert video.split('/')[-1].split('.')[0] == \
            detection.split('/')[-1].split('.')[0], \
            "Video and detection file names should be same"

        try:
            process_video(video, detection, model, args.output_folder, args.save_video)
            pass
        except Exception as e:
            print("Error processing video {}: {}".format(video, e))


if __name__ == '__main__':
    run_pose_inference()

