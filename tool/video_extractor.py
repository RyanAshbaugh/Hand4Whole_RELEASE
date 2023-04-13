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


def x1y1x2y2_to_xywh(bbox):
    bbox = [int(xx) for xx in bbox]
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]


def square_bbox(bbox):
    bbox = [int(xx) for xx in bbox]
    x1, y1, w, h = bbox
    if w > h:
        y1 = max(0, y1 - int((w - h) / 2))
        h = w
    elif h > w:
        x1 = max(0, x1 - int((h - w) / 2))
        w = h
    return [x1, y1, w, h]


def pad_bbox(bbox, width, height, padding=0.5):
    bbox = [int(xx) for xx in bbox]
    x1, y1, w, h = bbox
    x2 = x1 + w
    y2 = y1 + h
    x1 = max(0, x1 - int(w * padding))
    y1 = max(0, y1 - int(h * padding))
    x2 = min(width, x2 + int(w * padding))
    y2 = min(height, y2 + int(h * padding))
    return [x1, y1, x2 - x1, y2 - y1]


def process_video(video, detection, model, output_folder, save_video=False, padding=0.2):

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
        output_file_path = os.path.join(
            video_folder,
            '/'.join(video.split('/')[-5:])
        )
        if not os.path.exists('/'.join(output_file_path.split('/')[:-1])):
            os.makedirs('/'.join(output_file_path.split('/')[:-1]), exist_ok=True)

        video_writer_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            output_file_path,
            video_writer_fourcc,
            cap_fps,
            (int(width), int(height))
        )

    # pbar = tqdm(range(int(300)),
    #             desc=f'Processing {video.split("/")[-1]}')
    pbar = tqdm(range(int(num_frames)),
                desc=f'Processing {video.split("/")[-1]}')
    for ii in pbar:
        try:

            success, image = cap.read()

            if detections[str(ii)] == []:
                result.update({ii: []})
                video_writer.write(image)
                continue

            if not success:
                result.update({ii: []})
                print('Error reading frame {} from video {}'.format(ii, video))
                continue

            outputs = []
            bboxes = [process_bbox(x1y1x2y2_to_xywh(box[:4]), width, height)
                      for box in detections[str(ii)] if box[-1] == 1.0]
            # bboxes = [square_bbox(x1y1x2y2_to_xywh(box[:4])) for
            #           box in detections[str(ii)] if box[-1] == 1.0]
            # bboxes = [pad_bbox(box, width, height, padding) for box in bboxes]
            for bbox in bboxes:

                # bbox = process_bbox(bbox, width, height)
                img, img2bb_trans, bb2img_trans = generate_patch_image(
                    image, bbox, 1.0, 0.0, False, cfg.input_img_shape)
                img = transform(img.astype(np.float32)) / 255.
                img = img.to(device).unsqueeze(0)

                inputs = {'img': img}
                targets = {}
                meta_info = {}
                with torch.no_grad():
                    outputs.append(model(inputs, targets, meta_info, 'test'))

            if save_video:
                rendered_img = image.copy()
                for output, bbox in zip(outputs, bboxes):
                    mesh = output['smpl_mesh_cam'].detach().cpu().numpy()[0]

                    focal = [cfg.focal[0] / cfg.input_img_shape[1] * bbox[2],
                             cfg.focal[1] / cfg.input_img_shape[0] * bbox[3]]
                    principal_pt = [
                        cfg.princpt[0] / cfg.input_img_shape[1] * bbox[2] + bbox[0],
                        cfg.princpt[1] / cfg.input_img_shape[0] * bbox[3] + bbox[1]]
                    rendered_img = cv2.rectangle(
                        rendered_img,
                        (int(bbox[0]), int(bbox[1])),
                        (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                        (0, 255, 0), 2)

                    rendered_img = render_mesh(
                        rendered_img,
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

