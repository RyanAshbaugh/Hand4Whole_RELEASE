from pathlib import Path
from collections import OrderedDict


def remove_data_parallel(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')  # remove 'module.'
        new_state_dict[name] = v
    return new_state_dict


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


def keyword_filter(videos, keyword):
    if len(keyword) > 0:
        videos = [video for video in videos if keyword in video]
    return videos
