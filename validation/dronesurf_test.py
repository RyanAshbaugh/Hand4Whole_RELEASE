import pdb
import sys
from pathlib import Path
import argparse

sys.path.append(str(Path(__file__).absolute().parent.parent))

from data.dataset_dronesurf import DataDroneSurfSMPL


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_root', type=str,
        # default='/research/iprobe/datastore/datasets/face/iiit-dronesurf/')
        default='/research/iprobe-ashbau12/datasets/iiit-dronesurf/'
        'pose_extraction/pretrained/labeled_parameters/'
    )
    args = parser.parse_args()

    dataset = DataDroneSurfSMPL(args.dataset_root)

    pdb.set_trace()
