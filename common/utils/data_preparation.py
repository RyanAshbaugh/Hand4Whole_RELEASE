import os
import os.path as osp
import yaml
import pickle
from glob import glob
import pandas as pd

def load_configs(config_path):

    with open(config_path) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    return conf


def get_dataset_jsons(dataset_root):
	return glob(osp.join(dataset_root, '**/*.json'), recursive=True)


def get_filename_info(df):

    subjects, angles, clothes, sequences = [], [], [], []
    for ii, row in df.iterrows():
        split_fname = row.filenames.split('/')
        subject = int(split_fname[-3])
        angle, clothes_set = split_fname[-2].split('_')
        sequence = split_fname[-1].split('.')[0]

        subjects.append(subject)
        angles.append(angle)
        clothes.append(clothes_set)
        sequences.append(sequence)

    return subjects, angles, clothes, sequences


def get_filename_df(dataset_root):

    df = pd.DataFrame(get_dataset_jsons(dataset_root), columns=['filenames'])
    subjects, angles, clothes, sequences = get_filename_info(df)
    df['subjects'] = subjects
    df['angles'] = angles
    df['clothes'] = clothes
    df['sequences'] = sequences

    return df


def load_or_create_file_df(dataset_root, df_path='data/preparation/df.pkl',
                           force_create=False):
    if not os.path.exists(df_path) or force_create:
        df = get_filename_df(dataset_root)

        if not force_create:
            pickle.dump(df, open(df_path, 'wb'))
    else:
        df = pickle.load(open(df_path, 'rb'))

    df = df.set_index('filenames').sort_index().reset_index()
    df.at[:, 'subjects'] = df.subjects - 1
    return df
