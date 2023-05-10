import pdb
import os
import json
import random
import numpy as np
import pandas as pd
from glob import glob

import torch
import torch.utils.data as tordata


def get_filename_info(df):

    active_random, time_of_day, locations, subjects, fnames = [], [], [], [], []
    for ii, row in df.iterrows():
        split_fname = row.data_path.split('/')

        fnames.append(split_fname[-1].split('.')[0])
        subjects.append([int(xx) for xx in split_fname[-2].split(',')])
        locations.append(int(split_fname[-3][-1]))
        time_of_day.append(split_fname[-4].lower())
        active_random.append(split_fname[-5].replace('Tracking', '')
                             .replace('Activity', '').lower())

    return active_random, time_of_day, locations, subjects, fnames


def get_positive_indices(df, label):
    return list(np.where(df['subjects'] == label)[0])


def get_negative_indices(df, label):
    return list(np.where(df['subjects'] != label)[0])


def select_entries(column, num_sequence_frames):
    return column[:num_sequence_frames]


class DataDroneSurfSMPL(tordata.Dataset):
    def __init__(self, dataset_root, triplet=False, num_sequence_frames=15,
                 data_keys=['smpl_shape'], data_indices=[[0, 10]],
                 transform=None):

        json_columns = ['cam_trans', 'smpl_pose', 'smpl_shape', 'bbox',
                        'identity', 'frame']
        filenames = glob(os.path.join(dataset_root, '**/*.json'),
                         recursive=True)
        self.file_df = pd.DataFrame(filenames, columns=['data_path'])
        active_random, time_of_day, locations, subjects, fnames = \
            get_filename_info(self.file_df)
        self.file_df['active_random'] = active_random
        self.file_df['time_of_day'] = time_of_day
        self.file_df['location'] = locations
        self.file_df['subjects'] = [(np.array(xx) - 1).tolist() for xx in subjects]
        self.file_df['fname'] = fnames

        json_dataframes = []
        for ii, row in self.file_df.iterrows():
            data = []
            json_data = json.load(open(row.data_path, 'r'))
            for key, values in json_data.items():
                for value in values:
                    value['frame'] = key
                    data.append(value)

            json_df = pd.json_normalize(data)
            json_df['active_random'] = row.active_random
            json_df['time_of_day'] = row.time_of_day
            json_df['location'] = row.location
            json_df['fname'] = row.fname

            json_dataframes.append(json_df)

        self.df = pd.concat(json_dataframes, ignore_index=True)
        self.df['identity'] = (np.array(self.df['identity'].astype(int)) -
                               1).tolist()
        self.df.rename(columns={'identity': 'subject'}, inplace=True)
        self.df = self.df.groupby(
            ['fname', 'subject']).agg(
                {'active_random': 'first', 'time_of_day': 'first',
                 'location': 'first', 'cam_trans': list, 'smpl_pose': list,
                 'smpl_shape': list, 'bbox': list, 'frame': list}
            ).reset_index()

        self.data_keys = data_keys
        self.data_indices = data_indices
        self.num_sequence_frames = num_sequence_frames

        self.df.reset_index(inplace=True, drop=True)
        self.triplet = triplet
        self.neg_indices = {}
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        # if self.df.loc[idx, 'smpl_params'] == '':
        #     self.df.at[idx, 'smpl_params'] = self.load_sequence(idx)
        data = self.df.loc[idx, self.data_keys[0]]
        label_int = self.df.loc[idx, 'subject']

        if self.transform:
            data = self.transform(data)

        if not self.triplet:
            return data, label_int
        else:
            pos_indices = get_positive_indices(self.df, label_int)
            if self.neg_indices.get(label_int) is None:
                self.neg_indices[label_int] = self.df.index.to_numpy()[
                    ~self.df.index.isin(pos_indices)]

            pos_index = idx
            while pos_index == idx:
                pos_index = random.choice(pos_indices)
            neg_index = random.choice(self.neg_indices[label_int])

            if self.df.loc[pos_index, 'smpl_params'] == '':
                self.df.at[pos_index, 'smpl_params'] = self.load_sequence(pos_index)
            pos_data = self.df.loc[pos_index, 'smpl_params']
            pos_label = torch.as_tensor(self.df.loc[pos_index, 'subjects'])

            if self.df.loc[neg_index, 'smpl_params'] == '':
                self.df.at[neg_index, 'smpl_params'] = self.load_sequence(neg_index)
            neg_data = self.df.loc[neg_index, 'smpl_params']
            neg_label = torch.as_tensor(self.df.loc[neg_index, 'subjects'])

            if self.transform:
                pos_data = self.transform(pos_data)
                neg_data = self.transform(neg_data)

            return data, pos_data, neg_data, label_int, pos_label, neg_label


    def load_sequence(self, index):
        num_features = np.sum([x[1] - x[0] for x in self.data_indices])

        feature_counter = 0
        sequence = torch.zeros(num_features, self.num_sequence_frames)
        for key, indices in zip(self.data_keys, self.data_indices):
            num_key_features = indices[1] - indices[0]

            for ii, fname in enumerate(self.df.loc[index, 'selected_filenames']):
                data_list = json.load(open(fname, 'r'))[key]

                if any(isinstance(el, list) for el in data_list):
                    data_list = [item for sublist in data_list for item in sublist]

                data = data_list[indices[0]:indices[1]]
                end_index = feature_counter + num_key_features
                sequence[feature_counter:end_index, ii] = \
                    torch.as_tensor(data)

            feature_counter += num_key_features
        return sequence
