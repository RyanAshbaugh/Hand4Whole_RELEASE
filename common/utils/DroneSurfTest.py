import os
import pdb
import torch
import itertools
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))
from common.utils.metrics import DIR_FAR


class DroneSurfTest():
    def __init__(self, dataset, result_dir):
        self.dataset = dataset
        self.results_dir = result_dir

        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

        assert 'features' in dataset.df.columns, 'features not in dataset'

    def shape_test(self):
        pass

    def perform_tests(self, probe_gallery_conditions, probe_conditions,
                      gallery_conditions):

        for ii, condition in enumerate(probe_gallery_conditions):
            assert probe_conditions[ii] in self.dataset.df[condition].to_list(), \
                f'{probe_conditions[ii]} not in dataset[{condition}]'
            assert gallery_conditions[ii] in self.dataset.df[condition].to_list(), \
                f'{gallery_conditions[ii]} not in dataset[{condition}]'

        self.df = pd.DataFrame(
            columns=['probe_gallery_condition', 'probe_condition',
                     'gallery_condition', 'Rank1', 'EER'])
        completed_probe_conditions = []
        completed_gallery_conditions = []
        rank1s = []
        eers = []

        dataset = self.dataset.df.copy()
        dataset['features'] = dataset.features.apply(
            lambda x: np.asarray(x).mean(axis=0).astype(np.float32))

        pdb.set_trace()

        pbar = tqdm(total=len(probe_gallery_conditions))
        for ii, condition in enumerate(probe_gallery_conditions):

            probe_condition = probe_conditions[ii]
            gallery_condition = gallery_conditions[ii]

            probes = dataset[dataset[condition] == probe_condition]
            probe_features = torch.as_tensor(probes.features.to_list())

            # for gallery_angle in self.dataset.df['angles'].unique():
            pbar.set_description(f'probe_condition: {probe_condition}, '
                                 f'gallery_condition: {gallery_condition}')

            gallery = dataset[dataset[condition] == gallery_condition]
            gallery_features = torch.as_tensor(gallery.features.to_list())

            score_matrix = - torch.cdist(
                probe_features, gallery_features, p=2)

            labels = probes.subjects.to_numpy().reshape(-1, 1) == \
                gallery.subjects.to_numpy().reshape(-1, 1).T

            DIRs, FARs, thresholds, eer = DIR_FAR(
                score_matrix,
                labels,
                ranks=[1],
                get_equal_error_rate=True,
            )

            completed_probe_conditions.append(probe_condition)
            completed_gallery_conditions.append(gallery_condition)
            rank1s.append(DIRs[0])
            eers.append(eer)

            pbar.update(1)

        self.df['probe_condition'] = completed_probe_conditions
        self.df['gallery_condition'] = completed_gallery_conditions
        self.df['Rank1'] = rank1s
        self.df['EER'] = eers

        # filtered_df = df[df['probe_condition'] != df['gallery_angle']]
        # self.cross_view_df = filtered_df.groupby('probe_angle')[
        #     ['Rank1', 'EER']].mean()

        return self.df
