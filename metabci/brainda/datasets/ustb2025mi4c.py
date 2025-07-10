"""
Copyright (C) 2023 Qufu Normal University, Guangjin Liang
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

Author:  Guangjin Liang <3330635482@qq.com>
Update log: 2025-07-1 by Guangjin Liang <3330635482@qq.com>
"""

import os
import numpy as np
from typing import Union, Optional, Dict, List, cast
from pathlib import Path
import mne
from mne.io import Raw
from mne.channels import make_standard_montage
from .base import BaseDataset
from ..utils.download import mne_data_path



USTB2025MI4C_URL = "https://github.com/LiangXiaohan506/ustb2025mi4c/raw/refs/heads/main/Dataset/"


class USTB2025MI4C(BaseDataset):
    """USTB2025MI4C dataset.

    This dataset is a subset of the USTB2025 dataset, which is a public dataset
    for motor imagery classification. The dataset contains 4 classes of motor
    imagery tasks: left hand, right hand, both hands, and feet. The dataset
    contains 5 subjects, each subject has 5 sessions, and each session contains
    5 runs. Each run contains 10 trials, and each trial contains 4 seconds of
    data.

    The dataset is stored in a zip file, which contains 5 folders, each folder
    contains 5 files, each file contains 10 trials of data. The data is stored
    in the format of cnt, which is a format used by the MNE software.
    """

    _EVENTS = {
        "left_hand": (0, (0, 4)),
        "right_hand": (1, (0, 4)),
        "feet": (2, (0, 4)),
        "both_hands": (3, (0, 4))
    }

    _CHANNELS = [
        'FP1',
        'FP2',
        'FT7',
        'F7',
        'F3',
        'FZ',
        'F4',
        'F8',
        'FT8',
        'TP7',
        'FC3',
        'FCZ',
        'FC4',
        'TP8',
        'T7',
        'C3',
        'CZ',
        'C4',
        'T8',
        'CP3',
        'CP4',
        'M1',
        'M2',
        'P7',
        'P3',
        'PZ',
        'P4',
        'P8',
        'O1',
        'OZ',
        'O2'
    ]

    def __init__(self):
        super().__init__(
            dataset_code="ustb2025mi4c",
            subjects=list(range(1, 6)),
            events=self._EVENTS,
            channels=self._CHANNELS,
            srate=256,
            paradigm="imagery",
        )

    def data_path(
        self,
        subject: Union[str, int],
        path: Optional[Union[str, Path]] = None,
        force_update: bool = False,
        update_path: Optional[bool] = None,
        proxies: Optional[Dict[str, str]] = None,
        verbose: Optional[Union[bool, str, int]] = None,
    ) -> List[List[Union[str, Path]]]:

        if subject not in self.subjects:
            raise (ValueError("Invalid subject id"))
        subject = cast(int, subject)

        runs_set: List[Union[str, Path]] = []
        runs_fdt: List[Union[str, Path]] = []
        for run in range(1, 5):
            url_set = USTB2025MI4C_URL + "S0{:d}_run{:d}.set".format(subject, run)
            file_dest_set = mne_data_path(
                url_set,
                self.dataset_code,
                path=path,
                proxies=proxies,
                force_update=force_update,
                update_path=update_path,
            )
            runs_set.append(file_dest_set)

            url_fdt = USTB2025MI4C_URL + "S0{:d}_run{:d}.fdt".format(subject, run)
            file_fdt = mne_data_path(
                url_fdt,
                self.dataset_code,
                path=path,
                proxies=proxies,
                force_update=force_update,
                update_path=update_path,
            )
            runs_fdt.append(file_fdt)

        return runs_set, runs_fdt

    def _get_single_subject_data(
        self, subject: Union[str, int], verbose: Optional[Union[bool, str, int]] = None
    ) -> Dict[str, Dict[str, Raw]]:
        runs_set, _ = self.data_path(subject)
        montage = make_standard_montage("standard_1005")
        montage.rename_channels({ch_name: ch_name.upper() for ch_name in montage.ch_names})

        runs = dict()
        for irun, run_file in enumerate(runs_set):
            epochs = mne.read_epochs_eeglab(run_file)
            runs["run_{:d}".format(irun)] = epochs

        return runs

