"""Data loader for the PulseCam own recordings (follows the toolbox's "Adding a New Dataset" guide).

Folder layout (DATA_PATH):
    P01_still.mp4   P01_still_hr.csv   P01_move.mp4 ...
The recordings have a heart-rate reading but no pulse waveform, so labels are POS pseudo-labels:
set  USE_PSUEDO_PPG_LABEL: True  under the PREPROCESS block. Splits are by PERSON, never by video.
"""
import glob
import os

from dataset.data_loader.BaseLoader import BaseLoader
from dataset.data_loader.UBFCrPPGLoader import UBFCrPPGLoader


class OwnLoader(BaseLoader):
    def __init__(self, name, data_path, config_data, device=None):
        super().__init__(name, data_path, config_data, device)

    def get_raw_data(self, data_path):
        files = sorted(glob.glob(os.path.join(data_path, "*.mp4")))
        if not files:
            raise ValueError(self.dataset_name + " data paths empty!")
        return [{"index": os.path.basename(f)[:-4], "path": f} for f in files]

    def split_raw_data(self, data_dirs, begin, end):
        """Split the sorted list of PEOPLE (P01, P02, ...) so one person never lands in two splits."""
        people = sorted({d["index"].split("_")[0] for d in data_dirs})
        keep = set(people[int(begin * len(people)):int(end * len(people))])
        return [d for d in data_dirs if d["index"].split("_")[0] in keep]

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        if not config_preprocess.USE_PSUEDO_PPG_LABEL:
            raise ValueError("Own recordings have no pulse waveform: set USE_PSUEDO_PPG_LABEL: True")
        frames = UBFCrPPGLoader.read_video(data_dirs[i]["path"])
        bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)
        input_name_list, _ = self.save_multi_process(frames_clips, bvps_clips, data_dirs[i]["index"])
        file_list_dict[i] = input_name_list
