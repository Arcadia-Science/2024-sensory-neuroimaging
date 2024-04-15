import json
import os
import re

import numpy as np
import skimage.io as io


#@TODO: refactor into a single ImagingTrial class with a higher level ImagingTrialLoader
class ImagingTrialLoader:
    """
    Class to load imaging trials and their associated artifacts from a directory structure.
    """

    def __init__(self, params):
        """
        Initialize the ImagingTrialLoader with params metadata file
        """
        self.params = params
        self.base_path = params['s3fs_toplvl_path'] if params['load_from_s3'] else params['local_toplvl_path']
        self.exp_dirs, self.trial_dirs = self.collect_exps_and_trials()
        self.filtered_exp_dirs = self.exp_dirs
        self.filtered_trial_dirs = self.trial_dirs
        self.masks = []

    def __len__(self):
        return len(self.filtered_exp_dirs)

    def collect_exps_and_trials(self):
        """Collects all experiment and trial directories from the base path."""
        exps = []
        trials = []

        # List directories at the first level (exps)
        exp_dirs = [d for d in os.listdir(self.base_path) 
                    if os.path.isdir(os.path.join(self.base_path, d))]

        for exp in exp_dirs:
            # Path to the date directory
            exp_dir_path = os.path.join(self.base_path, exp)

            # List directories at the second level (trials)
            trial_dirs = [d for d in os.listdir(exp_dir_path) 
                          if os.path.isdir(os.path.join(exp_dir_path, d))]

            # Append date and exp information
            for trial in trial_dirs:
                exps.append(exp)
                trials.append(trial)

        return exps, trials

    def parse_filename(self, exp_dir, trial_dir):
        """Parse a filepath into its components."""
        camera, rec_time, limb, injection_type, *others = trial_dir.split("_")
        # If there are more tokens, append them to the remainder
        remainder = "_".join(others)

        return {"exp_dir": exp_dir, "camera": camera, "rec_time": rec_time, "limb": limb,
                "injection_type": injection_type, "remainder": remainder}

    def filter_exp_and_trial_dirs(self, **criteria):
        """Filter  trials based on criteria (wildcards allowed).
        """
        loaded_exp_dirs = []
        loaded_trial_dirs = []
        for exp_dir, trial_dir in zip(self.exp_dirs, self.trial_dirs, strict=True):
            file_metadata = self.parse_filename(exp_dir, trial_dir)
            if all(re.match(value, file_metadata[key]) for key, value in criteria.items()):
                loaded_trial_dirs.append(trial_dir)
                loaded_exp_dirs.append(exp_dir)
        self.filtered_exp_dirs = loaded_exp_dirs
        self.filtered_trial_dirs = loaded_trial_dirs
        print(f"Loaded {len(loaded_exp_dirs)} trials.")
        return loaded_exp_dirs, loaded_trial_dirs

    def load_mask_files(self):
        """
        Loads "mask.npy" files for all (optionally filtered) trials.
        """
        masks = []
        for e,t in zip(self.filtered_exp_dirs, self.filtered_trial_dirs, strict=True):
            mask_path = os.path.join(self.base_path,
                                     e,
                                     t,
                                     "mask_" + self.params['process_prefix'] + t + ".npy")

            print(f"Loading mask file from: {mask_path}")

            mask = np.load(mask_path)
            masks.append(mask)
        self.masks = masks
        print(f"Loaded {len(masks)} masks.")
        return masks

    def load_traces(self):
        """
        Loads "traces.npy" files for all (optinally filtered) trials.
        """
        traces = []
        for e,t,m in zip(self.filtered_exp_dirs, self.filtered_trial_dirs, self.masks, strict=True):
            processed_stack = io.imread(os.path.join(self.base_path, e, t, (self.params['process_prefix'] + t + ".tif")))
            trace = np.mean(processed_stack[:,m], axis=1)
            traces.append(trace)
        if len(traces) == 0:
            raise ValueError("No masks loaded. Please load masks first using `load_mask_files()`.")
        print(f"Loaded {len(traces)} traces.")
        return traces

    def plot_montage(self,
                     s_start:int,
                     s_end:int,
                     s_step=1,
                     montage_hw = (5,20),
                     montage_grid_shape=None):
        """
        Plots a montage of all (optinally filtered) trials.
        """
        import matplotlib.pyplot as plt
        from skimage.util import montage

        n_trials = len(self) # plot montage for each trial
        sync_infos = self.get_sync_infos()
        print(n_trials)
        f, axs = plt.subplots(nrows=n_trials,
                              ncols=1,
                              figsize=(montage_hw[0]*n_trials, montage_hw[1]), squeeze=False)
        for i, (e,t, si) in enumerate(zip(self.filtered_exp_dirs,
                                          self.filtered_trial_dirs,
                                          sync_infos,
                                          strict=True)):
            processed_stack = io.imread(os.path.join(self.base_path,
                                                     e,
                                                     t,
                                                     self.params['process_prefix'] + t + ".tif"))
            # get first frame closest to s_start
            downsampled_rate = (si['framerate_hz'] / self.params['downsample_factor'])
            frame_start, frame_end, frame_step = (int(downsampled_rate * s) for s in [s_start, s_end, s_step])
            n_frames = (frame_end - frame_start) // frame_step
            print(f"Frame start: {frame_start}, Frame end: {frame_end}, Frame step: {frame_step}, N frames: {n_frames}")
            montage_stack = processed_stack[frame_start:frame_end:frame_step,:,:]
            axs[i][0].imshow(montage(montage_stack,
                                 fill = 0,
                                 padding_width = 20,
                                 rescale_intensity=False,
                                 grid_shape= montage_grid_shape
                                 ))
            axs[i][0].set_title(f"{e} - {t}")
            axs[i][0].axis("off")

        plt.tight_layout()

        return f

    def get_sta_stacks(self, s_pre_stim = 1, s_post_stim = 5)->list:
        """
        Return a list of stimulus-triggered average stacks [n_trials x n_frames x h x w] for
        all (optinally filtered) trials.

        Inputs:
            s_pre_stim: int
                Number of seconds before stimulus onset to include in the stack.
            s_post_stim: int
                Number of seconds after stimulus onset to include in the stack.

        """

        sync_infos = self.get_sync_infos()

        sta_stacks = []
        for e,t, si in zip(self.filtered_exp_dirs, self.filtered_trial_dirs, sync_infos, strict=True):
            processed_stack = io.imread(os.path.join(self.base_path,
                                                     e,
                                                     t,
                                                     (self.params['process_prefix'] + t + ".tif")))

            downsampled_rate = (si['framerate_hz'] / self.params['downsample_factor'])
            frame_pre_stim, frame_post_stim = (int(downsampled_rate * s) for s in [s_pre_stim, s_post_stim])
            stim_onsets_downsampled = [int(sof // self.params['downsample_factor']) for sof in si['stim_onset_frame']]

            sta_stack = np.stack([processed_stack[sof-frame_pre_stim:sof+frame_post_stim, :,:] for sof in stim_onsets_downsampled[1:-1]])
            sta_stacks.append(sta_stack)

        return sta_stacks

    def get_sync_infos(self):
        """
        Loads "sync_info.json" files for all (optinally filtered) trials.
        """
        sync_infos = []
        for e,t in zip(self.filtered_exp_dirs, self.filtered_trial_dirs, strict=True):
            sync_info_path = os.path.join(self.base_path, e, t, "sync_info.json")
            with open(sync_info_path) as f:
                sync_info = json.load(f)
            sync_infos.append(sync_info)
        print(f"Loaded {len(sync_infos)} sync infos.")
        return sync_infos
