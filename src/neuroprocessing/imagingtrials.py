import json
import os
import re

import numpy as np
import skimage.io as io


class ImagingTrial:
    """
    Class to represent an imaging trial and its associated artifacts.
    """

    def __init__(self, base_path, exp_dir, trial_dir):
        """
        Initialize the ImagingTrial with a base_path, exp_dir, trial_dir
        """
        self.exp_dir = exp_dir
        self.trial_dir = trial_dir
        self.base_path = base_path
        self.params = self._load_params()
        self.sync_info = self._get_sync_info()
        self.mask = None

    def __repr__(self):
        return f"ImagingTrial({self.exp_dir}, {self.trial_dir})"

    def __str__(self):
        return f"ImagingTrial {self.exp_dir} - {self.trial_dir}"

    def _load_params(self):
        """
        Load the params metadata file for the trial.
        """
        params_path = os.path.join(self.base_path, self.exp_dir, self.trial_dir, "params.json")
        if not os.path.exists(params_path):
            print('Warning: No params file found for trial: ', self.trial_dir)
            return None
        with open(params_path) as f:
            params = json.load(f)
        return params

    def _parse_filename(self):
        """Parse a filepath into its components."""
        tokens = self.trial_dir.split("_")
        camera = tokens[0]
        rec_time = tokens[1]
        limb = tokens[2]
        injection_type = tokens[3]
        # If there are more tokens, append them to the remainder
        remainder = "_".join(tokens[4:])

        return {"exp_dir": self.exp_dir, "trial_dir":self.trial_dir, "camera": camera,
                "rec_time": rec_time, "limb": limb,
                "injection_type": injection_type, "remainder": remainder
                }

    def _load_processed_stack(self):
        """
        Loads the processed stack
        """
        processed_stack_path = os.path.join(self.base_path,
                                            self.exp_dir,
                                            self.trial_dir,
                                            self.params['process_prefix'] + self.trial_dir + ".tif")
        processed_stack = io.imread(processed_stack_path)
        return processed_stack

    def _get_sync_info(self):
        """
        Loads the "sync_info.json" file for the trial.
        """
        sync_info_path = os.path.join(self.base_path, self.exp_dir, self.trial_dir, "sync_info.json")
        with open(sync_info_path) as f:
            sync_info = json.load(f)
        return sync_info

    def _s_to_frames(self, time_s: float):
        """
        Convert a time in seconds to a frame number in the downsampled stack.
        """
        downsampled_rate = (self.sync_info['framerate_hz'] / self.params['downsample_factor'])
        return int(downsampled_rate * time_s)

    def _frames_to_s(self, frame: int):
        """
        Convert a frame number in the downsampled stack to a time in seconds.
        """
        downsampled_rate = (self.sync_info['framerate_hz'] / self.params['downsample_factor'])
        return frame / downsampled_rate

    def _get_roi_mask(self, dims:tuple,
                      roi:dict):
        roi_mask = np.zeros(dims, dtype=bool)
        x, y = roi['center']
        w, h = roi['width'], roi['height']
        roi_mask[y-h//2:y+h//2, x-w//2:x+w//2] = True
        return roi_mask

    def load_trace(self, roi=None):
        """
        Returns time vector (s) and trace of whole-brain activity (if ROI not defined) and
          ROI-bounded activity (if ROI dict is defined) for the trial.

        Inputs:
            roi: dict
                Dictionary with keys 'center', 'width', 'height' specifying the ROI.
                'center' is a tuple (x,y) with the center of the ROI.
                'width' and 'height' are integers specifying the width and height of the ROI.

        Returns:
            t: np.ndarray
                Time vector in seconds.
            trace: np.ndarray
        """
        mask = self.load_mask()

        if roi is not None: # make mask from ROI
            mask = self._get_roi_mask(mask.shape, roi)

        processed_stack = self._load_processed_stack()
        trace = np.mean(processed_stack[:,mask], axis=1)
        t = (np.arange(0, len(trace))) / (self.sync_info['framerate_hz'] / self.params['downsample_factor']) - self.params['secs_before_stim']
        return t, trace

    def match_exp_criteria(self, **criteria):
        """Match trial against criteria."""
        file_metadata = self._parse_filename()
        match = all(re.match(value, file_metadata[key]) for key, value in criteria.items())
        return match

    def load_mask(self):
        """
        Loads the "mask.npy" file for the trial.
        """
        mask_path = os.path.join(self.base_path,
                                 self.exp_dir,
                                 self.trial_dir,
                                 "mask_" + self.params['process_prefix'] + self.trial_dir + ".npy")

        print(f"Loading mask file from: {mask_path}")

        mask = np.load(mask_path)
        return mask

    def plot_montage(self,
                     s_start:int,
                     s_end:int,
                     s_step=1,
                     montage_grid_shape=None,
                     ax=None,
                     **plot_kwargs) -> np.ndarray:
        """
        Returns and plots a montage of the trial.

        Inputs:
            s_start: int
                Start time of montage in seconds.
            s_end: int
                End time of montage in seconds.
            s_step: int
                Step size in seconds.
            montage_grid_shape: tuple
                Shape of the montage grid.
            ax: matplotlib axis
                Axis to plot the montage (optional).
            plot_kwargs: dict
                Keyword arguments for the plot.

        Returns:
            montage_stack: np.ndarray
                Montage of the trial [frames x h x w].
        """
        import matplotlib.pyplot as plt
        from skimage.util import montage

        processed_stack = self._load_processed_stack()
        # get first frame closest to s_start
        frame_start, frame_end, frame_step = (self._s_to_frames(s) for s in [s_start, s_end, s_step])
        n_frames = (frame_end - frame_start) // frame_step
        print(f"Frame start: {frame_start}, Frame end: {frame_end}, Frame step: {frame_step}, N frames: {n_frames}")
        montage_stack = processed_stack[frame_start:frame_end:frame_step,:,:]

        if ax is None:
            _,ax = plt.subplots(figsize=(10,10))
        ax.imshow(montage(montage_stack,
                           fill = 0,
                           padding_width = 20,
                           rescale_intensity=False,
                           grid_shape= montage_grid_shape
                           ),
                           **plot_kwargs)
        return montage_stack

    def plot_max_projection(self, ax=None, **plot_kwargs):
        """
        Plots the maximum projection of the processed stack.
        """
        import matplotlib.pyplot as plt
        processed_stack = self._load_processed_stack()
        max_projection = np.max(processed_stack, axis=0)

        if ax is None:
            _,ax = plt.subplots(figsize=(5,5))
        ax.imshow(max_projection, **plot_kwargs)
        return ax

    def get_sta_avg(self):
        """ Return stimulus-triggered average for all trials"""
        processed_stack = self._load_processed_stack()

        # assume that stim duration is stim_duration_frames / # stimulations / downsample_factor
        #@TODO: this may not be the case for all trials in the future e.g. if stimulation epochs have different durations
        stim_duration_frames = self.sync_info['stim_duration_frames'] // self.sync_info['Number of stimulations'] // self.params['downsample_factor']
        stim_onsets_downsampled = [int(sof // self.params['downsample_factor']) for sof in self.sync_info['stim_onset_frame']]

        #@TODO: this assumes that the baseline duration = stim duration, which is not true
        stack_base = np.stack(processed_stack[sof - stim_duration_frames:sof,:,:] for sof in stim_onsets_downsampled[1:-1])
        stack_stim = np.stack(processed_stack[sof:sof+stim_duration_frames,:,:] for sof in stim_onsets_downsampled[1:-1])

        stack_diff = (stack_stim - stack_base).mean(axis=(0,1))
        return stack_diff

    def get_sta_stack(self, s_pre_stim = 1,
                      s_post_stim = 5,
                      s_baseline = 0.5,
                      roi=None):
        """
        Return a stimulus-triggered average stack [n_trials x n_frames x h x w] for the trial.

        Inputs:
            s_pre_stim: int
                Number of seconds before stimulus onset to include in the stack.
            s_post_stim: int
                Number of seconds after stimulus onset to include in the stack.
            s_baseline: int
                Number of seconds before stimulus onset to use as baseline for background subtraction.
            roi: dict
                Dictionary with keys 'center', 'width', 'height' specifying the ROI.
                'center' is a tuple (x,y) with the center of the ROI.
                'width' and 'height' are integers specifying the width and height of the ROI.

        Returns:
            sta_df_stack: np.ndarray
                Stimulus-triggered, background-subtracted DF stack [n_trials x n_frames x h x w] for the trial.
            sta_dff_trace: np.ndarray
                Stimulus-triggered, background-subtracted DF/F stack for the ROI [n_frames].
        """

        processed_stack = self._load_processed_stack()

        frame_pre_stim, frame_post_stim = (self._s_to_frames(s) for s in [s_pre_stim, s_post_stim])
        n_frames = frame_pre_stim + frame_post_stim
        print(f"Frames pre-stim: {frame_pre_stim}, Frames post-stim: {frame_post_stim}")
        stim_onsets_downsampled = [int(sof // self.params['downsample_factor']) for sof in self.sync_info['stim_onset_frame']]

        if roi is None:
            sta_stack = np.stack([processed_stack[sof-frame_pre_stim:sof+frame_post_stim, :,:] for sof in stim_onsets_downsampled[1:-1]])
        else:
            roi_mask = self._get_roi_mask(processed_stack.shape[1:], roi)
            sta_stack = np.stack([processed_stack[sof-frame_pre_stim:sof+frame_post_stim, roi_mask] for sof in stim_onsets_downsampled[1:-1]])

            # reshape stack back to [n_trials x n_frames x h_roi x w_roi]
            sta_stack = sta_stack.reshape((sta_stack.shape[0], n_frames, roi['height'], roi['width']))

        frames_baseline = self._s_to_frames(s_baseline)
        sta_0 = sta_stack[:,:frames_baseline,:,:].mean(axis=1) # get first X frames and average them
        sta_df_stack = (sta_stack - sta_0[:,np.newaxis,:,:])

        sta_df_nan = sta_df_stack.copy()
        sta_df_nan[sta_df_stack==0] = np.nan

        sta_0_nan = sta_0.copy()
        sta_0_nan[sta_0==0] = np.nan
        sta_dff_trace = sta_df_nan / sta_0_nan[0,np.newaxis,:,:]
        sta_dff_trace = np.nanmean(sta_dff_trace, axis=(0,2,3))

        sta_t = np.array([self._frames_to_s(f) for f in range(n_frames)])
        return sta_df_stack, sta_t, sta_dff_trace


class ImagingTrialLoader:
    """
    Class to load imaging trials and their associated artifacts from a directory structure.
    """

    def __init__(self, base_path):
        self.base_path = base_path
        exp_dirs, trial_dirs = self.collect_exps_and_trials()

        self.trials = [ImagingTrial(base_path, e,t) for e,t in zip(exp_dirs, trial_dirs, strict=True)]
        print(f"Initialized with {len(self)} trials.")

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        return self.trials[idx]
    def __iter__(self):
        return iter(self.trials)
    def __next__(self):
        return next(self.trials)
    def __repr__(self) -> str:
        return f"ImagingTrialLoader({self.base_path})"

    def __iter__(self):
        return iter(self.trials)

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


    def filter(self, **criteria):
        """Filter  ImagingTrials based on criteria (wildcards allowed).

        Input criteria are specified in `ImagingTrial._parse_filename()`.

        Examples:

        ```
        trials.filter(exp_dir='2024-03-19'
                      limb='LHL'
        ```
        loads only trials from the left hind limb (LHL) from the experiment on March 19.

        ```
        trials.filter(exp_dir='2024-03-19'
                      limb='(L|R)HL$',
                      injection_type='.*inj'
        ```
        loads only trials ending with "inj" from the left and right hind limbs (LHL, RHL) from the
        experiment on March 19, but not "RHLstim", "LHLstim" etc.

        """


        trials_filt = [t for t in self.trials if t.match_exp_criteria(**criteria)]
        self.trials = trials_filt
        print(f"Filtered to {len(self)} trials.")
