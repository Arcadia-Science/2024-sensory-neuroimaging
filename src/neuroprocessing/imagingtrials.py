import re, os, json
import numpy as np
import skimage.io as io

class ImagingTrials:
    def __init__(self, params):
        """
        Initialize the FileLoader with a list of files and their metadata.
        Each file's metadata should be in the format of a dictionary with keys
        for 'camera', 'rec_time', 'limb', 'injection_type', and 'iso'.

        Example params:
            params = {
                "downsample_factor": 8,
                "aligner_target_num_features": 700,
                "secs_before_stim": 60, # only process frames starting at X seconds before stimulus
                "preprocess_prefix": "aligned_downsampled_",
                "process_prefix": 'processed_',
                "s3fs_toplvl_path": "/Users/ilya_arcadia/arcadia-neuroimaging-pruritogens/Videos",
                "local_toplvl_path": "/Users/ilya_arcadia/Neuroimaging_local/Processed/Injections/",
                "load_from_s3": True,
                "save_to_s3": False,
                'crop_px' : 20,
                'bottom_percentile' : 5
    }
        """
        self.params = params
        self.base_path = params['s3fs_toplvl_path'] if params['load_from_s3'] else params['local_toplvl_path']
        self.exp_dirs, self.trial_dirs = self.collect_exps_and_trials()
        self.filtered_exp_dirs = self.exp_dirs
        self.filtered_trial_dirs = self.trial_dirs
        self.masks = []

    def collect_exps_and_trials(self):
        exps = []
        trials = []
        
        # List directories at the first level (exps)
        date_dirs = [d for d in os.listdir(self.base_path) if os.path.isdir(os.path.join(self.base_path, d))]
        
        for date in date_dirs:
            # Path to the date directory
            date_dir_path = os.path.join(self.base_path, date)
            
            # List directories at the second level (trials)
            exp_dirs = [d for d in os.listdir(date_dir_path) if os.path.isdir(os.path.join(date_dir_path, d))]
            
            # Append date and exp information
            for exp in exp_dirs:
                exps.append(date)
                trials.append(exp)
        
        return exps, trials

    def parse_filename(self, exp_dir, trial_dir):
            """Parse a filepath into its components, including the date and filename."""
            date = exp_dir
            
            tokens = trial_dir.split("_")
            camera = tokens[0]
            rec_time = tokens[1]
            limb = tokens[2]
            injection_type = tokens[3]
            iso = tokens[4]  # Joining the rest, assuming 'iso' may contain '_'
            
            return {"exp_dir": date, "camera": camera, "rec_time": rec_time, "limb": limb, "injection_type": injection_type, "iso": iso}

    def filter_exp_and_trial_dirs(self, **criteria):
        """Filter  trials based on criteria (wildcards allowed).
            Acceptable criteria are: "exp_dir", "camera", "rec_time", "limb", "injection_type", "iso"
        """
        loaded_exp_dirs = []
        loaded_trial_dirs = []
        for exp_dir, trial_dir in zip(self.exp_dirs, self.trial_dirs):
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
        Loads "mask.npy" files from the directories identified by loaded_files.
        """
        masks = []
        for e,t in zip(self.filtered_exp_dirs, self.filtered_trial_dirs):
            mask_path = os.path.join(self.base_path, e, t, "mask_" + self.params['process_prefix'] + t + ".npy")

            print(f"Loading mask file from: {mask_path}")

            mask = np.load(mask_path)
            masks.append(mask)
        self.masks = masks
        print(f"Loaded {len(masks)} masks.")
        return masks

    def load_traces(self):
        """
        Loads "traces.npy" files from the directories identified by loaded_files.
        """
        traces = []
        for e,t,m in zip(self.filtered_exp_dirs, self.filtered_trial_dirs, self.masks):
            processed_stack = io.imread(os.path.join(self.base_path, e, t, (self.params['process_prefix'] + t + ".tif")))
            trace = np.mean(processed_stack[:,m], axis=1)
            traces.append(trace)
        if len(traces) == 0:
            raise ValueError("No masks loaded. Please load masks first using `load_mask_files()`.")
        print(f"Loaded {len(traces)} traces.")
        return traces

    def get_sync_infos(self):
        """
        Loads "sync_info.json" files from the directories identified by loaded_files.
        """
        sync_infos = []
        for e,t in zip(self.filtered_exp_dirs, self.filtered_trial_dirs):
            sync_info_path = os.path.join(self.base_path, e, t, "sync_info.json")
            with open(sync_info_path, "r") as f:
                sync_info = json.load(f)
            sync_infos.append(sync_info)
        print(f"Loaded {len(sync_infos)} sync infos.")
        return sync_infos
