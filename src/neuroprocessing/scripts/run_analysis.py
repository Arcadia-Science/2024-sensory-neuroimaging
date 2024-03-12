# run_analysis.py

# temporary file to run analysis

from pathlib import Path
import tqdm
from neuroprocessing.scripts.analysis import preprocess_and_process_trial

if __name__ == '__main__':
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

    '''
    preprocess_and_process_trial("2024-03-06", "Zyla_15min_LHL_salineinj_1pt75pctISO_1", params)
    preprocess_and_process_trial("2024-03-06", "Zyla_15min_RHL_salineinj_1pt25pctISO_1", params)
    preprocess_and_process_trial("2024-03-06", "Zyla_30min_LHL_27mMhistinj_1pt75pctISO_1", params)
    preprocess_and_process_trial("2024-03-06", "Zyla_30min_RHL_27mMhistinj_1pt25pctISO_1", params)
    '''

    date = "2024-03-06"
    exp_path = Path(params['local_toplvl_path']) / date
    # get all folder names in exp_path
    trial_dirs = [x for x in Path(exp_path).iterdir() if x.is_dir()]
    for trial_dir in tqdm.tqdm(trial_dirs):
        preprocess_and_process_trial(date, trial_dir.name, params)