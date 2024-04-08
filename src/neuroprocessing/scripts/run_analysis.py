"""# run_analysis.py

Runs analysis pipeline (preprocess and process). Callable from command line.
 """

import argparse
import datetime
import json
from pathlib import Path

import tqdm
from neuroprocessing.scripts.analysis import preprocess_and_process_trial

if __name__ == '__main__':
    # Load default values from JSON
    with open(Path('analysis_runs') / 'default_analysis_params.json') as f:
        default_params = json.load(f)

    parser = argparse.ArgumentParser(description=(
        'Run analysis (pre-process and processing steps).',
        'Default parameters are loaded from `analysis_runs/default_analysis_params.json`.'))

    # Define arguments with defaults from JSON file
    parser.add_argument('--date', type=str, required=True,
                        help='Experiment date folder to analyze(required)')
    parser.add_argument('--aligner_target_num_features', type=int,
                        default=default_params.get('aligner_target_num_features'),
                        help='Number of target features for aligner')
    parser.add_argument('--secs_before_stim', type=int,
                        default=default_params.get('secs_before_stim'),
                        help='Only process frames starting at X seconds before stimulus')
    parser.add_argument('--preprocess_prefix', type=str,
                        default=default_params.get('preprocess_prefix'),
                        help='Prefix to use for preprocessed data')
    parser.add_argument('--process_prefix', type=str,
                        default=default_params.get('process_prefix'),
                        help='Prefix to use for processed data')
    parser.add_argument('--s3fs_toplvl_path', type=str,
                        default=default_params.get('s3fs_toplvl_path'),
                        help='Top level path for S3FS')
    parser.add_argument('--local_toplvl_path', type=str,
                        default=default_params.get('local_toplvl_path'),
                        help='Local top level path')
    parser.add_argument('--load_from_s3', action='store_true',
                        default=default_params.get('load_from_s3'),
                        help='Flag to load data from S3')
    parser.add_argument('--save_to_s3', action='store_false',
                        default=default_params.get('save_to_s3'),
                        help='Flag to save data to S3')
    parser.add_argument('--crop_px', type=int,
                        default=default_params.get('crop_px'),
                        help='Pixels to crop on each side to avoid edge effects after motion correction.')
    parser.add_argument('--bottom_percentile', type=int,
                        default=default_params.get('bottom_percentile'),
                        help='Bottom percentile of pixels to use for bleach correction.')
    parser.add_argument('--flood_connectivity', type=int,
                        default=default_params.get('flood_connectivity'),
                        help='Connectivity setting for flood-filling algorithm (skimage.segmentation.flood) to identify brain mask.')
    parser.add_argument('--flood_tolerance', type=int,
                        default=default_params.get('flood_tolerance'),
                        help='Tolerance setting for flood-filling algorithm (skimage.segmentation.flood) to identify brain mask.')

    args = parser.parse_args()

    params = vars(args)  # Convert argparse Namespace to dict

    exp_path = Path(args.local_toplvl_path) / args.date
    assert exp_path.exists(), f"Experiment date folder {exp_path} does not exist."

    trial_dirs = [x for x in Path(exp_path).iterdir() if x.is_dir()]
    assert trial_dirs, f"No trial directories found in {exp_path}."

    for trial_dir in tqdm.tqdm(trial_dirs):
        '''
        need to identify which trial is tactile stim and which is injection
        Right now just using the information that tactile stims are 5 mins and injection stims are 
        longer
        '''
        is_tactile_stim_trial = '_5min_' in trial_dir.name
        params['sync_csv_col'] = 'stim' if is_tactile_stim_trial else 'button'
        params['downsample_factor'] = 2 if is_tactile_stim_trial else 8
        params['secs_before_stim'] = 0 if is_tactile_stim_trial else 60


        preprocess_and_process_trial(args.date, trial_dir.name, params)
