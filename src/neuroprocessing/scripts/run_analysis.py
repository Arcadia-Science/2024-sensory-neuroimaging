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
    parser = argparse.ArgumentParser(description=(
        'Run analysis (pre-process and processing steps).',
        'Default parameters are loaded from `analysis_runs/default_analysis_params.json`.'))

    # Define arguments with defaults from JSON file
    parser.add_argument('--date', type=str, required=True,
                        help='Experiment date folder to analyze(required)')
    parser.add_argument('--trial', type=str, required=False,
                        help='Trial folder to analyze (optional). If not provided, all trials in '
                        'the experiment folder will be analyzed.')
    parser.add_argument('--params_file', type=str, required=True,
                        help='Path to JSON file containing analysis parameters (required). See README.md for details.')
    parser.add_argument('--reanalyze', action='store_true',
                        help='If True, reanalyze all trials, even if already processed.'
                        'Processed folders have a params.json file.')

    args = parser.parse_args()

    args = vars(args)  # Convert argparse Namespace to dict
    # Load parameters from JSON file
    if not Path(args['params_file']).exists():
        raise FileNotFoundError(f"Parameters file {args['params_file']} does not exist.")
    with open(args['params_file']) as f:
        params = json.load(f)

    exp_path = Path(params['local_toplvl_path']) / args['date']
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment date folder {exp_path} does not exist.")

    if args['trial']:
        trial_dirs = [Path(args['trial'])]
    else:
        trial_dirs = [t_d for t_d in exp_path.iterdir() if t_d.is_dir()]
    if not trial_dirs:
        raise FileNotFoundError(f"No trial directories found in {exp_path}.")

    for trial_dir in tqdm.tqdm(trial_dirs):
        '''
        This is how we identify which trial is tactile stim and which is injection
        Right now just using the information that tactile stims are 5 mins and injection stims are
        longer (15 or 30 mins)

        See README.md for details on values chosen for the parameters below.
        '''
        is_tactile_stim_trial = '_5min_' in trial_dir.name

        params['sync_csv_col'] = 'stim' if is_tactile_stim_trial else 'button'
        params['downsample_factor'] = 2 if is_tactile_stim_trial else 8
        params['secs_before_stim'] = 0 if is_tactile_stim_trial else 60

        if args['reanalyze'] or not (trial_dir / 'params.json').exists():
            print(f"Processing {trial_dir.name}...")
            preprocess_and_process_trial(args['date'], trial_dir.name, params)
        else:
            print(f"Skipping {trial_dir.name} (already processed).")
