"""# run_analysis.py

Runs analysis pipeline (preprocess and process). Callable from command line.
 """

import json
from pathlib import Path

import click
import tqdm
from neuroprocessing.pipeline import load_user_config, preprocess_and_process_trial


@click.command()
@click.option('--date', type=str, required=True, help='Experiment date folder to analyze.')
@click.option('--params_file', type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path), required=True, help='Path to JSON file containing analysis parameters. See README.md for details.')
@click.option('--trial', type=str, required=False, help='Trial folder to analyze. If not provided, all trials in the experiment folder will be analyzed.')
@click.option('--reanalyze', is_flag=True, help='If set, reanalyze all trials, even if already processed. Processed folders have a params.json file.')
def execute_pipeline(date, trial, params_file, reanalyze):

    with open(params_file) as f:
        params = json.load(f)

    config = load_user_config('default')

    exp_path = Path(config['processed_data_dir']) / date
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment date folder {exp_path} does not exist.")

    if trial:
        trial_dirs = [Path(trial)]
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

        if reanalyze or not (trial_dir / 'params.json').exists():
            print(f"Processing {trial_dir.name}...")
            preprocess_and_process_trial(date, trial_dir.name, params)
        else:
            print(f"Skipping {trial_dir.name} (already processed).")



if __name__ == '__main__':
    execute_pipeline()
