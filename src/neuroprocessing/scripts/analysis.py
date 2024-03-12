from pathlib import Path
import json
from natsort import natsorted
import numpy as np
import pandas as pd

from skimage import io
from skimage.measure import block_reduce
from skimage.segmentation import flood

from neuroprocessing.scripts.parse_csv import parse_csv, process_data
from neuroprocessing.align import StackAligner

from scipy import ndimage

def identify_trial_save_paths(exp_dir:str, trial_dir:str, params:dict) -> tuple:
    """Identify the paths to the raw and processed tiff stacks for a single trial
    
    Inputs:
        exp_dir: str
            Name of the experiment directory e.g. "2024-03-06"
        trial_dir: str
            Name of the trial directory e.g. "Zyla_30min_LHL_27mMhistinj_1pt75pctISO_1
        params: dict
            Run parameters
    Returns:
        tuple
            (Path to the raw tiff stack, Path to the processed tiff stack)
    """
    load_from_s3 = params["load_from_s3"]
    save_to_s3 = params["save_to_s3"]
    if load_from_s3:
        print("Loading raw stack from S3")
    else:
        print("Loading raw stack from local filesystem")

    trial_path = params["s3fs_toplvl_path"] if load_from_s3 else params["local_toplvl_path"]
    save_path = params["s3fs_toplvl_path"] if save_to_s3 else params["local_toplvl_path"]

    trial_path = Path(trial_path) / exp_dir / trial_dir
    save_path = Path(save_path) / exp_dir / trial_dir
    return (trial_path, save_path)

def _get_sync_info(sync_csv_path):
    """Get dict of sync info (stim time, etc) from the sync csv file
    """
    df_daq = parse_csv(sync_csv_path)

    df_frames = process_data(
        df_daq,
        col_camera="camera",
        col_stim="button"
    )

    stim_onset_frame = int(df_frames.loc[df_frames["stimulated"], "frame"].iloc[0])
    df_frames.set_index("frame", inplace=True)
    # print info
    print(f"Stimulus onset frame: {stim_onset_frame}")
    print(f"Stimulus onset time (s): {df_frames.loc[stim_onset_frame, 'time']}")
    stim_duration_frames = sum(df_frames['stimulated'])
    frame_time_s = df_frames.loc[stim_onset_frame, 'frametime']
    framerate_hz = 1/frame_time_s

    sync_info = {
        "stim_onset_frame": stim_onset_frame,
        "stim_duration_frames": stim_duration_frames,
        "frame_time_s": frame_time_s,
        "framerate_hz": framerate_hz
    }
    print(f"Stimulus duration (frames): {stim_duration_frames}")
    print(f"Stimulus duration (s): {stim_duration_frames/framerate_hz}")
    print(f"Frame duration (ms): {frame_time_s*1000}")
    print(f"Framerate (Hz): %.2f" % (framerate_hz))
    return sync_info

def _get_brain_mask(stack):
    """Return a binary mask of the brain from the tiff stack
    """
    stack_max = stack.max(axis=0)
    stack_max_clipped = stack_max.copy()
    stack_max_clipped[stack_max_clipped > np.median(stack_max) ] = np.median(stack_max)
    mask = flood(stack_max_clipped, (int(stack_max_clipped.shape[0]/2), int(stack_max_clipped.shape[1]/2)), tolerance=100)
    mask = ndimage.binary_dilation(mask, iterations=10)
    return mask

def process_trial(exp_dir:str, trial_dir:str, params:dict):
    """Process single trial
    
    Process a single imaging trial and save the processed tiff stack to the same directory as the original tiff stack.

    An imaging trial can be split into 2+ videos due to tiff file size limits.

    Stimulus is assumed to be in the first video.
    
    Inputs:
        exp_dir: str 
            Experiment dir e.g. "2024-03-06"
        trial_dir: str 
            Trial dir e.g. "Zyla_15min_LHL_salineinj_1pt75pctISO_1"
    """

    trial_path, save_path = identify_trial_save_paths(exp_dir, trial_dir, params)

    # load sync json if exists
    if (save_path / "sync_info.json").exists():
        with open(save_path / "sync_info.json", "r") as f:
            sync_info = json.load(f)
    else:
        sync_info = {}
        sync_info["stim_onset_frame"] = 3000
        sync_info["framerate_hz"] = 10
        print(f"Warning: No sync_info.json found. Using default value of {sync_info['stim_onset_frame']} for stimulus onset time and {sync_info['framerate_hz']} Hz for framerate.")
    
    # load processed tiff stack
    fp_processed_tif = save_path / (params["preprocess_prefix"] + trial_dir + ".tif")
    if not fp_processed_tif.exists():
        raise FileNotFoundError(f"Error: No preprocessed file exisits: {fp_processed_tif}")

    stack = io.imread(fp_processed_tif)
    # crop image to remove the black border that occurs due to motion registration
    crop_px = params["crop_px"]
    stack = stack[:, crop_px:-crop_px, crop_px:-crop_px]

    # only process frames starting at X seconds before stimulus
    frames_before_stim = int(params["secs_before_stim"] * sync_info["framerate_hz"])
    stack = stack[int((sync_info["stim_onset_frame"] - frames_before_stim)/params['downsample_factor']):, :, :]

    mask = _get_brain_mask(stack)
    
    # find bottom X% of pixels in brain
    bottomX = np.percentile(stack[:, mask], params['bottom_percentile'], axis=1, keepdims=True).astype(np.uint16)

    # subtract bottom X% from all pixels
    stack -= bottomX[:,np.newaxis]
    stack -= stack.min(axis=0, keepdims=True)
    stack[stack > 30000] = 0

    # save mask
    np.save(save_path / ('mask_' + params["process_prefix"] + trial_dir + '.npy'), mask)
    
    # save tiff stack with the name of the first tiff in the stack
    io.imsave(save_path / (params["process_prefix"] + trial_dir + '.tif'), stack)


def preprocess_trial(exp_dir:str, trial_dir:str, params:dict):
    """Preprocessing of a single imaging trial

    A single trial may have multiple videos due to tiff file size limits.
        1. Downsample the tiff stack
        2. Motion correction
    
        Notes: 
            - Tiff stacks with MMStack in the name will be processed and concatenated if there are multiple. imread is used with `is_ome=False` and `is_mmstack=False` to make sure it doesn't try to read the OME metadata and load the entire file sequence
        Usage:
            preprocess_trial("2024-03-06", "Zyla_15min_LHL_salineinj_1pt75pctISO_1", params)
        
    Inputs:
        exp_dir: str 
            Experiment dir e.g. "2024-03-06"
        trial_dir: str 
            Trial dir e.g. "Zyla_15min_LHL_salineinj_1pt75pctISO_1"
    """
    
    trial_path, save_path = identify_trial_save_paths(exp_dir, trial_dir, params)

    fp_tifs = natsorted(Path(trial_path).glob("*MMStack*.tif"))
    fp_csv = natsorted(Path(trial_path).glob("*.csv"))
    
    if len(fp_csv) == 0:
        print(f"No sync file found in {trial_path}")
    else:
        sync_info = _get_sync_info(fp_csv[0])
        # save sync info as json
        with open(save_path / "sync_info.json", "w") as f:
            json.dump(sync_info, f)
    
    stack_list = []
    for fp_tif in fp_tifs:
        stack_list.append(io.imread(fp_tif, is_ome=False, is_mmstack=False))

    # Downsample image
    stack_downsampled = block_reduce(np.concatenate(stack_list, axis=0), block_size=(params['downsample_factor'], 1, 1), func=np.mean)

    # temporarily save downsampled stack
    # io.imsave(save_path / ("downsampled_" + trial_dir + ".tif"), stack_downsampled.astype(np.uint16))
    del(stack_list)

    aligner = StackAligner(
            stack=stack_downsampled,
            target_num_features = params["aligner_target_num_features"]
        )
    aligner.align()
    
    # save aligned stack
    stack_aligned = aligner.stack_aligned

    io.imsave(save_path / (params["preprocess_prefix"] + trial_dir + ".tif"), stack_aligned.astype(np.uint16))

def preprocess_and_process_trial(exp_dir:str, trial_dir:str, params:dict):
    """Preprocess and process a single trial
    
    Inputs:
        exp_dir: str 
            Experiment dir e.g. "2024-03-06"
        trial_dir: str 
            Trial dir e.g. "Zyla_15min_LHL_salineinj_1pt75pctISO_1"
    """
    preprocess_trial(exp_dir, trial_dir, params)
    process_trial(exp_dir, trial_dir, params)

