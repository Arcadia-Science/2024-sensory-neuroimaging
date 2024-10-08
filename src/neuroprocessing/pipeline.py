import json
from pathlib import Path

import numpy as np
from natsort import natsorted
from scipy import ndimage
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks, spectrogram
from skimage import io
from skimage.segmentation import flood
from skimage.transform import downscale_local_mean

from neuroprocessing.align import StackAligner
from neuroprocessing.parse_sync_file import parse_nidaq_csv, process_nidaq_dataframe


def load_user_config(config_name: str = "default") -> dict:
    """Load user config file containing user-set paths"""

    with open(Path(__file__).parents[2] / "config" / (config_name + ".json")) as f:
        config = json.load(f)
    return config


def compute_breathing_rate(signal: np.array, fs: float) -> np.array:
    """Compute breathing rate from signal using spectrogram.
    Not currently being used in the pipeline.

    Example usage:
    ```python
        img = io.imread('path/to/stack.tif')
        center = np.array(img.shape) // 2
        roi = img[:,center[1]-50:center[1]+50, center[2]-50:center[2]+50]
        roi_mean = np.mean(roi, axis=(1,2))
        f_breathing = compute_breathing_rate(roi_mean, fs = 10)
        f,ax = plt.subplots()
        plt.plot(f_breathing)
    ```

    Inputs:
        signal (np.ndarray): 1D array of signal
        fs (float): sampling frequency

    Returns:
        np.array: 1D numpy array of breathing rate, interpolated to be the same length as signal
    """

    min_peak_height = 100  # minimum height of freq peak in spectrogram
    min_breathing_freq, max_breathing_freq = (0.5, 2)  # freq range to look for breathing rate (Hz)
    spectrogram_nperseg = 200  # number of samples per segment in spectrogram
    spectrogram_noverlap = 50  # number of samples to overlap between segments in spectrogram

    f, t, Sxx = spectrogram(
        signal, fs, nperseg=spectrogram_nperseg, noverlap=spectrogram_noverlap, detrend="linear"
    )

    breathing_freqs = (f > min_breathing_freq) & (f < max_breathing_freq)
    Sxx_breathing = Sxx[breathing_freqs]

    # find peaks in spectrogram
    f_peak_array = []
    for i in range(Sxx_breathing.shape[1]):
        peaks, _ = find_peaks(Sxx_breathing[:, i], height=min_peak_height, distance=4)
        # if >1 peak, keep only highest
        if len(peaks) > 0:
            peaks = [peaks[np.argmax(Sxx_breathing[peaks, i])]]
            f_peak = f[breathing_freqs][peaks]
            f_peak_array.append(f_peak[0])
        else:
            f_peak_array.append(np.nan)

    f_peak_array = np.array(f_peak_array)

    # remove nans
    t_peak_array = t[~np.isnan(f_peak_array)]
    f_peak_array = f_peak_array[~np.isnan(f_peak_array)]

    # interpolate back to be the same size as roi_mean
    f_peak_interp = CubicSpline(t_peak_array, f_peak_array, bc_type="natural")

    return f_peak_interp(np.arange(0, len(signal)) / fs)


def _identify_trial_save_paths(exp_dir: str, trial_dir: str, params: dict) -> tuple:
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
    config = load_user_config("default")

    trial_path = Path(config["raw_data_dir"]) / exp_dir / trial_dir
    save_path = Path(config["processed_data_dir"]) / exp_dir / trial_dir

    # create directory to store output if it does not already exist
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=False)

    return (trial_path, save_path)


def _get_sync_info(sync_csv_path, stimulus_column_name="button"):
    """Get dict of sync info (stim time, etc) from the sync csv file

    Inputs:
        sync_csv_path: str
            Path to the sync csv file
        stimulus_column_name: str
            Column name in the csv file that should be used for stimulus.
            'button' means read the button channel.
            'stim' means read the vibration motor channel.
    """
    df_daq = parse_nidaq_csv(sync_csv_path)

    df_frames = process_nidaq_dataframe(
        df_daq,
        camera_column_name="camera",
        stimulus_column_name=stimulus_column_name,
    )

    # stim_onset_frame = int(df_frames.loc[df_frames["stimulated"], "frame"].iloc[0])
    df_frames.set_index("frame", inplace=True)

    stim_onset_frame = np.where(
        (df_frames["stimulated"]) & (~df_frames["stimulated"].shift(1).astype(bool))
    )[0].tolist()  # noqa: E712, E501

    n_stims = len(stim_onset_frame)
    print(f"Stimulus onset frame: {stim_onset_frame}")
    print(f"Stimulus onset time (s): {df_frames.loc[stim_onset_frame, 'time'].values}")
    stim_duration_frames = sum(df_frames["stimulated"])
    frame_time_s = df_frames.loc[stim_onset_frame, "frametime"].mean()
    framerate_hz = 1 / frame_time_s

    sync_info = {
        "Number of stimulations": n_stims,
        "stim_onset_frame": stim_onset_frame,
        "stim_duration_frames": stim_duration_frames,
        "frame_time_s": frame_time_s,
        "framerate_hz": framerate_hz,
    }
    print(f"Number of stimulations: {n_stims}")
    print(f"Stimulus duration (overall) (frames): {stim_duration_frames}")
    print(f"Stimulus duration (overall) (s): {stim_duration_frames/framerate_hz}")
    print(f"Frame duration (ms): {frame_time_s*1000}")
    print(f"Framerate (Hz): {framerate_hz:.2f}")

    return sync_info


def _get_brain_mask(stack, flood_connectivity=20, flood_tolerance=1000):
    """Return a binary mask of the brain from the tiff stack using the maximum
    projection of the stack and flood-fill algorithm."""
    stack_max = stack.max(axis=0)
    stack_max_clipped = stack_max.copy()
    stack_max_clipped[stack_max_clipped > np.median(stack_max)] = np.median(stack_max)

    # create mask for the brain, starting from the center of the image
    mask = flood(
        stack_max_clipped,
        (int(stack_max_clipped.shape[0] / 2), int(stack_max_clipped.shape[1] / 2)),
        connectivity=flood_connectivity,
        tolerance=flood_tolerance,
    )
    mask = ndimage.binary_dilation(mask, iterations=10)
    return mask


def process_trial(exp_dir: str, trial_dir: str, params: dict):
    """Process single trial

    Process a single imaging trial and save the processed tiff stack to the same directory as the
    original tiff stack.

    Inputs:
        exp_dir: str
            Experiment dir e.g. "2024-03-06"
        trial_dir: str
            Trial dir e.g. "Zyla_15min_LHL_salineinj_1pt75pctISO_1"
        params: dict
            Parameters of the run (see `run_analysis.py`)
    """

    _, save_path = _identify_trial_save_paths(exp_dir, trial_dir, params)

    # load sync json if exists
    if (save_path / "sync_info.json").exists():
        with open(save_path / "sync_info.json") as f:
            sync_info = json.load(f)
    else:
        raise FileNotFoundError(f"Error: No sync file found in {save_path}")

    # load pre-processed tiff stack
    fp_processed_tif = save_path / (params["preprocess_prefix"] + trial_dir + ".tif")
    if not fp_processed_tif.exists():
        raise FileNotFoundError(f"Error: No preprocessed file exisits: {fp_processed_tif}")
    stack = io.imread(fp_processed_tif)

    # crop image to remove the black border that occurs due to motion registration
    crop_px = params["crop_px"]
    stack = stack[:, crop_px:-crop_px, crop_px:-crop_px]

    # only process frames starting at X seconds before first stimulus
    frames_before_stim = int(params["secs_before_stim"] * sync_info["framerate_hz"])

    # if 0, process all frames, otherwise only process frames starting at X seconds before stimulus
    if frames_before_stim > 0:
        initial_stim_frame = int(
            (sync_info["stim_onset_frame"][0] - frames_before_stim) / params["downsample_factor"]
        )
        stack = stack[initial_stim_frame:, :, :]

    mask = _get_brain_mask(
        stack,
        flood_connectivity=params["flood_connectivity"],
        flood_tolerance=params["flood_tolerance"],
    )

    # find bottom X% of pixels in brain
    bottomX = np.percentile(
        stack[:, mask],
        params["bottom_percentile"],
        axis=1,
        keepdims=True,
    ).astype(np.uint16)

    # subtract bottom X% from all pixels, preventing overflow (rudimentary bleach correction)
    stack -= stack.clip(None, bottomX[:, np.newaxis])
    stack -= stack.min(axis=0, keepdims=True)

    # set pixels outside of mask to 0
    stack[:, ~mask] = 0

    # save mask
    np.save(save_path / ("mask_" + params["process_prefix"] + trial_dir + ".npy"), mask)

    # save params
    with open(save_path / "params.json", "w") as f:
        json.dump(params, f)
    # save tiff stack with the name of the first tiff in the stack
    io.imsave(save_path / (params["process_prefix"] + trial_dir + ".tif"), stack)


def preprocess_trial(exp_dir: str, trial_dir: str, params: dict):
    """Preprocessing of a single imaging trial

    A single trial may have multiple videos due to tiff file size limits.
        1. Load the tiff stack (or stacks if there are multiple due to file size limits)
        2. Downsample the tiff stack in time to de-noise (breathing artifact, shot noise)
        3. Motion correction using `StackAligner`

        Notes:
            - Tiff stacks with MMStack in the name will be processed and concatenated if there are
              multiple. imread is used with `is_ome=False` and `is_mmstack=False` to make sure it
              doesn't try to read the OME metadata and load the entire file sequence
        Usage:
            preprocess_trial("2024-03-06", "Zyla_15min_LHL_salineinj_1pt75pctISO_1", params)

    Inputs:
        exp_dir: str
            Experiment dir e.g. "2024-03-06"
        trial_dir: str
            Trial dir e.g. "Zyla_15min_LHL_salineinj_1pt75pctISO_1"
        params: dict
            Parameters of the run (see `run_analysis.py`)
    """

    trial_path, save_path = _identify_trial_save_paths(exp_dir, trial_dir, params)

    fp_tifs = natsorted(Path(trial_path).glob("*MMStack*.tif"))
    fp_csv = natsorted(Path(trial_path).glob("*.csv"))

    if not fp_csv:
        print(f"No sync file found in {trial_path}")
    else:
        sync_info = _get_sync_info(fp_csv[0], stimulus_column_name=params["sync_csv_col"])
        # save sync info as json
        with open(save_path / "sync_info.json", "w") as f:
            json.dump(sync_info, f)

    stack_list = []
    for fp_tif in fp_tifs:
        stack_list.append(io.imread(fp_tif, is_ome=False, is_mmstack=False))

    if not stack_list:
        raise FileNotFoundError(f"No tiff files found in {trial_path}")

    # Downsample and average image in time to denoise
    stack_downsampled = downscale_local_mean(
        np.concatenate(stack_list, axis=0), factors=(params["downsample_factor"], 1, 1)
    )
    # delete large stack_list variable to free up memory
    del stack_list

    aligner = StackAligner(
        stack=stack_downsampled,
        target_num_features=params["aligner_target_num_features"],
    )
    aligner.align()

    # save params
    with open(save_path / "params.json", "w") as f:
        json.dump(params, f)
    io.imsave(
        save_path / (params["preprocess_prefix"] + trial_dir + ".tif"),
        (aligner.stack_aligned).astype(np.uint16),
    )


def preprocess_and_process_trial(exp_dir: str, trial_dir: str, params: dict):
    """Preprocess and process a single trial

    Inputs:
        exp_dir: str
            Experiment dir e.g. "2024-03-06"
        trial_dir: str
            Trial dir e.g. "Zyla_15min_LHL_salineinj_1pt75pctISO_1"
        params: dict
            Parameters of the run (see `run_analysis.py`)
    """
    preprocess_trial(exp_dir, trial_dir, params)
    process_trial(exp_dir, trial_dir, params)
