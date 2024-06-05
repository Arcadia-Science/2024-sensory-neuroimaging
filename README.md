![sample_montage](https://github.com/Arcadia-Science/2024-neuroimaging-pruritogens/assets/4419151/8f50e257-c0b4-449f-b7d3-684038b42816)

# 2024-neuroimaging-pruritogens

[![run with conda](http://img.shields.io/badge/run%20with-conda-3EB049?labelColor=000000&logo=anaconda)](https://docs.conda.io/projects/miniconda/en/latest/)

## Overview

This repo contains microscope control software, data analysis scripts, and notebooks for the translation pilot "Brain imaging of pruritogen responses". This includes all the code necessary to reproduce figures and results from the pub.

## Installation and Setup

This repository uses conda to manage software environments and installations.

```bash
conda create -y --name neuroimaging-analysis --file envs/all-dependencies.yml
conda activate neuroimaging-analysis
```

If you run into problems solving this environment, try installing only the direct dependencies:

```bash
conda env update -y --name neuroimaging-analysis --file envs/direct-dependencies.yml
```

To install the `neuroimaging` package in development mode, run:

```bash
pip install -e .
```

## Microscope control code

The Arduino firmware code to control the LED and the tactile stimulator is in `microscope/arduino/`. This code should be loaded onto the Teensy microcontroller using the Arduino IDE (version 2.3.2, [download here](https://www.arduino.cc/en/software)). ([Teensy 4.0](https://www.pjrc.com/store/teensy40.html) was used for this project). User should make sure that the pin assignments within the code correspond to the physical circuit (see [Couto et al 2021](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8788140/)) as a starting point.

The Python script to control the microcontroller is `microscope/python/launch_stim.py`. This script should be run on the computer that controls the microscope hardware.

This script requires its own conda environment:
```bash
conda create --name neuroimaging-microscope --file envs/microscope-control.yml
conda activate neuroimaging-microscope
```
This environment should be created on the computer that will control the microscope.

Run the script using the command line to initiate the LED and the stimulator (if needed). Example use cases are shown in the file.

## Neuroimaging analysis pipeline

This pipeline is designed to preprocess and analyze *in vivo* brain imaging data collected with a widefield microscope. Included in the pipeline are the following steps:

* Preprocessing in the `preprocess_trial` function in `src/neuroprocessing/pipeline.py`
    * Load TIFF stack of raw imaging data, concatenate stacks if needed
    * Load NIDAQ sync file to align imaging data with stimulus
    * Downsample stack in time
    * Correct for motion artifacts (see [Motion correction](#motion-correction) below)
* Processing in the `process_trial` function in `src/neuroprocessing/pipeline.py`
    * Automatically mask out the brain
    * Correct for photobleaching
* Analysis and visualization in the `ImagingTrialLoader` class in `src/neuroprocessing/imagingtrials.py`
    * Aggregate all imaging trials
    * Filters imaging trails based on metadata (e.g. hindlimb stimulation only)
  

### Path configuration file

Paths to raw and processed data on the user's computer should be set in the configuration file `config/default.json` by default. Use the following template to create the configuration file:

```json
{
    "processed_data_dir": "path/to/processed_data",
    "raw_data_dir": "path/to/rawdata"
}
```

This template is also located in `config/default_template.json`. Rename it to `default.json` and add the real data paths to run the scripts in this repository.


## Dataset

The raw unprocessed experimental data are stored in a Zenodo repository [ADD LINK](). Processed data are stored in a Zenodo repository [ADD LINK]().

### Experiment file structure

* Raw imaging and NIDAQ sync data is stored in S3 buckets that are accessible using [S3FS](https://github.com/s3fs-fuse/s3fs-fuse). Follow instructions on S3FS to mount the S3 bucket to a local directory.
* File structure is assumed to be `{top-level exp dir}/{exp date}/{trial dir}/{Tiff stacks and nidaq CSV files here}`
* Tiff stack filenames are assumed to be in the form: `{camera}_{duration}_{stimulus_location}_{stimulus_pattern}_{notes}` where:
    * `camera` e.g. `"Zyla"` is the camera name
    * `duration` e.g. `"5min"` is the duration of the trial
    * `stimulus_location` e.g. `"LHLstim"` is the type of stimulus (left hindlimb stimulation)
    * `stimulus_pattern` e.g. `"2son4soff"` if tactile stimulation (2 seconds on, 4 seconds off) or, if injection, injection type (e.g. `saline`, `histamine`)
    * `notes` e.g. `"1pt75pctISO"`: iso concentration (but can be other notes)

### How to run the pipeline

1. Copy the raw folder names for the trials that you want to analyze from the top-level S3 dir to a local directory, e.g. `data/2024-03-06/`
2. Adjust default parameters in `analysis_runs/default_analysis_params.json`. Specifically, set `"s3fs_toplvl_path"` to the S3FS mounted directory, and `"local_toplvl_path"` to the local directory
3. Run the pipeline using `python src/neuroprocessing/scripts/run_analysis.py --date "2024-02-29"`
4. The pipeline will output the processed data to the local directory specified in the parameters file (see [Analysis parameters](#analysis-parameters))

### Analysis parameters

Analysis parameters are stored in a JSON file, e.g. `analysis_runs/default_analysis_params.json`. The file contains the following parameters:

 * `aligner_target_num_features`: Default is 700. Number of target features for aligner (larger number is more accurate but slower).
 * `preprocess_prefix`: Default is `"aligned_downsampled_"`. Prefix used for preprocessed image files.
 * `process_prefix`: Default is `"processed_"`. Prefix used for processed image files.
 * `s3fs_toplvl_path`: Default is `"/s3/path/to/videos/"`. Set this to the mounted S3 bucket path.
 * `local_toplvl_path`: Default is` "/localpath/to/videos/"`. Set this to the local directory where the raw data is copied.
 * `load_from_s3`: Default is `true`.
 * `save_to_s3`: Default is `false`.
 * `crop_px`: Default is `20`. Number of pixels to crop from each edge of the image to eliminate edge artifacts from motion correction.
 * `bottom_percentile`: Default is `5`. Percent of pixels to subtract from every frame to correct for photobleaching.
 * `flood_connectivity`: Default is `20`. Connectivity setting for flood-filling algorithm (`skimage.segmentation.flood`) to identify brain mask.
 * `flood_tolerance`:  Default is `1000`. Tolerance setting for flood-filling algorithm (`skimage.segmentation.flood`) to identify brain mask.


**Additional parameters are added with `run_analysis` during runtime as follows:**

 * `sync_csv_col` is set to `"stim"` if this is a tactile stim trial; otherwise `"button"`. `"stim"` is the channel in the sync file indicating when the vibration motor is on. `"button"` is the channel in the sync file indicating when the user pushed the button to indicate that the  injection is happening.
 * `downsample_factor` is set to `2` this is a tactile stim trial; otherwise `8`. For tactile, 2 was chosen because tactile stim on times are only ~20 frames long, so we don't want to downsample so much that we lose the stim. For injection trials, 8 was chosen bc it is approximately the breathing rate of the animal.
 * `secs_before_stim` is set to 0 if this is a tactile stim trial; otherwise `60`. For tactile, process the whole recording. For injections, process starting at 60 s (1 min) before injection to avoid artifacts.

`run_analysis` determines whether the current trial is a tactile stim trial or an injection trial based on the trial name. Tactile stimulation trials are typically 5 mins long and have `"_5min_"` in the trial name. Injection trials are typically 15 or 30 mins long.

## Scripts

To display a help message for any script:

```bash
python src/neuroprocessing/scripts/{script.py} --help
```

### Overall pipeline

To analyze imaging data, see `src/neuroprocessing/scripts/run_analysis.py`. The script includes steps for preprocessing (downsample, motion correction) and processing (segmentation, bleach correction). For example, to analyze all experiments from a single day, run:

```bash
conda activate neuroimaging
python src/neuroprocessing/scripts/run_analysis.py \
    --date 2024-02-21 \
    --params_file path/to/file.json
```

To analyze a single trial in a day, run:
```bash
python src/neuroprocessing/scripts/run_analysis.py \
    --date 2024-02-21 \
    --params_file path/to/file.json \
    --trial Zyla_5min_LFLstim_2son4soff_1pt25pctISO_deeper_1
```

During analysis you may see the following warning: 

```
<tifffile.TiffFile 'Zyla_30min_RHL_…ack_Pos0.ome.tif'> ImageJ series metadata invalid or corrupted file
``` 

This warning is expected because we are not using the OME metadata in TIFF files. It can be ignored. You will also see warnings like `UserWarning: {filename} is a low contrast image`. This is also expected and can be ignored.

### Motion correction

To apply motion correction to a timelapse:

```bash
python src/neuroprocessing/scripts/correct_motion.py \
    --filename /path/to/timelapse.ome.tif
```

Note that this script is somewhat computationally expensive and runs on 8 processors by default. Multiprocessing can be turned off by setting the number of workers argument to 1.
```bash
python src/neuroprocessing/scripts/correct_motion.py \
    --filename /path/to/timelapse.ome.tif \
    --num-workers 1
```

To help minimize the computational expense of motion correction, there is a built-in optimization routine for auto-adjusting the SIFT + RANSAC parameters to arrive at a target number of features. A higher number of target features should result in a more accurate alignment at the expense of time and processing power. By default the target number of features is set to 150, which was found to be a satisfactory balance between accuracy and computation time for our data and computational resources. The target number of features can be adjusted when running motion correction from the command line via
```bash
python src/neuroprocessing/scripts/correct_motion.py \
    --filename /path/to/timelapse.ome.tif \
    --target-num-features 200
```
Alternatively, it can be set as a parameter (`aligner_target_num_features`) in `pipeline_params/default_pipeline_params.json` when running batch processing.

## Contributing

See how we recognize [feedback and contributions to our code](https://github.com/Arcadia-Science/arcadia-software-handbook/blob/main/guides-and-standards/guide-credit-for-contributions.md).
