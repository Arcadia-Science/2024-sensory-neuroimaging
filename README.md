![sample_montage](https://github.com/Arcadia-Science/2024-neuroimaging-pruritogens/assets/4419151/8f50e257-c0b4-449f-b7d3-684038b42816)

# 2024-sensory-neuroimaging

[![run with conda](http://img.shields.io/badge/run%20with-conda-3EB049?labelColor=000000&logo=anaconda)](https://docs.conda.io/projects/miniconda/en/latest/)

## Overview

This repo contains microscope control software, data analysis scripts, and notebooks for the translation pilot ["Brain imaging of pruritogen responses"](https://doi.org/10.57844/arcadia-b963-15ac). This repository includes all the code necessary to reproduce the figures and results from the pub.

## Installation and Setup

This repository uses conda to manage software environments and installations.

```bash
conda env create -y --name neuroimaging-analysis --file envs/all-dependencies.yml
conda activate neuroimaging-analysis
```

If conda cannot solve this environment, try installing only the direct dependencies:

```bash
conda env create -y --name neuroimaging-analysis --file envs/direct-dependencies.yml
```

To install the `neuroimaging` package in development mode, run:

```bash
pip install -e .
```

## Microscope control code

The Arduino firmware code to control the LED and the tactile stimulator is in `microscope/arduino/`. This code should be loaded onto the Teensy microcontroller using the Arduino IDE (version 2.3.2; [download here](https://www.arduino.cc/en/software)). ([Teensy 4.0](https://www.pjrc.com/store/teensy40.html) was used for this project). User should make sure that the pin assignments within the code correspond to the physical circuit (see [Couto et al 2021](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8788140/)) as a starting point.

The Python script to control the microcontroller is `microscope/python/launch_stim.py`. This script should be run on the computer that controls the microscope hardware.

This script requires its own conda environment:
```bash
conda env create --name neuroimaging-microscope --file envs/microscope-control.yml
conda activate neuroimaging-microscope
```
This environment should be created on the computer that will control the microscope.

Run the script using the command line to initiate the LED and the stimulator (if needed). Example use cases are shown in the file.

## Neuroimaging analysis pipeline

This pipeline is designed to preprocess and analyze *in vivo* brain imaging data collected with a widefield microscope. Included in the pipeline are the following steps:

* Preprocessing (in the `preprocess_trial` function in `src/neuroprocessing/pipeline.py`)
    * Load TIFF stack of raw imaging data, concatenate stacks if needed
    * Load NIDAQ sync file to align imaging data with stimulus
    * Downsample stack in time
    * Correct for motion artifacts (see [Motion correction](#motion-correction) below)
* Processing (in the `process_trial` function in `src/neuroprocessing/pipeline.py`)
    * Automatically mask out the brain
    * Correct for photobleaching
* Analysis and visualization (in the `ImagingTrialLoader` class in `src/neuroprocessing/imagingtrials.py`)
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


### Datasets

The raw and processed experimental data are stored in a [Zenodo repository](https://doi.org/10.5281/zenodo.11585535).

The raw data is divided into five zip files, each corresponding to a subdirectory in the zip file of processed data. These subdirectories are named with dates in the YYYY-MM-DD format. The raw data zip files follow the naming convention `pub-data-raw-YYYY-MM-DD.zip`.

To reorganize the raw data to match the directory structure of the processed data, follow these steps:

1. Extract each of the five raw data zip files into directories named according to their timestamps (e.g., YYYY-MM-DD).
1. Move all these timestamped directories into a single raw-data/ directory.

By following these steps, the raw data will be structured in the same way as the processed data, facilitating further analysis and comparison.

### Experiment file structure

* File structure for raw and processed data is `{top-level exp dir}/{exp date}/{trial dir}/{Tiff stacks and nidaq CSV files here}`, e.g. `data-processed/2024-02-29/Zyla_5min_LHLstim_2son4soff_1pt25pctISO_2`
* Tiff stack filenames are assumed to be in the form: `{camera}_{duration}_{stimulus_location}_{stimulus_pattern}_{notes}` where:
    * `camera` e.g. `"Zyla"` is the camera name
    * `duration` e.g. `"5min"` is the duration of the trial
    * `stimulus_location` e.g. `"LHLstim"` is the type of stimulus (left hindlimb stimulation)
    * `stimulus_pattern` e.g. `"2son4soff"` if tactile stimulation (2 seconds on, 4 seconds off) or, if injection, injection type (e.g. `saline`, `histamine`)
    * `notes` e.g. `"1pt75pctISO"`: iso concentration (but can be other notes)

### Processing raw imaging data

1. Download the [dataset from Zenodo](#dataset). Note: the dataset is large (~80 Gb).
2. Unzip the raw data file `pub-data-processed.zip` to a directory on your computer.
3. Create a new empty directory for the processed data.
4. [Update the path configuration file](#path-configuration-file) to point to the raw and processed data directories.
5. Run the pipeline for all experiment dates (**note: this may take >5 hours if running locally**):

```bash

conda activate neuroimaging
for date in 2024-02-21 2024-02-29 2024-03-06 2024-03-18 2024-03-19; do
    python src/neuroprocessing/scripts/run_pipeline.py \
        --date $date \
        --params_file pipeline_params/default_pipeline_params.json \
        --reanalyze
done

```

### Reproducing figures from the pub

1. Download the [dataset from Zenodo](#dataset) or re-generate it using the steps above. If downloaded, unzip the processed data file `pub-data-processed.zip` to a directory on your computer.
2. [Update the path configuration file](#path-configuration-file) to point to the processed data directory.
3. Run `notebooks/generate_figures.ipynb`. Static figures will be displayed inline in the notebook. Animations of brain activity will be saved in `notebooks/figs/` as TIFFs.

### Pipeline parameters

Parameters for processing the raw data are stored in a JSON file in `pipeline_params`. The parameters are:

 * `aligner_target_num_features`: Default is 700. Number of target features for aligner (larger number is more accurate but slower).
 * `preprocess_prefix`: Default is `"aligned_downsampled_"`. Prefix used for preprocessed image files.
 * `process_prefix`: Default is `"processed_"`. Prefix used for processed image files.
 * `crop_px`: Default is `20`. Number of pixels to crop from each edge of the image to eliminate edge artifacts from motion correction.
 * `bottom_percentile`: Default is `5`. Percent of pixels to subtract from every frame to correct for photobleaching.
 * `flood_connectivity`: Default is `20`. Connectivity setting for flood-filling algorithm (`skimage.segmentation.flood`) to identify brain mask.
 * `flood_tolerance`:  Default is `1000`. Tolerance setting for flood-filling algorithm (`skimage.segmentation.flood`) to identify brain mask.

**Additional parameters are added within `run_pipeline.py` at runtime:**

 * `sync_csv_col` is set to `"stim"` if this is a tactile stimulation trial; otherwise `"button"`. `"stim"` is the channel in the sync file indicating when the vibration motor is on. `"button"` is the channel in the sync file indicating when the user pushed the button to indicate that the  injection is happening.
 * `downsample_factor` is set to `2` if this is a tactile stim trial; otherwise `8`. For tactile trials, 2 was chosen because tactile stimulus on times are only ~20 frames long, so we don't want to downsample so much that we lose the stimulus window. For injection trials, 8 was chosen because it is approximately the breathing rate of the animal.
 * `secs_before_stim` is set to 0 if this is a tactile stim trial; otherwise `60`. For tactile, process the whole recording. For injections, process starting at 60 s (1 min) before injection to avoid artifacts.

Whether the current trial is a tactile stim trial or an injection trial is determined within `run_pipeline.py` based on the trial name. Tactile stimulation trials are typically 5 minutes long and have `"_5min_"` in the trial name. Injection trials are typically 15 or 30 minutes long.

### Image processing pipeline

To process raw imaging data, use `src/neuroprocessing/scripts/run_pipeline.py`. The script includes steps for preprocessing (downsampling and motion correction) and processing (segmentation and bleach correction). For example, to analyze all experiments from a single day, run:

```bash
conda activate neuroimaging
python src/neuroprocessing/scripts/run_analysis.py \
    --date 2024-02-21 \
    --params_file pipeline_params/default_pipeline_params.json
```

To analyze a single trial in a day, run:
```bash
python src/neuroprocessing/scripts/run_analysis.py \
    --date 2024-02-21 \
    --params_file pipeline_params/default_pipeline_params.json \
    --trial Zyla_5min_LFLstim_2son4soff_1pt25pctISO_deeper_1
```

For additional functionality, see `src/neuroprocessing/scripts/run_pipeline.py --help`

During analysis you may see the following warning: 

```
<tifffile.TiffFile '...'> ImageJ series metadata invalid or corrupted file
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
Alternatively, it can be set using the parameter `aligner_target_num_features` in `pipeline_params/default_pipeline_params.json` when running batch processing.

## Contributing

See how we recognize [feedback and contributions to our code](https://github.com/Arcadia-Science/arcadia-software-handbook/blob/main/guides-and-standards/guide-credit-for-contributions.md).
