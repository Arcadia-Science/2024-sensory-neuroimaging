# 2024-neuroimaging-pruritogens

[![run with conda](http://img.shields.io/badge/run%20with-conda-3EB049?labelColor=000000&logo=anaconda)](https://docs.conda.io/projects/miniconda/en/latest/)


Data analysis scripts and notebooks for the translation pilot "Brain imaging of pruritogen responses"

## Purpose


## Installation and Setup

This repository uses conda to manage software environments and installations.

```{bash}
conda create -n neuro --file envs/dev.yml
conda activate neuro
```

To install the package in development mode, run:

```{bash}
pip install -e .
```



## Overview

## Neuroimaging analysis pipeline

This pipeline is designed to preprocess and analyze data *in vivo* brain imaging data collected with a widefield microscope. Included in the pipeline are the following steps:

* Preprocessing (`preprocess_trial` in `src/neuroprocessing/scripts/analysis.py`)
    * Load TIFF stack of raw imaging data, concatenate stacks if needed
    * Load NIDAQ sync file to align imaging data with stimulus
    * Downsample stack in time
    * Correct for motion artifacts (see [Motion correction](#motion-correction))
* Processing (`process_trial` in `src/neuroprocessing/scripts/analysis.py`)
    * Automatically mask out the brain
    * Correct for photobleaching
    * Additional steps TBD
* Analysis and visualization (`ImagingTrialLoader` in `src/neuroprocessing/imagingtrials.py`)
    * Aggregate all imaging trials, filter them based on metadata (e.g. hindlimb stimulation only), output results
    * Sample usage in `notebooks/injection_analysis.ipynb`

**Note: Pipeline only supports single-stimulation trials (i.e. injections), where no averaging across trials needs to be performed**

### How to run the pipeline
Assumed file structure
* Raw imaging and NIDAQ sync data is stored in S3 buckets that are accessible using [S3FS](https://github.com/s3fs-fuse/s3fs-fuse). Follow instructions on S3FS to mount the S3 bucket to a local directory.
* File structure is assumed to be `{top-level exp dir}/{exp date}/{trial dir}/{Tiff stacks and nidaq CSV files here}`
* Tiff stack filenames are assumed to be in the form: `Zyla_5min_LHLstim_2son4soff_1pt75pctISO_1` where:
    * `Zyla` is the camera name
    * `5min` is the duration of the trial
    * `LHLstim` is the type of stimulus (left hindlimb stimulation)
    * `2son4soff` is the stimulus pattern (2 seconds on, 4 seconds off) or, if injection, injection type (e.g. `saline`, `histamine`)
    * `1pt75pctISO` iso concentration (but can be other notes)

1. Copy the raw folder names for the trials that you want to analyze from the top-level S3 dir to a local directory, e.g. `data/2024-03-06/`
2. Adjust default parameters in `analysis_runs/default_analysis_params.json` if needed
2. Run the pipeline using CLI `python src/neuroprocessing/scripts/run_analysis.py --date "2024-02-29"`
3. The pipeline will output the processed data to the local directory you specified in `default_analysis_params.json` or in the CLI arguments
4. See `notebooks/injection_analysis.ipynb` for an example of how to load and analyze the processed data

## Scripts

To display a help message for any script:
```python
python src/neuroprocessing/scripts/{script.py} --help
```

### Run injection analysis

To analyze injection imaging data, see `src/neuroprocessing/scripts/run_analysis.py`. The script includes steps for preprocessing (downsample, motion correction) and processing (segmentation, bleach correction). For example, to analyze a single experiment day, run:

```bash
conda activate neuro
python src/neuroprocessing/scripts/run_analysis.py --date "2024-02-29"
```

During analysis you may see the following warning: `<tifffile.TiffFile 'Zyla_30min_RHL_…ack_Pos0.ome.tif'> ImageJ series metadata invalid or corrupted file`. This warning is expected because we are not using the OME metadata in TIFF files. It can be ignored. You will also see warnings like `UserWarning: {filename} is a low contrast image`. This is also expected and can be ignored.

### Motion correction

To apply motion correction to a timelapse:
```python
python src/neuroprocessing/scripts/correct_motion.py --filename {/path/to/timelapse.ome.tif}
```

Note that this script is somewhat computationally expensive and runs on 8 processors by default. Multiprocessing can be turned off by setting the number of workers argument to 1.
```python
python src/neuroprocessing/scripts/correct_motion.py --filename {/path/to/timelapse.ome.tif} --num-workers 1
```


## Contributing

See how we recognize [feedback and contributions to our code](https://github.com/Arcadia-Science/arcadia-software-handbook/blob/main/guides-and-standards/guide-credit-for-contributions.md).
