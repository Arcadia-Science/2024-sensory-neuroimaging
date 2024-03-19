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


## Overview


## Scripts

To display a help message for any script:
```python
python src/neuroprocessing/scripts/{script.py} --help
```

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
