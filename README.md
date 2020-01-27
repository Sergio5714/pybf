## PyBF: Python Ultrasound Beamformer

# Introduction

This repository contains the work in progress on a Python Ultrasound Beamformer Library (PyBF).

It contains:
- input/output utilities to load raw RF data from a dataset and save reconstructed images (enabled by **h5py** package),
- RF signal processing functions (enabled by **scipy**),
- delay and sum beamformer (implemented with **numpy** ),
- visualization utilities (enabled by **matplotlib** and **plotly** libraries)

# Structure of the repository

This repository contains:

- `pybf` folder contains the source code with low-level routines.
 
- `scripts` folder contains high-level functions such as `beamformer_cartesian.py` that implements entire pipeline for image reconstruction.

- `tests` folder contains 
    - simple tests for the library (located in `code` directory):
        - `plane_wave_3_test` demonstrates the cartesian dynamic receive beamformer that inputs simulated RF data captured with plane wave modality, reconstructs and then visualizes the images of virtual scatters.
        - `make_video` demonstrates how to produce a video from a reconstructed image dataset.
    - sample datasets required for the tests (located in `data` directory) + brief description of the datasets' internal structure.


# Installation and usage

To install the library we advise using [miniconda](https://docs.conda.io/en/latest/miniconda.html) package manager and create a virtual environment.
To do it:
1. Download the repository to your local PC 
2. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html)
3. Add additional channels for necessary packages by executing the following commands in terminal
```bash
conda config --append channels plotly
conda config --append channels conda-forge
```
4. Move to the library directory
5. Execute the command below to create a virtual environment named `pybf_env` and install all necessary libraries listed in `requirements.txt`
```bash
conda create --name pybf_env python=3.6 --file requirements.txt
```
**Note:** If you have problems with installing the packages you can do it manually.  Critical packages for the library are:
- numpy
- matplotlib
- scipy
- h5py
- plotly
- plotly-orca
- psutil
- requests
- opencv

The docs for the library and additional functionality will come later. To use existing features we advise exploring the provided tests.

To run the test: 
1. Run a terminal and activate conda environment
```bash
conda activate pybf_env
```
2. Navigate to the directory of the test
3. Execute 
```
python main.py
```

# License
All source code is released under Apache v2.0 license unless noted otherwise, please refer to the LICENSE file for details.
Example datasets under `tests/data` provided under a [Creative Commons Attribution 4.0 International License][cc-by] 

[cc-by]: http://creativecommons.org/licenses/by/4.0/