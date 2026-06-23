# Central Asia Compound Heatwaves

This repository contains Python scripts used for analyzing daytime, nighttime, and compound heatwaves in Central Asia based on GSOD station observations, ERA5 reanalysis data, and MODIS land-cover data.

The scripts accompany the manuscript:

**Day-night asymmetry and transition toward compound heatwaves in arid Central Asia: evidence from station observations and ERA5 reanalysis**

## Description

The repository provides scripts for heatwave identification, heatwave-index calculation, spatial trend analysis, ERA5-GSOD comparison, land-use-based analysis, robustness tests, and figure generation.

The heatwave types analyzed in the study include:

* Daytime heatwaves
* Nighttime heatwaves
* Compound day-night heatwaves

The main heatwave indices include:

* Heatwave frequency
* Heatwave duration
* Heatwave intensity

## Data sources

The raw datasets are not redistributed in this repository because they are publicly available from their official data providers.

* GSOD station observations: NOAA National Centers for Environmental Information
* ERA5 reanalysis data: Copernicus Climate Data Store
* MODIS MCD12Q1 land-cover data: NASA LP DAAC

Users should download the original datasets from the official data portals and adjust the file paths in the scripts according to their local directory structure.

## Repository contents

* `*.py`: Python scripts for data processing, heatwave identification, statistical analysis, and figure generation.
* `requirements.txt`: Python package dependencies.
* `script_index.csv`: Mapping between cleaned script names and original script files.

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Before running the scripts, modify the input and output paths according to your local data directory.

A typical workflow includes:

1. Prepare GSOD station observations, ERA5 daily temperature data, and MODIS land-cover data.
2. Run the heatwave identification scripts.
3. Calculate heatwave frequency, duration, and intensity.
4. Compare ERA5 and GSOD results.
5. Generate spatial maps, interannual series, land-use comparisons, and robustness-test results.

## Notes

The scripts were organized for manuscript submission and research reproducibility. File paths may need to be modified before running the scripts on another computer.

Large raw datasets and intermediate files are not included in this repository.

## Citation

If you use this repository, please cite the associated manuscript and the archived release of this repository when available.

## Path configuration

The scripts retain the original local directory structure used in the analysis. Users should modify the input and output paths in each script according to their own local data directories before running the code.
