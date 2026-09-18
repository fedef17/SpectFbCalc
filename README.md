# SpectFbCalc
[![Documentation Status](https://readthedocs.org/projects/spectfbcalc/badge/?version=latest)](https://spectfbcalc.readthedocs.io/en/latest/?badge=latest)*Tools for the calculation of radiative feedbacks and sensitivities, both broad-band and spectrally-resolved.*

SpectFbCalc is a Python-based tool designed for calculating radiative anomalies and climate feedbacks using synthetic kernels and climate model outputs. It facilitates the analysis of radiative biases and climate feedbacks and explores the impact of parameter tuning on climate model performance and sensitivity.

## Features
- **Kernel-based computation:** Supports both broadband (Huang, ERA5) and spectral kernels.
- **CMIP Compatibility:** Works seamlessly with standard CMIP CMOR output.
- **Configurable Setup:** Fully driven by a YAML configuration file for easy experimental setups.
- **Dask Integration:** Preserves lazy evaluation for handling large model outputs efficiently.

## Documentation
For full installation instructions, usage tutorials, theoretical framework, and API reference, please visit our **[ReadTheDocs Documentation](https://spectfbcalc.readthedocs.io/en/latest/)**.

## Quick Install
```bash
git clone [https://github.com/fedef17/SpectFbCalc/](https://github.com/fedef17/SpectFbCalc/)
cd SpectFbCalc
bash install.sh
```

## Quickstart 
### Downloading Kernels and Data
Radiative kernels (spectral and broadband) are hosted separately on Zenodo
and Mendeley Data and are not included in the repository. Download them with:
```bash
cd spectfbcalc/
nohup python download_data.py > download_data.log 2>&1 &
disown
```
This runs in the background. Track progress with:
```bash
tail -f download_data.log
```
**Note**: If interrupted, simply re-run the same command: already-downloaded files are skipped automatically.

### Try the tool
To see SpectFbCalc in action rapidly, we provide a Jupyter Notebook with sample low-resolution data. 

1. Clone the repository and download the kernels as shown above.
2. Launch Jupyter: `jupyter notebook`
3. Open `template_spectfbcalc.ipynb` and run the cells. 