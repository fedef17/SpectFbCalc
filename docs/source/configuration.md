# Configuration File (config.yaml)
The software is designed to handle data from CMIP6/CMIP7 models. It supports three kernel types:
1. **Huang** kernels [^1]
2. **SPECTRAL** kernels [^2]

Where the first two are broadband.

## File Paths
Paths should be specified appropriately based on how datasets are organized: 
- Use * for standard wildcards. 
- Use **/* if data is organized into subdirectories. 

## Paramaters
- **`anomaly_method`** : the tool allows different methods for calculating anomalies and handling climatology. Options are *climatology* (monthly averaged) or *running_mean*.
- **`save_pattern`** : if **`True`**, saves the full spatial anomaly patterns alongside the global means.
- **`num_year_regr`** : number of years to group into chunks for the linear regression feedback calculation.
- **`time_range_clim`** / **`time_range_exp`** : restricts the analysis to a specific temporal window. Leave **`time_range_exp`** empty to automatically match the reference dataset's length.

[^1]: Dataset: Huang, Yi (2022), “ERA-interim reanalysis based radiative kernels”, Mendeley Data, V1, doi: 10.17632/3drx8fmmz9.1 
Huang, Y., Y. Xia, and X. Tan (2017), On the pattern of CO2 radiative forcing and poleward energy transport, J. Geophys. Res. Atmos., 122, 10,578–10,593. https://doi.org/10.1002/2017JD027221 
[^2]: Dataset: Della Fera, S. (2026). Clear-sky Spectral Kernels [Data set]. Zenodo. https://doi.org/10.5281/zenodo.21245639
Della Fera, S., Fabiano, F., Raspollini, P., Ridolfi, M., Von Hardenberg, J., & Cortesi, U. (2025). Reproducing and Attributing IASI Radiance Trends with EC-Earth Climate Model Simulations. Journal of Climate, 38(23), 6943-6959. https://doi.org/10.1175/JCLI-D-25-0034.1 