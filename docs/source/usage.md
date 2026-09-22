# Usage
## Running the Notebook
This tool can be used in Jupyter Notebook or directly in VS Code.
To execute the code interactively:
1. Activate the environment:
```bash
conda activate spectfbcalc
```
2. Open Jupyter Notebook and run:
```bash
jupyter notebook
```
3. Navigate to **`template_spectfbcalc.ipynb`** and execute the cells to see an example usage.
**Important**: see instruction to download the dataset in the notebook or in the README.md.

## Initial Setup in the Notebook
Before running the core functions for calculating anomalies and feedbacks, create a **`config.yaml/`** file by copying **`config_template.yaml/`** and modifying paths and configurations based on your purpose: 
```bash
cp config_template.yaml config.yaml
```
Then the data needs to be processed trough the **`sfc.preprocess_data`** function. The user must define the config file path:
```bash
config = "path/to/config/file/config.yaml"
control, experiment, kernel = sfc.preprocess_data(config_file=config_file, ker='KERNEL')
```
It may also be necessary to define the list of **`raw_variables`** if it differs from the default: **`STD_VARS_NOALB`**. The list of currently available lists is shown below: 
```bash
STD_VARS = {"hus", "rlut", "rsdt", "rlutcs", "alb", "rsut", "rsutcs", "ta", "tas", "ts"}
STD_VARS_LOGQ = {"hus_log", "rlut", "rsdt", "rlutcs", "alb", "rsut", "rsutcs", "ta", "tas", "ts"}
STD_VARS_NOALB = {"hus", "rlut", "rsdt", "rlutcs", "rsut", "rsutcs", "ta", "tas", "ts", "rsds", "rsus"}
STD_VARS_ECE4 = {"hus", "rlut", "rsdt", "rlntcs", "rsut", "rsntcs", "alb", "ta", "tas", "ts"}
```
Users can define a new one themselves to replace the default.
Once the pre-processing is complete, everything is ready to carry out the necessary analyses

## The core functions
The **`spectfbcalc_lib`** contains several functions for radiative feedback calculations: 
- Core functions for radiance anomaly calculation: these functions compute radiance anomalies under both clear-sky and all-sky conditions. 
- Feedback calculations: includes functions for computing the following radiative feedbacks either individually or all at once: Planck, Albedo, Lapse-rate, Water-vapor and Cloud. It is also possible to calculate inter-annual feedback.
- Spectral computation: dedicated workflows (**`calc_fb_spectral`**)
- Feedback pattern calculation: it is possible to obtain the feedback pattern for each lat/lon point using a flag (**`save_pattern=True`**) in the core functions.

## What the user can do 
The user can choose what to compute according to its need:
- compute individual radiance anomalies;
- compute all radiance anomalies at once using the **`calc_anoms`** function;
- compute all broadband feedback at once using **`calc_fb_from_exp`**;
- isolate the short-term climate variability using the **`calc_fb_interannual`** function;
- compute one feedback at a time using **`calc_single_feedback`**; 
- analyze the spectral longwave dimension changes using **`calc_fb_spectral`**