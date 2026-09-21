import os
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd
import re
import glob
import fnmatch
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from climtools import climtools_lib as ctl

from matplotlib.backends.backend_pdf import PdfPages

# ---------- Saving ouput -------------
def save_feedback_output(output, out_path_txt, out_path_nc=None):
    """
    Save feedback regression results to text and NetCDF files.

    The function writes global feedback coefficients (with errors) to a .txt file,
    and optionally saves spatial patterns of feedback slopes and errors to a .nc file.

    Parameters
    ----------
    output : dict
    Dictionary containing feedback regression results with the following keys:
        - "fb_coeffs": dict
            Radiative feedback coefficients and errors for clear-sky and all-sky components.
            Keys are tuples of the form (sky_type, component), e.g. ('cld', 'planck-surf').

        - "fb_cloud": float or None
            Global mean cloud radiative feedback coefficient.

        - "fb_cloud_err": float or None
            Estimated uncertainty of the cloud radiative feedback.

        - "fb_pattern": dict or None
            Spatial patterns of feedback slopes and errors.
            Keys are tuples (sky_type, component), values are (slope, stderr) DataArrays.
    out_path_txt : str
        Path to the output .txt file (always required).
    out_path_nc : str, optional
        Path to the output .nc file. Required if feedback patterns are included.
    """
    if not isinstance(output, dict):
        raise ValueError("Expected output to be a dict with keys: fb_coeffs, fb_cloud, fb_cloud_err, fb_pattern")

    fb_coeffs = output.get("fb_coeffs")
    fb_pattern = output.get("fb_pattern")

    # ---------- TXT output ----------
    if out_path_txt:
        with open(out_path_txt, "w") as f:
            def write_block(name, results_dict):
                f.write(f"{name} feedback:\n")
                for key in ['planck-surf', 'planck-atmo', 'lapse-rate', 'water-vapor', 'water-vapor-lw', 'albedo', 'cloud']:
                    result = results_dict.get((name, key))
                    if result is not None:
                        f.write(f"{key.replace('_', '-') + ' feedback'}: {result.slope:.4f}\n")
                        f.write(f"{key.replace('_', '-') + ' feedback error'}: {result.stderr:.4f}\n")

            if fb_coeffs is not None:
                write_block("cld", fb_coeffs)
                write_block("clr", fb_coeffs)

    print(f"Saved feedback coefficients to {out_path_txt}, new one")

    # ---------- NetCDF output ----------
    if out_path_nc and fb_pattern is not None:
        # Dummy lat/lon if not available in pattern
        lat = np.linspace(-90, 90, 73)
        lon = np.linspace(0, 360, 144, endpoint=False)

        data_vars = {}

        # Add standard component patterns
        if fb_pattern:
            for (cloud_type, component), (slope, stderr) in fb_pattern.items():
                key_slope = f"{cloud_type}_{component}_slope"
                key_stderr = f"{cloud_type}_{component}_stderr"
                safe_key_slope = key_slope.replace("(", "").replace(")", "").replace(",", "").replace("'", "").replace(" ", "_")
                data_vars[safe_key_slope] = (["lat", "lon"], slope.data if hasattr(slope, "data") else slope)
                safe_key_stderr = key_stderr.replace("(", "").replace(")", "").replace(",", "").replace("'", "").replace(" ", "_")
                data_vars[safe_key_stderr] = (["lat", "lon"], stderr.data if hasattr(stderr, "data") else stderr)

        ds = xr.Dataset(data_vars=data_vars, coords={"lat": lat, "lon": lon})
        ds.to_netcdf(out_path_nc)
        print(f"Saved feedback spatial patterns to {out_path_nc}")

# -------- Spatial pattern plot -----------
def plot_fb_pattern(slope, stderr, title, output_folder, filename_prefix="fb_pattern", pdf=None):
    """
    Plot spatial patterns of feedback slopes and standard errors.

    Creates two global maps (one for slope, one for stderr) using cartopy with coastlines, 
    borders, and gridlines. Each map is saved as a PNG, and optionally appended to a 
    combined multi-page PDF.

    Parameters
    ----------
    slope : xarray.DataArray
        2D field of feedback slopes (W/m²/K).
    stderr : xarray.DataArray
        2D field of feedback standard errors (W/m²/K).
    title : str
        Plot title applied to both maps.
    output_folder : str
        Directory where PNG files will be saved (created if missing).
    filename_prefix : str, optional
        Prefix for the output PNG filenames (default: "fb_pattern").
    pdf : PdfPages, optional
        If provided, figures are also added as pages to the given PDF object.
    """
    os.makedirs(output_folder, exist_ok=True)

    def plot_field(field, cmap, label, fname):
        fig, ax = plt.subplots(figsize=(10, 4.5),
                               subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)})
        # just plot normally, without add_label
        im = field.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            robust=True,
            cbar_kwargs={"label": label}
        )

        # Continents & borders
        ax.coastlines(linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, alpha=0.5)
        ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3, zorder=-1)

        # Gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False

        # Set title manually
        ax.set_title(title, fontsize=16, weight="bold")

        fig.tight_layout()
        fig.savefig(os.path.join(output_folder, fname), dpi=300)
        if pdf:
            pdf.savefig(fig)
        plt.close(fig)

    plot_field(slope, "RdBu_r", "W/m²/K", f"{filename_prefix}_slope.png")
    plot_field(stderr, "viridis", "W/m²/K", f"{filename_prefix}_stderr.png")

def save_all_fb_patterns_to_pdf(ds: xr.Dataset, output_folder: str, pdf_name: str = "all_fb_patterns.pdf", components: list = None, skies: list = None, plot_function=plot_fb_pattern, run_label: str = "exp"):
    """
    Generate and save feedback pattern maps for multiple components and sky conditions.

    For each (component, sky) pair in the dataset, this function creates slope and stderr
    maps using the provided plotting function. Each map is saved as an individual PNG and 
    appended to a combined multi-page PDF.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing feedback fields, with variables named like "<sky>_<component>_slope"
        and "<sky>_<component>_stderr".
    output_folder : str
        Directory where PNG files and the combined PDF will be saved.
    pdf_name : str, optional
        Name of the combined PDF file (default: "all_fb_patterns.pdf").
    components : list, optional
        List of feedback components to plot (default: 
        ["albedo", "water-vapor", "lapse-rate", "planck-atmo", "planck-surf", "cloud"]).
    skies : list, optional
        List of sky conditions to plot (default: ["clr", "cld"]).
    plot_function : callable
        Function used to create each map. Must accept slope, stderr, title, output_folder,
        filename_prefix, and pdf as arguments.
    run_label : str, optional
        Label for the experiment, added to plot titles (e.g., "cold", "warm").

    Output
    ------
    - Saves one PNG per (component, sky) pair in `output_folder`.
    - Saves a multi-page PDF combining all maps into `output_folder/pdf_name`.
    """
    if components is None:
        components = ["albedo", "water-vapor", "lapse-rate", "planck-atmo", "planck-surf", "cloud"]
    if skies is None:
        skies = ["clr", "cld"]

    os.makedirs(output_folder, exist_ok=True)
    pdf_path = os.path.join(output_folder, pdf_name)

    with PdfPages(pdf_path) as pdf:
        for comp in components:
            for sky in skies:
                var_slope = f"{sky}_{comp}_slope"
                var_stderr = f"{sky}_{comp}_stderr"

                if var_slope not in ds or var_stderr not in ds:
                    print(f"⚠️  {var_slope} or {var_stderr} not found, skipping.")
                    continue

                slope = ds[var_slope]
                stderr = ds[var_stderr]

                # ✅ build title with your logic
                feedback_name = comp.capitalize()
                cloud_label = sky.upper() if sky == "cld" else sky
                title = f"{feedback_name} ({cloud_label}) - {run_label}"

                filename_prefix = f"fb_pattern_{sky}_{comp}"

                try:
                    plot_function(
                        slope=slope,
                        stderr=stderr,
                        title=title,
                        output_folder=output_folder,
                        filename_prefix=filename_prefix,
                        pdf=pdf
                    )
                except Exception as e:
                    print(f"Error for {comp} - {sky}: {e}")
                    raise e

    print(f"Combined feedback PDF saved to: {pdf_path}")

# -------- Gregory plot -----------
def plot_feedback_slope(feedback_file, sim_label="exp1", save_path="feedback_summary.png"):
    """
    Parse a feedback summary file and plot slopes with 95% CI.

    Reads slopes and errors for cloud ("cld") and clear-sky ("clr") feedbacks 
    from a text file, builds a DataFrame, and plots each component with error bars. 
    Blue = cld, Orange = clr. Saves the plot as PNG.

    Parameters
    ----------
    feedback_file : str
        Path to the feedback summary text file.
    sim_label : str, optional
        Label for the simulation (used in the plot title).
    save_path : str, optional
        Output filename for the saved figure.
    """
    feedback_data = {}
    section = ""
    
    with open(feedback_file, "r") as f:
        for line in f:
            line = line.strip()

            if "cld feedback" in line.lower():
                section = "cld"
            elif "clr feedback" in line.lower():
                section = "clr"
            else:
                # Match value
                match = re.match(r"(.+?) feedback: ([-\d\.Ee]+)", line)
                if match:
                    name, val = match.groups()
                    key = f"{section}_{name.strip()}"
                    feedback_data[key] = float(val)
                    continue

                # Match error
                match_err = re.match(r"(.+?) feedback error: ([-\d\.Ee]+)", line)
                if match_err:
                    name, val = match_err.groups()
                    key = f"{section}_{name.strip()}_error"
                    feedback_data[key] = float(val)

    # Prepare DataFrame for plotting
    records = []

    for key in feedback_data:
        if "_error" not in key:
            base = key  # e.g., "cld_planck-surf"
            name_only = base.replace("cld_", "").replace("clr_", "").replace("-", "_")
            err_key = base + "_error"

            records.append({
                "feedback": name_only.capitalize(),
                "cs": "clr_" in base,
                "slope": feedback_data[base],
                "std_err": feedback_data.get(err_key, np.nan),
            })

    df = pd.DataFrame(records)
    feedbacks = df["feedback"].unique()
    x_base = np.arange(len(feedbacks))

    # Plot
    plt.figure(figsize=(10, 5))
    width = 0.25

    for i, fb in enumerate(feedbacks):
        for cs in [False, True]:
            row = df[(df["feedback"] == fb) & (df["cs"] == cs)]
            if not row.empty:
                y = row["slope"].values[0]
                err = row["std_err"].values[0]
                if pd.isna(y) or pd.isna(err):
                    continue
                err = 1.96 * err
                x = i + (-width if not cs else width)

                # Choose color
                color = "#1f77b4" if not cs else "#ff7f0e"  # blue = cld, orange = clr
                label = "cld" if not cs else "clr"
                plt.errorbar(
                    x, y, yerr=err, fmt='o', color=color, capsize=5, ecolor='gray',
                    label=label if i == 0 else None  # only add legend once
                )

    plt.axhline(0, color='black', linestyle='--')
    plt.xticks(x_base, feedbacks, rotation=45, ha='right')
    plt.ylabel("Slope [W/m²/K]")
    plt.title(f"Gregory Feedback Slopes ± 95% CI ({sim_label})", fontsize=20)
    plt.grid(True)
    plt.legend(title="Sky condition")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

# --------- Plot dRt for each feedback and for different parameters change variation ------------
# Add plot of each parameter with all change variations value?
def plot_dRt_fb_all_params(base_folder, xlabel, ylabel, title, feedback_file, output_file=None, subfolder_pattern="*", param_map=None, param_order=None, show_zero_line=True, mapping_file="param_mappings.yml", model_type=None):
    """
    Plot dRt values for different parameters. Supports flexible mappings via dictionary or YAML.

    Parameters
    ----------
    base_folder : str
        The base folder containing the subfolders with the data files.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    title : str
        Title for the plot.
    feedback_file : str
        The specific feedback file to use (e.g., "dRt_planck-surf_global_clr_climatology-HUANGkernels.nc").
    output_file : str, optional
        File path to save the plot (default: None, plot shown interactively).
    subfolder_pattern : str, optional
        Subfolder pattern to match files (default: "*").
    param_map : dict, optional
        Dictionary mapping subfolder name patterns to pretty parameter names.
        Example: {"pi*a": "ENTRORG", "pi*b": "RPRCON"}.
    param_order : list, optional
        Order of parameters to plot (default: alphabetical).
    show_zero_line : bool, optional
        Whether to draw a horizontal line at 0 (default: True).
    mapping_file : str, optional
        YAML file containing mappings for multiple model types.
    model_type : str, optional
        If provided, will select mapping from YAML by key (e.g., "ece3", "ece4").
    """

    # --- Load mapping if not provided ---
    if param_map is None and os.path.exists(mapping_file):
        with open(mapping_file) as f:
            mappings = yaml.safe_load(f)
        if model_type and model_type in mappings:
            param_map = mappings[model_type]
            print(f"✅ Loaded parameter mapping for {model_type} from {mapping_file}")
        else:
            print(f"⚠️ No model_type provided or not found in {mapping_file}. Will use raw subfolder names.")

    param_names = []
    dRt_values = []

    subfolders = glob.glob(os.path.join(base_folder, subfolder_pattern))

    for subfolder_path in subfolders:
        if os.path.isdir(subfolder_path):
            subfolder_name = os.path.basename(subfolder_path)

            # Match subfolder name against param_map
            matched_param = subfolder_name
            if param_map:
                for pattern, real_name in param_map.items():
                    if fnmatch.fnmatch(subfolder_name, pattern):
                        matched_param = real_name
                        break

            file_path = os.path.join(subfolder_path, feedback_file)
            if not os.path.exists(file_path):
                continue

            try:
                ds = xr.open_dataset(file_path)
                if "__xarray_dataarray_variable__" in ds.data_vars:
                    dRt_mean = float(ds["__xarray_dataarray_variable__"].values)
                    param_names.append(matched_param)
                    dRt_values.append(dRt_mean)
            except Exception as e:
                print(f"⚠️ Error processing {file_path}: {e}")

    df = pd.DataFrame({"param": param_names, "dRt": dRt_values})

    # Sorting if requested
    if param_order:
        df["param"] = pd.Categorical(df["param"], categories=param_order, ordered=True)
        df = df.sort_values("param")
    else:
        df = df.sort_values("param")

    # --- Plot ---
    plt.figure(figsize=(8, 6))
    colors = plt.cm.tab10(np.arange(len(df)))

    # scatter points
    for i, (param, dRt) in enumerate(zip(df["param"], df["dRt"])):
        plt.scatter(i, dRt, color=colors[i % len(colors)], s=100)

    # xticks with custom colors
    xticks = range(len(df))
    plt.xticks(xticks, df["param"], rotation=45, ha="right")

    ax = plt.gca()
    for ticklabel, color in zip(ax.get_xticklabels(), colors):
        ticklabel.set_color(color)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, fontweight="bold", fontsize=12)
    if show_zero_line:
        plt.axhline(0, color="black", linestyle="--", linewidth=1)

    if output_file:
        plt.savefig(output_file, bbox_inches="tight")
    plt.show()

    return df


# --------- Time series ------------
def plot_toa_anomaly(experiment, dRt_dict, title, sky="clr", output_file=None):
    """
    Plot Net TOA anomaly (simulation - control climatology) alongside dRt component time series.

    Parameters
    ----------
    experiment : Experiment
        The experiment object containing the computed anomalies (ds_anom).
    dRt_dict : dict
        Dictionary of radiative anomalies, typically loaded via open_dRt() 
        or returned by calc_anoms(). Keys must be (sky, component).
    title : str
        Title for the plot.
    sky : {'clr', 'cld'}
        Sky condition to plot.
    output_file : str, optional
        Path to save the figure. If None, plt.show() is called.
    """

    var_toa = 'net_toa_cs' if sky == 'clr' else 'net_toa'
    if var_toa not in experiment.ds_anom:
        raise KeyError(f"Variable {var_toa} not found in ds_anom. Have you executed compute_net_TOA() before compute_anomalies()?")
        
    # Fai la media globale e poi la media annuale
    toa_anom = experiment.ds_anom[var_toa]
    toa_anom_gm = ctl.global_mean(toa_anom)
    
    time_coord = 'time' if 'time' in toa_anom_gm.coords else 'time_counter'
    toa_annual = toa_anom_gm.groupby(f'{time_coord}.year').mean(time_coord)
    
    years = toa_annual['year'].values
    vals_toa = toa_annual.values

    expected_comps = ['planck-surf', 'planck-atmo', 'lapse-rate', 'water-vapor', 'albedo']
    if sky == 'cld':
        expected_comps.append('cloud')

    dRt_components = []
    comp_labels = []

    for comp in expected_comps:
        if (sky, comp) in dRt_dict:
            da = dRt_dict[(sky, comp)]
            if time_coord in da.coords or time_coord in da.dims:
                da = da.groupby(f'{time_coord}.year').mean(time_coord)
            
            dRt_components.append(da)
            comp_labels.append(comp.replace('-', ' ').capitalize())
            
    if not dRt_components:
        raise ValueError(f"No dRt components found in the dictionary for sky='{sky}'")
    
    aligned = xr.align(*dRt_components, join="inner")
    dRt_sum = sum(aligned)

    plt.figure(figsize=(12, 5))
    
    plt.plot(years.astype(int), vals_toa, color="black", marker="o", linestyle="-", label="Net TOA Anomaly (Model)")

    color_cycle = ["tab:blue", "tab:green", "tab:purple", "tab:orange", "tab:cyan", "tab:brown"]
    for i, (comp_da, lab) in enumerate(zip(aligned, comp_labels)):
        plt.plot(
            comp_da['year'].values.astype(int), 
            comp_da.values, 
            marker="s", linestyle="--", alpha=0.7,
            label=lab, color=color_cycle[i % len(color_cycle)]
        )

    plt.plot(
        dRt_sum['year'].values.astype(int), 
        dRt_sum.values, 
        color="red", linestyle="-", linewidth=2.5,
        label="Sum of dRt Components (Kernels)"
    )

    plt.xlabel("Year")
    plt.ylabel("Radiative Anomaly [W/m²]")
    plt.title(f"{title} ({sky.upper()} sky)", fontsize=14, fontweight='bold')
    plt.xticks(years, [str(int(y)) for y in years])
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Plot saved: {output_file}")
    else:
        plt.show()
    plt.close()