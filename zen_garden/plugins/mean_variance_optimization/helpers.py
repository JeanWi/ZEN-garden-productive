from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def _get_capex_specific(optimization_setup):
    """
    Reads all capex parameters from the optimization setup.
    """
    capex_specific_conversion = optimization_setup.parameters.capex_specific_conversion
    capex_specific_conversion = capex_specific_conversion.rename(
        {'level_0': 'set_technologies',
         'node': 'set_location',
         'year': 'set_time_steps_yearly'}
    )
    capex_specific_conversion = capex_specific_conversion.expand_dims(
        {"set_capacity_types": ["energy"]}
    )
    capex_specific_storage = optimization_setup.parameters.capex_specific_storage
    capex_specific_storage = capex_specific_storage.rename(
        {'set_storage_technologies': 'set_technologies',
         'set_nodes': 'set_location'}
    )

    capex_specific_transport = optimization_setup.parameters.capex_specific_transport
    capex_specific_transport = capex_specific_transport.rename(
        {'set_transport_technologies': 'set_technologies',
         'set_edges': 'set_location'}
    )
    capex_specific_transport = capex_specific_transport.expand_dims(
        {"set_capacity_types": ["power"]}
    )

    capex_specific = xr.concat(
        [
            capex_specific_conversion,
            capex_specific_storage,
            capex_specific_transport,
        ],
        dim="set_technologies",
        join="outer"
    )
    return capex_specific


def _get_sd(optimization_setup):
    """
    Reads all sd values from file
    """
    tech_capex_path = Path(optimization_setup.analysis.dataset) / "mean_variance" / "technology_capex"
    technologies = list(optimization_setup.sets["set_technologies"])

    sd = pd.read_csv(tech_capex_path / "sd.csv", index_col=0)

    sd_xr = xr.DataArray(
        sd.loc[technologies, "value"].values,
        dims=("set_technologies",),
        coords={"set_technologies": technologies},
    )

    return sd_xr


def _get_correlation(optimization_setup):
    """
    Reads all correlations from file and preprocess them
    """
    tech_capex_path = Path(optimization_setup.analysis.dataset) / "mean_variance" / "technology_capex"
    technologies = list(optimization_setup.sets["set_technologies"])
    time_steps_yearly = optimization_setup.sets["set_time_steps_yearly"]
    nodes = list(optimization_setup.sets["set_nodes"])
    edges = list(optimization_setup.sets["set_edges"])

    correlation_df = pd.read_csv(tech_capex_path / "correlation.csv", index_col=0)
    correlation_np = correlation_df.to_numpy()
    lam = 1e-4
    correlation_reg = (1 - lam) * correlation_np + lam * np.eye(correlation_np.shape[0])
    correlation = pd.DataFrame(
        correlation_reg,
        index=correlation_df.index,
        columns=correlation_df.index
    )

    auto_correlation = pd.read_csv(tech_capex_path / "autocorrelation.csv", index_col=0)

    # Correlation matrix
    corr_xr = xr.DataArray(
        correlation.loc[technologies, technologies].values,
        dims=("set_technologies_i", "set_technologies_j"),
        coords={
            "set_technologies_i": technologies,
            "set_technologies_j": technologies,
        },
    )

    # Autocorrelation matrix
    auto_corr_xr = xr.DataArray(
        auto_correlation.loc[technologies, "value"].values,
        dims=("set_technologies",),
        coords={"set_technologies": technologies},
    )
    time_xr = xr.DataArray(
        time_steps_yearly,
        dims=("set_time_steps_yearly",),
        coords={"set_time_steps_yearly": time_steps_yearly},
    )
    dt = abs(
        time_xr.rename(set_time_steps_yearly="set_time_steps_yearly_i")
        - time_xr.rename(set_time_steps_yearly="set_time_steps_yearly_j")
    )
    time_corr = auto_corr_xr ** dt

    # Full correlation
    full_corr = (
            corr_xr
            * time_corr.rename(set_technologies="set_technologies_i")
    )

    return full_corr


def generate_covariance_pairs(absolute_sd_per_tech, corr_df):
    valid_tech_cap = set(absolute_sd_per_tech.index)
    valid_df = pd.DataFrame(list(valid_tech_cap), columns=["tech", "cap"])
    pairs = valid_df.add_suffix("_i").merge(valid_df.add_suffix("_j"), how="cross")
    pairs = pairs.merge(corr_df, on=["tech_i", "tech_j"], how="inner")
    return pairs


def calculate_correlation_matrix(optimization_setup):
    corr_xr = _get_correlation(optimization_setup)
    corr_series = (
        corr_xr.to_series()
        .groupby(level=["set_technologies_i", "set_technologies_j"])
        .mean()
        .dropna()
    )
    corr_series = corr_series[corr_series != 0]
    corr_df = corr_series.reset_index()
    corr_df.columns = ["tech_i", "tech_j", "correlation"]
    return corr_df


def calculate_absolute_sd(optimization_setup):
    # Calculate absolute SD per technology
    capex_specific_xr = _get_capex_specific(optimization_setup)
    relative_sd_xr = _get_sd(optimization_setup)
    absolute_sd_xr = (capex_specific_xr * relative_sd_xr).stack(
        all_dims=["set_technologies", "set_location", "set_time_steps_yearly", "set_capacity_types"]
    ).dropna("all_dims")
    absolute_sd_per_tech = (
        absolute_sd_xr.to_series()
        .groupby(level=["set_technologies", "set_capacity_types"])
        .mean()
        .dropna()
    )
    absolute_sd_per_tech = absolute_sd_per_tech[absolute_sd_per_tech != 0]
    return absolute_sd_per_tech
