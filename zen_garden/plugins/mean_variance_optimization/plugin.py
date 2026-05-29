import linopy as lp
import pandas as pd
import numpy as np
import xarray as xr
import json
from pathlib import Path
import logging
from tqdm import tqdm

from zen_garden.plugin_system.events import Event, EventPublisher


config = {
    "weighting_factor": None,
    "include_variances_for": ["technology_capex", "technology_opex", "import", "export", "demand_shedding"],
}


def _get_capex_specific(optimization_setup):
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
    tech_capex_path = Path(optimization_setup.analysis.dataset) / "mean_variance" / "technology_capex"
    technologies = list(optimization_setup.sets["set_technologies"])
    time_steps_yearly = optimization_setup.sets["set_time_steps_yearly"]
    nodes = list(optimization_setup.sets["set_nodes"])
    edges = list(optimization_setup.sets["set_edges"])
    locations = nodes + edges
    set_capacity_types = ["power", "energy"]

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

def _spatial_and_time_correlation(optimization_setup, quadratic_term):
    # Capacity additions
    capacity_addition = optimization_setup.model.variables["capacity_addition"]

    # Absolute standard deviation of capex
    capex_specific_xr = _get_capex_specific(optimization_setup)
    relative_sd_xr = _get_sd(optimization_setup)
    absolute_sd_xr = (capex_specific_xr * relative_sd_xr).stack(
        all_dims=["set_technologies", "set_location", "set_time_steps_yearly", "set_capacity_types"]
    ).dropna("all_dims")
    absolute_sd_dict = absolute_sd_xr.to_series().to_dict()

    # Correlation matrix
    valid_keys = set(absolute_sd_dict.keys())

    corr_xr = _get_correlation(optimization_setup)
    corr_series = corr_xr.to_series().dropna()
    corr_series = corr_series[corr_series != 0]

    # Build pairs DataFrame from valid_keys
    valid_df = pd.DataFrame(list(valid_keys), columns=["tech", "loc", "time", "cap"])
    pairs = valid_df.add_suffix("_i").merge(valid_df.add_suffix("_j"), how="cross")

    # Merge with correlation values (loc/cap correlation = 1, so just look up tech×time)
    pairs = pairs.merge(
        corr_series.rename("correlation"),
        left_on=["tech_i", "tech_j", "time_i", "time_j"],
        right_index=True,
        how="inner"
    )

    pair_dict = pairs.set_index(
        ["tech_i", "tech_j", "time_i", "time_j", "loc_i", "loc_j", "cap_i", "cap_j"]
    )["correlation"].to_dict()

    #
    # techs = list({k[0] for k in valid_keys})
    # locs = list({k[1] for k in valid_keys})
    # times = list({k[2] for k in valid_keys})
    # caps = list({k[3] for k in valid_keys})
    #
    # corr_sub = corr_xr.sel(
    #     set_technologies_i=techs,
    #     set_technologies_j=techs,
    #     set_time_steps_yearly_i=times,
    #     set_time_steps_yearly_j=times,
    #     set_locations_i=locs,
    #     set_locations_j=locs,
    #     set_capacity_types_i=caps,
    #     set_capacity_types_j=caps,
    # )
    #
    # stacked = corr_sub.stack(all_dims=list(corr_sub.dims)).dropna("all_dims")
    # stacked = stacked.where(stacked != 0, drop=True)
    #
    # pair_dict = {
    #     idx: float(val)
    #     for idx, val in zip(stacked.indexes["all_dims"], stacked.values)
    #     if (idx[0], idx[4], idx[2], idx[6]) in valid_keys  # key_i
    #        and (idx[1], idx[5], idx[3], idx[7]) in valid_keys  # key_j
    # }
    #
    # valid_keys_list = list(valid_keys)
    #
    # pair_dict = {}
    # for key_i in tqdm(valid_keys_list, desc="Building pair_dict"):
    #     tech_i, loc_i, time_i, cap_i = key_i
    #     for key_j in valid_keys_list:
    #         tech_j, loc_j, time_j, cap_j = key_j
    #         value = float(corr_xr.sel(
    #             set_technologies_i=tech_i,
    #             set_technologies_j=tech_j,
    #             set_time_steps_yearly_i=time_i,
    #             set_time_steps_yearly_j=time_j,
    #             set_locations_i=loc_i,
    #             set_locations_j=loc_j,
    #             set_capacity_types_i=cap_i,
    #             set_capacity_types_j=cap_j,
    #         ))
    #         if np.isfinite(value) and value != 0:
    #             pair_dict[(tech_i, tech_j, time_i, time_j, loc_i, loc_j, cap_i, cap_j)] = value

    #
    # stacked_corr = corr_xr.stack(all_dims=list(corr_xr.dims))
    #
    # pair_dict = {}
    # for idx, value in zip(stacked_corr.indexes["all_dims"], stacked_corr.values):
    #     if not np.isfinite(value) or value == 0:
    #         continue
    #     tech_i, tech_j, time_i, time_j, loc_i, loc_j, cap_i, cap_j = idx
    #     key_i = (tech_i, loc_i, time_i, cap_i)
    #     key_j = (tech_j, loc_j, time_j, cap_j)
    #     if key_i in valid_keys and key_j in valid_keys:
    #         pair_dict[idx] = value

    #
    # stacked = corr_xr.stack(all_dims=corr_xr.dims)
    # valid_entries = absolute_sd_xr.notnull() & (absolute_sd_xr != 0)
    # mask_i = valid_entries.sel(
    #     set_technologies=stacked["set_technologies_i"],
    #     set_location=stacked["set_locations_i"],
    #     set_time_steps_yearly=stacked["set_time_steps_yearly_i"],
    #     set_capacity_types=stacked["set_capacity_types_i"],
    # )
    #
    # mask_j = valid_entries.sel(
    #     set_technologies=stacked["set_technologies_j"],
    #     set_location=stacked["set_locations_j"],
    #     set_time_steps_yearly=stacked["set_time_steps_yearly_j"],
    #     set_capacity_types=stacked["set_capacity_types_j"],
    # )
    #
    # mask = mask_i & mask_j
    # stacked_filtered = stacked.where(mask, drop=True)
    #
    # pair_dict = {
    #     tuple(idx): value
    #     for idx, value in zip(stacked_filtered.indexes["all_dims"], stacked_filtered.values)
    #     if np.isfinite(value) and value != 0
    # }
    # len(pair_dict)

    for tech_pair, correlation in tqdm(pair_dict.items(), total=len(pair_dict),
                                       desc="Constructing quadratic variance term for technology capex"):
        tech_i, tech_j, time_i, time_j, loc_i, loc_j, cap_i, cap_j = tech_pair

        absolute_sd_1 = absolute_sd_dict[(tech_i, loc_i, time_i, cap_i)]
        absolute_sd_2 = absolute_sd_dict[(tech_j, loc_j, time_j, cap_j)]

        capacity_addition_1 = capacity_addition.sel(
            set_technologies=tech_i, set_time_steps_yearly=time_i,
            set_location=loc_i, set_capacity_types=cap_i)
        capacity_addition_2 = capacity_addition.sel(
            set_technologies=tech_j, set_time_steps_yearly=time_j,
            set_location=loc_j, set_capacity_types=cap_j)

        quadratic_term += correlation * absolute_sd_1 * absolute_sd_2 * capacity_addition_1 * capacity_addition_2

    return quadratic_term

def _only_technology_correlation(optimization_setup, quadratic_term):
    """Simplified variance term: aggregate capacity additions over locations and time steps,
    and compute correlations only per technology pair (not per location/time).

    Introduces an auxiliary variable ``capacity_addition_tech_agg`` (one per technology /
    capacity-type pair) that equals the sum of ``capacity_addition`` over all locations and
    yearly time steps, and constrains it accordingly.  The quadratic variance term is then
    built from products of these scalar variables, which linopy can handle as a proper QP.
    """
    model = optimization_setup.model
    capacity_addition = model.variables["capacity_addition"]

    # ------------------------------------------------------------------ #
    # 1. New variable:  C_agg[tech, cap] ≥ 0                            #
    # ------------------------------------------------------------------ #
    technologies = list(optimization_setup.sets["set_technologies"])
    set_capacity_types = ["power", "energy"]

    if "capacity_addition_tech_agg" in model.variables:
        capacity_addition_tech_agg = model.variables["capacity_addition_tech_agg"]
    else:
        capacity_addition_tech_agg = model.add_variables(
            lower=0,
            coords=[
                pd.Index(technologies, name="set_technologies"),
                pd.Index(set_capacity_types, name="set_capacity_types"),
            ],
            name="capacity_addition_tech_agg",
        )

    # ------------------------------------------------------------------ #
    # 2. Constraint:  C_agg[tech, cap] == Σ_{loc,t} capacity_addition   #
    # ------------------------------------------------------------------ #
    if "constraint_capacity_addition_tech_agg" not in model.constraints:
        capacity_addition_agg_expr = capacity_addition.sum(["set_location", "set_time_steps_yearly"])
        model.add_constraints(
            capacity_addition_tech_agg - capacity_addition_agg_expr == 0,
            name="constraint_capacity_addition_tech_agg",
        )

    # ------------------------------------------------------------------ #
    # 3. σ per (tech, cap_type): mean of absolute SD over loc / time    #
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # 4. Technology-pair correlation (averaged over time steps)          #
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # 5. Build (tech_i, cap_i) × (tech_j, cap_j) pairs with correlation #
    # ------------------------------------------------------------------ #
    valid_tech_cap = set(absolute_sd_per_tech.index)
    valid_df = pd.DataFrame(list(valid_tech_cap), columns=["tech", "cap"])
    pairs = valid_df.add_suffix("_i").merge(valid_df.add_suffix("_j"), how="cross")
    pairs = pairs.merge(corr_df, on=["tech_i", "tech_j"], how="inner")
    # pairs.loc[(pairs["tech_i"] == pairs["tech_j"]) & (pairs["cap_i"] != pairs["cap_j"]),"correlation"] = 0.98
    pairs.to_excel("Correlation.xlsx")
    absolute_sd_per_tech.to_excel("Absolute_SD.xlsx")
    weighting_factor = config.get("weighting_factor")

    # ------------------------------------------------------------------ #
    # 6. Quadratic term using the auxiliary variable                     #
    # ------------------------------------------------------------------ #
    log_rows = []

    for _, row in tqdm(pairs.iterrows(), total=len(pairs),
                       desc="Constructing quadratic variance term (technology-only correlation)"):
        tech_i, cap_i = row["tech_i"], row["cap_i"]
        tech_j, cap_j = row["tech_j"], row["cap_j"]
        correlation = row["correlation"]

        sigma_i = absolute_sd_per_tech[(tech_i, cap_i)]
        sigma_j = absolute_sd_per_tech[(tech_j, cap_j)]

        C_i = capacity_addition_tech_agg.sel(set_technologies=tech_i, set_capacity_types=cap_i)
        C_j = capacity_addition_tech_agg.sel(set_technologies=tech_j, set_capacity_types=cap_j)

        scalar_coeff = weighting_factor * correlation * sigma_i * sigma_j
        quadratic_term += scalar_coeff * C_i * C_j

        log_rows.append({
            "tech_i": tech_i,
            "cap_i": cap_i,
            "tech_j": tech_j,
            "cap_j": cap_j,
            "correlation": correlation,
            "sigma_i": sigma_i,
            "sigma_j": sigma_j,
            "weighting_factor": weighting_factor,
            "scalar_coeff (wf*corr*sigma_i*sigma_j)": scalar_coeff,
            "C_i_var": f"capacity_addition_tech_agg[{tech_i}, {cap_i}]",
            "C_j_var": f"capacity_addition_tech_agg[{tech_j}, {cap_j}]",
        })

    log_df = pd.DataFrame(log_rows)
    log_path = Path("quadratic_term_log.csv")
    log_df.to_csv(log_path, index=False)
    logging.getLogger(__name__).info(f"Quadratic term log written to {log_path.resolve()}")

    return quadratic_term

@EventPublisher.register(Event.after_model_construction)
def construct_mean_variance_objective(optimization_setup=None):


    quadratic_term = 0

    if "technology_capex" in config.get("include_variances_for"):
        # quadratic_term = _spatial_and_time_correlation(optimization_setup, quadratic_term)
        quadratic_term = _only_technology_correlation(optimization_setup, quadratic_term)


    optimization_setup.model.remove_objective()
    # rule = VarianceRules(optimization_setup)
    # objective = rule.constraint_variance_term()
    npv_term = optimization_setup.model.variables["net_present_cost"].sum("set_time_steps_yearly")

    objective = quadratic_term + npv_term
    sense = "min"
    optimization_setup.model.add_objective(objective, sense=sense)

