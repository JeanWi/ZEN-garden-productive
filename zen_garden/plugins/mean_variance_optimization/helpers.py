from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import scipy.sparse as sp
import itertools

def get_capex_specific(optimization_setup):
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

def _get_correlation_matrix(optimization_setup):
    """
    Reads all correlations from file and preprocess them.

    If no_correlation=True, returns identity correlation matrix (diagonal=1, off-diagonal=0).
    """
    tech_capex_path = Path(optimization_setup.analysis.dataset) / "mean_variance" / "technology_capex"

    tec_correlation_df = pd.read_csv(tech_capex_path / "correlation.csv", index_col=0)

    R_T = tec_correlation_df.values
    R_T = sp.csr_matrix(R_T)

    index = list(itertools.product(
        tec_correlation_df.index,
    ))
    index_map = {
        key: i for i, key in enumerate(index)
    }

    return (index_map, R_T)


def _calculate_absolute_sd(optimization_setup):
    # Calculate absolute SD per technology
    capex_specific_xr = get_capex_specific(optimization_setup)
    relative_sd_xr = _get_sd(optimization_setup)
    absolute_sd_xr = (capex_specific_xr * relative_sd_xr).stack(
        all_dims=["set_technologies", "set_location", "set_time_steps_yearly", "set_capacity_types"]
    ).dropna("all_dims")
    absolute_sd_per_tech = (
        absolute_sd_xr.to_series()
    )
    absolute_sd_per_tech = absolute_sd_per_tech[absolute_sd_per_tech != 0]
    return absolute_sd_per_tech


def _determine_allowed_aggregation(df):
    results = {}

    dims = ["set_location", "set_time_steps_yearly", "set_capacity_types"]

    for tech, sub in df.groupby("set_technologies"):

        results[tech] = {}

        aggregation_dimensions = []
        for dim in dims:
            # check if within-dim values are all identical
            is_constant = sub.groupby(dim)["value"].mean().nunique() == 1
            if is_constant:
                aggregation_dimensions.append(dim)

        results[tech] = aggregation_dimensions

    return results

def _aggregate_by_structure(df, aggregation):
    out = []

    for tech, sub in df.groupby("set_technologies"):

        var_dims = aggregation[tech]
        group_dims = [
            dim for dim in [
                "set_location",
                "set_time_steps_yearly",
                "set_capacity_types"
            ]
            if dim not in var_dims
        ]

        agg = sub.groupby(
            ["set_technologies"] + group_dims
        )["value"].mean().reset_index()


        # replace collapsed dims with "aggregated"
        for dim in ["set_location", "set_time_steps_yearly", "set_capacity_types"]:
            if dim not in group_dims:
                agg[dim] = "aggregated"

        out.append(agg)

    return pd.concat(out, ignore_index=True)

def is_psd(A, tol=1e-8):
    eigvals = np.linalg.eigvalsh(A)
    return eigvals.min() >= -tol


def _get_covariance_matrix(absolute_sd_per_tech_aggregated, correlation_matrix, correlation_matrix_index_map):
    tuples = list(
        absolute_sd_per_tech_aggregated[
            [
                "set_technologies",
                "set_location",
                "set_time_steps_yearly",
                "set_capacity_types",
            ]
        ].itertuples(index=False, name=None)
    )

    print(f"Correlation matrix is PSD:{is_psd(correlation_matrix)}")


    tech_idx = np.array([
        correlation_matrix_index_map[(t[0],)]
        for t in tuples
    ])

    expanded = correlation_matrix[np.ix_(tech_idx, tech_idx)]

    correlation_matrix_expanded = pd.DataFrame(
        expanded,
        index=tuples,
        columns=tuples,
    )

    print(f"Expanded correlation matrix is PSD:{is_psd(expanded)}")


    covariance_index = list(
        absolute_sd_per_tech_aggregated[
            [
                "set_technologies",
                "set_location",
                "set_time_steps_yearly",
                "set_capacity_types",
            ]
        ].itertuples(index=False, name=None)
    )

    covariance_index_map = {
        idx: i
        for i, idx in enumerate(covariance_index)
    }
    sd = absolute_sd_per_tech_aggregated["value"].to_numpy()

    R = sp.coo_matrix(correlation_matrix_expanded.values)


    Sigma = sp.coo_matrix(
        (
            R.data * sd[R.row] * sd[R.col],
            (R.row, R.col),
        ),
        shape=R.shape,
    ).tocsr()

    Sigma = sp.triu(Sigma, format="csr")


    print(f"Covariance matrix is PSD:{is_psd(Sigma.toarray())}")


    return covariance_index_map, Sigma

def _regularize_correlation_matrix(correlation_matrix, term):
    corr = correlation_matrix.toarray()
    lam = term
    correlation_matrix = (1 - lam) * corr + lam * np.eye(corr.shape[0])
    return correlation_matrix

def generate_covariance_matrix(optimization_setup):

    # Get absolute SD
    absolute_sd_per_tech = _calculate_absolute_sd(optimization_setup)
    absolute_sd_per_tech = absolute_sd_per_tech.swaplevel(1, 2)

    aggregation = _determine_allowed_aggregation(absolute_sd_per_tech.reset_index(name="value"))
    absolute_sd_per_tech_aggregated = _aggregate_by_structure(absolute_sd_per_tech.reset_index(name="value"),
                                                             aggregation)

    # Get correlation matrix
    correlation_matrix_index_map, correlation_matrix = _get_correlation_matrix(optimization_setup)
    correlation_matrix = _regularize_correlation_matrix (correlation_matrix, term = 10e-6)

    # Get covariance matrix
    covariance_index_map, covariance_matrix = _get_covariance_matrix(absolute_sd_per_tech_aggregated, correlation_matrix, correlation_matrix_index_map)

    return covariance_index_map, covariance_matrix

def get_non_zero_elements(covariance_matrix, covariance_matrix_indexmap):
    coo = covariance_matrix.tocoo()
    inv_index_map = {v: k for k, v in covariance_matrix_indexmap.items()}
    nonzero_entries = [
        (inv_index_map[i], inv_index_map[j])
        for i, j in zip(coo.row, coo.col)
    ]
    return nonzero_entries


def generate_sum_list(technology, location, time_step_year, capacity_type):
    sum_list = []
    selection_dict = {"set_technologies":technology}
    if location != "aggregated":
        selection_dict["set_location"] = location
    else:
        sum_list.append("set_location")

    if time_step_year != "aggregated":
        selection_dict["set_time_steps_yearly"] = time_step_year
    else:
        sum_list.append("set_time_steps_yearly")

    if capacity_type != "aggregated":
        selection_dict["set_capacity_types"] = capacity_type
    else:
        sum_list.append("set_capacity_types")
    return selection_dict, sum_list