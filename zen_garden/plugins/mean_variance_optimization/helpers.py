from pathlib import Path
from preprocessing.helpers import ModelApi

import numpy as np
import pandas as pd
import xarray as xr
import scipy.sparse as sp
import itertools
import os


def is_psd(A, tol=1e-8):
    eigvals = np.linalg.eigvalsh(A)
    return eigvals.min() >= -tol

def get_non_zero_elements(covariance_matrix, covariance_matrix_indexmap):
    coo = covariance_matrix.tocoo()
    inv_index_map = {v: k for k, v in covariance_matrix_indexmap.items()}
    nonzero_entries = [
        (inv_index_map[i], inv_index_map[j])
        for i, j in zip(coo.row, coo.col)
    ]
    return nonzero_entries

def _get_sd(optimization_setup, variance_on):
    """
    Reads all sd values from file
    """
    tech_capex_path = Path(optimization_setup.analysis.dataset) / "mean_variance" / variance_on
    sd = pd.read_csv(tech_capex_path / "sd.csv", index_col=0)

    return sd

def _get_correlation_matrix(optimization_setup, variance_on):
    """
    Reads all correlations from file and preprocess them.

    If no_correlation=True, returns identity correlation matrix (diagonal=1, off-diagonal=0).
    """
    tech_capex_path = Path(optimization_setup.analysis.dataset) / "mean_variance" / variance_on

    tec_correlation_df = pd.read_csv(tech_capex_path / "correlation.csv", index_col=0)
    tec_correlation_df = tec_correlation_df.fillna(0)

    R_T = tec_correlation_df.values
    R_T = sp.csr_matrix(R_T)

    index = list(itertools.product(
        tec_correlation_df.index,
    ))
    index_map = {
        key: i for i, key in enumerate(index)
    }

    return (index_map, R_T)

def _regularize_correlation_matrix(correlation_matrix, term):
    corr = correlation_matrix.toarray()
    lam = term
    correlation_matrix = (1 - lam) * corr + lam * np.eye(corr.shape[0])
    return correlation_matrix


class CovarianceCalculation:

    variance_on = None
    aggregation_set = None
    all_sets = []
    other_sets = []

    def __init__(self, optimization_setup):
        self.optimization_setup = optimization_setup

        self.cost = None
        self.absolute_sd = None

        self.correlation_matrix = None
        self.correlation_matrix_index_map = None

        self.covariance_matrix = None
        self.covariance_index_map = None

    def _calculate_absolute_sd(self):
        # Calculate absolute SD per technology
        relative_sd_df = _get_sd(self.optimization_setup, self.variance_on)
        aggregation_items = list(self.optimization_setup.sets[self.aggregation_set])

        relative_sd_xr = xr.DataArray(
            relative_sd_df.loc[aggregation_items, "value"].values,
            dims=(self.aggregation_set,),
            coords={self.aggregation_set: aggregation_items},
        )

        absolute_sd_xr = (self.cost * relative_sd_xr).stack(
            all_dims=self.all_sets
        ).dropna("all_dims")
        absolute_sd = (
            absolute_sd_xr.to_series()
        )
        absolute_sd = absolute_sd[absolute_sd != 0]
        return absolute_sd

    def _get_covariance_matrix(self, absolute_sd_aggregated):
        tuples = list(
            absolute_sd_aggregated[
                self.all_sets
            ].itertuples(index=False, name=None)
        )

        print(f"Correlation matrix is PSD:{is_psd(self.correlation_matrix)}")

        idx = np.array([
            self.correlation_matrix_index_map[(t[0],)]
            for t in tuples
        ])

        expanded = self.correlation_matrix[np.ix_(idx, idx)]

        correlation_matrix_expanded = pd.DataFrame(
            expanded,
            index=tuples,
            columns=tuples,
        )

        print(f"Expanded correlation matrix is PSD:{is_psd(expanded)}")

        covariance_index = list(
            absolute_sd_aggregated[
                self.all_sets
            ].itertuples(index=False, name=None)
        )

        covariance_index_map = {
            idx: i
            for i, idx in enumerate(covariance_index)
        }
        sd = absolute_sd_aggregated["value"].to_numpy()

        R = sp.coo_matrix(correlation_matrix_expanded.values)

        Sigma = sp.coo_matrix(
            (
                R.data * sd[R.row] * sd[R.col],
                (R.row, R.col),
            ),
            shape=R.shape,
        ).tocsr()

        covariance_matrix = sp.triu(Sigma, format="csr")
        print(f"Covariance matrix is PSD:{is_psd(Sigma.toarray())}")

        return covariance_index_map, covariance_matrix

    def _determine_allowed_aggregation(self, df):
        results = {}

        for item, sub in df.groupby(self.aggregation_set):

            results[item] = {}

            aggregation_dimensions = []
            for dim in self.other_sets:
                # check if within-dim values are all identical
                is_constant = sub.groupby(dim)["value"].mean().nunique() == 1
                if is_constant:
                    aggregation_dimensions.append(dim)

            results[item] = aggregation_dimensions

        return results

    def _aggregate_by_structure(self, df, aggregation):
        out = []

        for item, sub in df.groupby(self.aggregation_set):

            var_dims = aggregation[item]
            group_dims = [
                dim for dim in self.other_sets
                if dim not in var_dims
            ]

            agg = sub.groupby(
                [self.aggregation_set] + group_dims
            )["value"].mean().reset_index()


            # replace collapsed dims with "aggregated"
            for dim in self.other_sets:
                if dim not in group_dims:
                    agg[dim] = "aggregated"

            out.append(agg)

        return pd.concat(out, ignore_index=True)

    def _get_costs(self):
        pass



class CovarianceTechnologies(CovarianceCalculation):

    variance_on = "technology_capex"
    aggregation_set = "set_technologies"
    all_sets = ["set_technologies", "set_location", "set_time_steps_yearly", "set_capacity_types"]
    other_sets = ["set_location", "set_time_steps_yearly", "set_capacity_types"]

    def generate_covariance_matrix(self):

        # Get absolute SD
        self.cost = self._get_costs()
        self.absolute_sd = self._calculate_absolute_sd()
        self.absolute_sd = self.absolute_sd[self.absolute_sd != 0]

        aggregation = self._determine_allowed_aggregation(self.absolute_sd.reset_index(name="value"))
        absolute_sd_aggregated = self._aggregate_by_structure(self.absolute_sd.reset_index(name="value"),
                                                                  aggregation)

        # Get correlation matrix
        self.correlation_matrix_index_map, correlation_matrix = _get_correlation_matrix(self.optimization_setup,
                                                                                   self.variance_on)
        self.correlation_matrix = _regularize_correlation_matrix(correlation_matrix, term=10e-6)

        # Get covariance matrix
        self.covariance_index_map, self.covariance_matrix = self._get_covariance_matrix(absolute_sd_aggregated)

        return self.covariance_index_map, self.covariance_matrix

    def _get_costs(self):
        """
        Reads all capex parameters from the optimization setup.
        """
        optimization_setup = self.optimization_setup
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

        return xr.concat(
            [
                capex_specific_conversion,
                capex_specific_storage,
                capex_specific_transport,
            ],
            dim="set_technologies",
            join="outer"
        )

    @classmethod
    def generate_sum_list(cls, technology, location, time_step_year, capacity_type):
        sum_list = []
        selection_dict = {"set_technologies": technology}
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

class CovarianceImports(CovarianceCalculation):

    variance_on = "imports"
    aggregation_set = "set_carriers"
    all_sets = ["set_carriers", "set_nodes", "set_time_steps_operation"]
    other_sets = ["set_nodes", "set_time_steps_operation"]

    def generate_covariance_matrix(self):

        # Get absolute SD
        self.cost = self._get_costs()
        self.absolute_sd = self._calculate_absolute_sd()

        aggregation = self._determine_allowed_aggregation(self.absolute_sd.reset_index(name="value"))
        absolute_sd_aggregated = self._aggregate_by_structure(self.absolute_sd.reset_index(name="value"),
                                                                  aggregation)

        # Get correlation matrix
        self.correlation_matrix_index_map, correlation_matrix = _get_correlation_matrix(self.optimization_setup,
                                                                                   self.variance_on)
        self.correlation_matrix = _regularize_correlation_matrix(correlation_matrix, term=10e-6)

        # Get covariance matrix
        self.covariance_index_map, self.covariance_matrix = self._get_covariance_matrix(absolute_sd_aggregated)

        return self.covariance_index_map, self.covariance_matrix

    def _get_costs(self):

        optimization_setup = self.optimization_setup
        return optimization_setup.parameters.price_import

    @classmethod
    def generate_sum_list(cls, carrier, location, time_step_operation):
        sum_list = []
        selection_dict = {"set_carriers": carrier}
        if location != "aggregated":
            selection_dict["set_nodes"] = location
        else:
            sum_list.append("set_nodes")

        if time_step_operation != "aggregated":
            selection_dict["set_time_steps_operation"] = time_step_operation
        else:
            sum_list.append("set_time_steps_operation")

        return selection_dict, sum_list

