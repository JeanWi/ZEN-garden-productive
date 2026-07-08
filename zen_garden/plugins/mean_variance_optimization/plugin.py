import pandas as pd
from tqdm import tqdm

from zen_garden.plugin_system.events import Event, EventPublisher
from zen_garden.plugins.mean_variance_optimization.helpers import CovarianceTechnologies, CovarianceImports, \
    get_non_zero_elements

config = {
    "method": "weighting_factor", # cost_constraint, weighting_factor, regularization_only
    "weighting_factor": None,
    "cost_constraint": None,
    "include_correlation": True,
    "regularization_factor": 1e-8,
    "include_variances_for": ["technology_capex", "import"],
}

def _technology_correlation(optimization_setup, quadratic_term):
    """Simplified variance term: aggregate capacity additions over locations and time steps,
    and compute correlations only per technology pair (not per location/time).

    Introduces an auxiliary variable ``capacity_addition_tech_agg`` (one per technology /
    capacity-type pair) that equals the sum of ``capacity_addition`` over all locations and
    yearly time steps, and constrains it accordingly.  The quadratic variance term is then
    built from products of these scalar variables, which linopy can handle as a proper QP.
    """
    model = optimization_setup.model

    # Get covariance matrix
    covariance_calculation = CovarianceTechnologies(optimization_setup)
    covariance_matrix_indexmap, covariance_matrix  = covariance_calculation.generate_covariance_matrix()

    # Get non-zero entries
    covariance_pairs = get_non_zero_elements(covariance_matrix, covariance_matrix_indexmap)

    # Construct variables (only have integer indexing)
    agg_index = pd.Index(
        range(len(covariance_matrix_indexmap)),
        name="agg_index"
    )
    inverse_index_map = {
        v: k
        for k, v in covariance_matrix_indexmap.items()
    }

    optimization_setup.model.add_variables(
        name="capacity_addition_tech_agg",
        coords=[agg_index],
    )

    include_correlation = config.get("include_correlation")

    # Construct constraints for aggregate variables
    for variable_index, variable_key in tqdm(inverse_index_map.items(), total=len(inverse_index_map),
                       desc="Constructing aggregate variables (technology correlation"):


        technology = variable_key[0]
        location = variable_key[1]
        time_step_year = variable_key[2]
        capacity_type = variable_key[3]

        selection_dict, sum_list = covariance_calculation.generate_sum_list(technology, location, time_step_year, capacity_type)

        lhs_exp = model.variables["capacity_addition_tech_agg"].sel(agg_index=variable_index)
        rhs_exp = model.variables["capacity_addition"].sel(selection_dict).sum(sum_list)
        constraint_capacity_addition = lhs_exp == rhs_exp

        optimization_setup.constraints.add_constraint(
            f"constraint_capacity_addition_tech_agg{variable_index}", constraint_capacity_addition
        )

    # Construct quadratic term
    for pair in tqdm(covariance_pairs, total=len(covariance_pairs),
                       desc="Constructing quadratic variance term (technology correlation)"):
        index_i = covariance_matrix_indexmap[pair[0]]
        index_j = covariance_matrix_indexmap[pair[1]]

        C_i = model.variables["capacity_addition_tech_agg"].sel(agg_index=index_i)
        C_j = model.variables["capacity_addition_tech_agg"].sel(agg_index=index_j)

        covariance = covariance_matrix[index_i, index_j]

        if index_i != index_j:
            factor = 2
            if include_correlation:
                quadratic_term += factor * covariance * C_i * C_j
        else:
            factor = 1
            quadratic_term += factor * covariance * C_i * C_j

    return quadratic_term

def _import_correlation(optimization_setup, quadratic_term):

    model = optimization_setup.model

    covariance_calculation = CovarianceImports(optimization_setup)
    covariance_matrix_indexmap, covariance_matrix = covariance_calculation.generate_covariance_matrix()

    covariance_pairs = get_non_zero_elements(covariance_matrix, covariance_matrix_indexmap)

    agg_index = pd.Index(
            range(len(covariance_matrix_indexmap)),
            name="agg_index"
        )
    inverse_index_map = {
        v: k
        for k, v in covariance_matrix_indexmap.items()
    }

    optimization_setup.model.add_variables(
        name="imports_agg",
        coords=[agg_index],
    )

    include_correlation = config.get("include_correlation")

    # Construct constraints for aggregate variables
    for variable_index, variable_key in tqdm(inverse_index_map.items(), total=len(inverse_index_map),
                       desc="Constructing aggregate variables (import cost correlation)"):

        carrier = variable_key[0]
        location = variable_key[1]
        time_step_operation = variable_key[2]

        selection_dict, sum_list = covariance_calculation.generate_sum_list(carrier, location, time_step_operation)

        lhs_exp = model.variables["imports_agg"].sel(agg_index=variable_index)
        rhs_exp = model.variables["flow_import"].sel(selection_dict).sum(sum_list)
        constraint_capacity_addition = lhs_exp == rhs_exp

        optimization_setup.constraints.add_constraint(
            f"constraint_imports_agg{variable_index}", constraint_capacity_addition
        )

    # Construct quadratic term
    for pair in tqdm(covariance_pairs, total=len(covariance_pairs),
                       desc="Constructing quadratic variance term (import cost correlation)"):
        index_i = covariance_matrix_indexmap[pair[0]]
        index_j = covariance_matrix_indexmap[pair[1]]

        C_i = model.variables["imports_agg"].sel(agg_index=index_i)
        C_j = model.variables["imports_agg"].sel(agg_index=index_j)

        covariance = covariance_matrix[index_i, index_j]

        if index_i != index_j:
            factor = 2
            if include_correlation:
                quadratic_term += factor * covariance * C_i * C_j
        else:
            factor = 1
            quadratic_term += factor * covariance * C_i * C_j

    return quadratic_term



def _create_diagonal_squared_terms(optimization_setup):
    regularization_term = 0
    regularization_factor = config.get("regularization_factor")
    for var in optimization_setup.model.variables:
        regularization_term += regularization_factor * (optimization_setup.model.variables[var] * optimization_setup.model.variables[var]).sum()
    return regularization_term

def _postprocess_technology_variance(postprocessing, variance_cumsum):
    optimization_setup = postprocessing.optimization_setup
    covariance_calculation = CovarianceTechnologies(optimization_setup)
    covariance_matrix_indexmap, covariance_matrix  = covariance_calculation.generate_covariance_matrix()
    covariance_pairs = get_non_zero_elements(covariance_matrix, covariance_matrix_indexmap)
    covariance_rows = []
    for pair in tqdm(covariance_pairs, total=len(covariance_pairs),
                       desc="Postprocessing variance (technology correlation)"):

        technology_i = pair[0][0]
        technology_j = pair[1][0]

        location_i = pair[0][1]
        location_j = pair[1][1]

        time_step_year_i = pair[0][2]
        time_step_year_j = pair[1][2]

        capacity_type_i = pair[0][3]
        capacity_type_j = pair[1][3]

        index_i = covariance_matrix_indexmap[pair[0]]
        index_j = covariance_matrix_indexmap[pair[1]]
        covariance = covariance_matrix[index_i, index_j]

        selection_dict_i, sum_list_i = covariance_calculation.generate_sum_list(technology_i, location_i, time_step_year_i, capacity_type_i)
        selection_dict_j, sum_list_j = covariance_calculation. generate_sum_list(technology_j, location_j, time_step_year_j, capacity_type_j)

        C_i = float(optimization_setup.model.variables["capacity_addition"].solution.sel(selection_dict_i).sum(sum_list_i))
        C_j = float(optimization_setup.model.variables["capacity_addition"].solution.sel(selection_dict_j).sum(sum_list_j))


        if index_i != index_j:
            factor = 2
        else:
            factor = 1
        variance_cumsum += factor * covariance * C_i * C_j

        covariance_rows.append({
            "pair": pair,
            "covariance": covariance * C_i * C_j,
            "covariance_factor": covariance
        })

    pd.DataFrame(covariance_rows).to_csv(
        postprocessing.name_dir / "covariance_pairs_reporting.csv", index=False
    )

    return variance_cumsum

def _postprocess_imports_variance(postprocessing, variance_cumsum):
    optimization_setup = postprocessing.optimization_setup


    covariance_calculation = CovarianceImports(optimization_setup)
    covariance_matrix_indexmap, covariance_matrix = covariance_calculation.generate_covariance_matrix()

    covariance_pairs = get_non_zero_elements(covariance_matrix, covariance_matrix_indexmap)

    # Construct quadratic term
    for pair in tqdm(covariance_pairs, total=len(covariance_pairs),
                       desc="Constructing quadratic variance term (import cost correlation)"):
        index_i = covariance_matrix_indexmap[pair[0]]
        index_j = covariance_matrix_indexmap[pair[1]]

        carrier_i = pair[0][0]
        carrier_j = pair[1][0]

        location_i = pair[0][1]
        location_j = pair[1][1]

        time_step_operation_i = pair[0][2]
        time_step_operation_j = pair[1][2]

        covariance = covariance_matrix[index_i, index_j]


        selection_dict_i, sum_list_i = covariance_calculation.generate_sum_list(carrier_i, location_i, time_step_operation_i)
        selection_dict_j, sum_list_j = covariance_calculation.generate_sum_list(carrier_j, location_j, time_step_operation_j)
        C_i = float(optimization_setup.model.variables["flow_import"].solution.sel(selection_dict_i).sum(sum_list_i))
        C_j = float(optimization_setup.model.variables["flow_import"].solution.sel(selection_dict_j).sum(sum_list_j))

        if index_i != index_j:
            factor = 2
            variance_cumsum += factor * covariance * C_i * C_j
        else:
            factor = 1
            variance_cumsum += factor * covariance * C_i * C_j


    return variance_cumsum


@EventPublisher.register(Event.after_postprocessing)
def calculate_variance_from_solution(postprocessing=None):
    """Compute the realized capex variance from the solved capacity_addition values.

    Variance = Σ_{i,j} ρ_{ij} · σ_i · σ_j · C_i · C_j

    where C_k = Σ_{loc, t} capacity_addition[tech_k, cap_k, loc, t]  (from solution).

    The scalars are injected into ``model._solution`` so that ``save_var`` exports
    them automatically as ``capex_variance`` and ``capex_sd``.
    """

    # Get covariance matrix
    variance_cumsum = 0.0

    if "technology_capex" in config.get("include_variances_for"):
        variance_cumsum = _postprocess_technology_variance(postprocessing, variance_cumsum)
    if "imports" in config.get("include_variances_for"):
        variance_cumsum = _postprocess_imports_variance(postprocessing, variance_cumsum)

    plugin_reporting = {}
    plugin_reporting["variance"]= variance_cumsum
    plugin_reporting["standard_deviation"]= variance_cumsum ** 0.5


    model = postprocessing.optimization_setup.model
    objective_value = float(model.objective.value)
    npv_value = float(model.variables["net_present_cost"].solution.sum("set_time_steps_yearly"))

    plugin_reporting["objective_value"] = objective_value
    plugin_reporting["npv"] = npv_value
    plugin_reporting["weighting_factor"] = config.get("weighting_factor")

    postprocessing.write_file(postprocessing.name_dir.joinpath("mean_variance_dict"), plugin_reporting, mode="w", format="json")


@EventPublisher.register(Event.after_model_construction)
def construct_mean_variance_objective(optimization_setup=None):
    variance_term = 0


    optimization_setup.model.remove_objective()
    npv_term = optimization_setup.model.variables["net_present_cost"].sum("set_time_steps_yearly")
    method = config.get("method")

    if method == "weighting_factor":
        weighting_factor = config.get("weighting_factor")
        if weighting_factor is None: weighting_factor = 0

        if weighting_factor != 0:
            if "technology_capex" in config.get("include_variances_for"):
                variance_term = _technology_correlation(optimization_setup, variance_term)
            if "imports" in config.get("include_variances_for"):
                variance_term = _import_correlation(optimization_setup, variance_term)
            objective = weighting_factor * variance_term + npv_term
        else:
            objective = npv_term

        sense = "min"
        optimization_setup.model.add_objective(objective, sense=sense)

    # elif method == "cost_constraint":
    #     # objective is variance
    #     objective = variance_term
    #     sense = "min"
    #     optimization_setup.model.add_objective(objective, sense=sense)
    #
    #     # Limit cost
    #     constraint_cost_objective = npv_term <= config.get("cost_constraint")
    #     optimization_setup.constraints.add_constraint(
    #         f"constraint_cost_variance_plugin", constraint_cost_objective
    #     )

    elif method == "regularization_only":
        regularization_term = _create_diagonal_squared_terms(optimization_setup)
        objective = regularization_term + npv_term
        sense = "min"
        optimization_setup.model.add_objective(objective, sense=sense)

