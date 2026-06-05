import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import os

from zen_garden.plugin_system.events import Event, EventPublisher
from zen_garden.plugins.mean_variance_optimization.helpers import calculate_absolute_sd, calculate_correlation_matrix, \
    generate_covariance_pairs

from zen_garden.model.element import Element

config = {
    "weighting_factor": None,
    "include_variances_for": ["technology_capex", "technology_opex", "import", "export", "demand_shedding"],
}


def _only_technology_correlation(optimization_setup, quadratic_term):
    """Simplified variance term: aggregate capacity additions over locations and time steps,
    and compute correlations only per technology pair (not per location/time).

    Introduces an auxiliary variable ``capacity_addition_tech_agg`` (one per technology /
    capacity-type pair) that equals the sum of ``capacity_addition`` over all locations and
    yearly time steps, and constrains it accordingly.  The quadratic variance term is then
    built from products of these scalar variables, which linopy can handle as a proper QP.
    """
    model = optimization_setup.model

    if "capacity_addition_tech_agg" not in model.variables:
        variables = optimization_setup.variables

        variables.add_variable(
            model,
            name="capacity_addition_tech_agg",
            index_sets=Element.create_custom_set(
                [
                    "set_technologies",
                    "set_capacity_types"
                ],
                optimization_setup,
            ),
            bounds=(0, np.inf),
            doc="size of installed technology at location l and time t",
            unit_category={"energy_quantity": 1, "time": -1},
        )


    # Create constraint aggregating technology capacities over locations and investment periods
    if "constraint_capacity_addition_tech_agg" not in model.constraints:
        lhs_exp = model.variables["capacity_addition_tech_agg"]
        rhs_exp = model.variables["capacity_addition"].sum(["set_location", "set_time_steps_yearly"])
        constraint_capacity_addition = lhs_exp == rhs_exp


        optimization_setup.constraints.add_constraint(
            "constraint_capacity_addition_tech_agg", constraint_capacity_addition
        )


    # Absolute SD per technology
    absolute_sd_per_tech = calculate_absolute_sd(optimization_setup)

    # Correlation per technology pair
    corr_df = calculate_correlation_matrix(optimization_setup)

    # Build (tech_i, cap_i) × (tech_j, cap_j) pairs with correlation
    pairs = generate_covariance_pairs(absolute_sd_per_tech, corr_df)
    #
    # Quadratic term using the auxiliary variable
    covariance_rows = []
    capacity_addition_tech_agg = model.variables["capacity_addition_tech_agg"]
    for _, row in tqdm(pairs.iterrows(), total=len(pairs),
                       desc="Constructing quadratic variance term (technology-only correlation)"):
        tech_i, cap_i = row["tech_i"], row["cap_i"]
        tech_j, cap_j = row["tech_j"], row["cap_j"]
        correlation = row["correlation"]

        sigma_i = absolute_sd_per_tech[(tech_i, cap_i)]
        sigma_j = absolute_sd_per_tech[(tech_j, cap_j)]

        C_i = capacity_addition_tech_agg.sel(set_technologies=tech_i, set_capacity_types=cap_i)
        C_j = capacity_addition_tech_agg.sel(set_technologies=tech_j, set_capacity_types=cap_j)

        scalar_coeff = correlation * sigma_i * sigma_j
        quadratic_term += scalar_coeff * C_i * C_j

    #     covariance_rows.append({
    #         "tech_i": tech_i,
    #         "cap_i": cap_i,
    #         "tech_j": tech_j,
    #         "cap_j": cap_j,
    #         "covariance": correlation * sigma_i * sigma_j,
    #     })

    # dir = Path(optimization_setup.analysis.folder_output).joinpath(os.path.basename(optimization_setup.analysis.dataset))
    # pd.DataFrame(covariance_rows).to_csv(
    #     dir / "covariance_pairs_objective_construction.csv", index=False
    # )

    return quadratic_term


@EventPublisher.register(Event.after_postprocessing)
def calculate_variance_from_solution(postprocessing=None):
    """Compute the realized capex variance from the solved capacity_addition values.

    Variance = Σ_{i,j} ρ_{ij} · σ_i · σ_j · C_i · C_j

    where C_k = Σ_{loc, t} capacity_addition[tech_k, cap_k, loc, t]  (from solution).

    The scalars are injected into ``model._solution`` so that ``save_var`` exports
    them automatically as ``capex_variance`` and ``capex_sd``.
    """
    # Absolute SD per technology
    absolute_sd_per_tech = calculate_absolute_sd(postprocessing.optimization_setup)

    # Correlation per technology pair
    corr_df = calculate_correlation_matrix(postprocessing.optimization_setup)

    # Build upper-triangle (tech_i, cap_i) × (tech_j, cap_j) pairs with correlation
    pairs = generate_covariance_pairs(absolute_sd_per_tech, corr_df)

    # Solved capacity additions → aggregate over locations and time steps
    capacity_addition_sol = (
        postprocessing.optimization_setup.model.variables["capacity_addition"]
        .solution
        .sum(["set_location", "set_time_steps_yearly"])
    )

    variance = 0.0
    covariance_rows = []
    for _, row in pairs.iterrows():
        tech_i, cap_i = row["tech_i"], row["cap_i"]
        tech_j, cap_j = row["tech_j"], row["cap_j"]
        correlation = row["correlation"]

        sigma_i = absolute_sd_per_tech[(tech_i, cap_i)]
        sigma_j = absolute_sd_per_tech[(tech_j, cap_j)]

        C_i = float(capacity_addition_sol.sel(set_technologies=tech_i, set_capacity_types=cap_i))
        C_j = float(capacity_addition_sol.sel(set_technologies=tech_j, set_capacity_types=cap_j))

        variance += correlation * sigma_i * sigma_j * C_i * C_j

        covariance_rows.append({
            "tech_i": row["tech_i"],
            "cap_i": row["cap_i"],
            "tech_j": row["tech_j"],
            "cap_j": row["cap_j"],
            "covariance": row["correlation"] * sigma_i * sigma_j,
        })

    plugin_reporting = {}
    plugin_reporting["variance"]= variance
    plugin_reporting["standard_deviation"]= variance ** 0.5


    model = postprocessing.optimization_setup.model
    objective_value = float(model.objective.value)
    npv_value = float(model.variables["net_present_cost"].solution.sum("set_time_steps_yearly"))

    plugin_reporting["objective_value"] = objective_value
    plugin_reporting["npv"] = npv_value
    plugin_reporting["weighting_factor"] = config.get("weighting_factor")

    postprocessing.write_file(postprocessing.name_dir.joinpath("mean_variance_dict"), plugin_reporting, mode="w", format="json")
    pd.DataFrame(covariance_rows).to_csv(
        postprocessing.name_dir / "covariance_pairs_reporting.csv", index=False
    )



@EventPublisher.register(Event.after_model_construction)
def construct_mean_variance_objective(optimization_setup=None):


    quadratic_term = 0

    weighting_factor = config.get("weighting_factor")
    if weighting_factor:
        if "technology_capex" in config.get("include_variances_for"):
            quadratic_term = _only_technology_correlation(optimization_setup, quadratic_term)


    optimization_setup.model.remove_objective()

    npv_term = optimization_setup.model.variables["net_present_cost"].sum("set_time_steps_yearly")

    weighting_factor = config.get("weighting_factor")
    if weighting_factor is None: weighting_factor = 0

    objective = weighting_factor * quadratic_term + npv_term
    sense = "min"
    optimization_setup.model.add_objective(objective, sense=sense)




