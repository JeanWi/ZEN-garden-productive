from datetime import datetime
from pathlib import Path
import json
import os

from preprocessing.helpers import get_all_runs, ModelApi, generate_samples
from zen_garden.plugins.mean_variance_optimization.helpers import *


# SETTINGS
run_on = "local" #epse_server, euler, local
example_dataset = False
base = 10e-6
# weights = [x * base for x in [25, 50, 75, 100]]
weights = [x * base for x in [25]]
dir_extension = "fixed_cross_terms_1periods_with_operation"
n_samples = 10

# PATHS
# load_settings
with open('run_settings.json') as json_file:
    settings = json.load(json_file)

root_path = Path(settings[run_on]['root_path'])
if example_dataset:
    dataset = "8_yearly_variation"
    os.chdir(root_path)
else:
    dataset = "Crystal_Ball"
    os.chdir(root_path / "data")


now = datetime.now()
time_str = now.strftime("%Y%m%d-%H%M%S")
result_folder = f"./outputs_{time_str}_{dir_extension}/linear"
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

with open("./config.json") as f:
    config = json.load(f)
config["solver"]["solver_options"]["LogFile"] = f"{result_folder}/solver.log"

with open("./config.json", "w") as f:
    json.dump(config, f, indent=4)

objective = pd.DataFrame()
# Model API
m_api = ModelApi(config="./config.json", dataset=dataset, folder_output=result_folder)
m_api.build_model()

# Absolute SD per technology
covariance_map, covariance_matrix_upper = generate_covariance_matrix(m_api.optimization_setup)
sample = generate_samples(covariance_matrix_upper, covariance_map, n_samples=n_samples)
sample.to_csv(f"./outputs_{time_str}_{dir_extension}/sample.csv", index=False)

m_api.solve_model()

key = "linear"

# Fix variables
m_api.fix_design_variables()
m_api.fix_operational_variables()
m_api.delete_not_required_constraints()
m_api.optimization_setup.solver.solver_options["Method"] = 0
m_api.solve_model(skip_postprocess = True, skip_scaling=True)

for index, row in sample.iterrows():
    m_api.optimization_setup.solver.solver_options["Method"] = 0
    m_api.optimization_setup.solver.solver_options["NumericFocus"] = 3
    m_api.optimization_setup.solver.solver_options["FeasibilityTol"] = 1e-3
    m_api.reconstruct_cost_constraints(row)

    try:
        m_api.solve_model(skip_postprocess=True, skip_scaling=True)

        total_cost = m_api.optimization_setup.model.objective.value


    except:
        total_cost = -1

    objective.loc[index, key] = total_cost
    pd.DataFrame(objective).to_csv(f"./outputs_{time_str}_{dir_extension}/objective_samples.csv",
                                   index=False)



include_var_for = {
    "capex": ["technology_capex"],
    # "capex_opex": ["capex", "opex"],
    # "capex_opex_import_export": ["capex", "opex", "import", "export"],
    # "all": ["technology_capex", "technology_opex", "import", "export", "demand_shedding"]
}


for variance_inclusion in include_var_for.keys():

    for i, weight in enumerate(weights):
        result_folder = f"./outputs_{time_str}_{dir_extension}/{variance_inclusion}/lambda_{str(weight)}"

        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        with open("./config.json") as f:
            config = json.load(f)

        config["plugins"]["mean_variance_optimization"] = {}
        config["plugins"]["mean_variance_optimization"]["weighting_factor"] = weight
        config["plugins"]["mean_variance_optimization"]["include_variances_for"] = include_var_for[variance_inclusion]
        config["solver"]["solver_options"]["LogFile"] =  f"{result_folder}/solver.log"

        with open("./config_quadratic.json", "w") as f:
            json.dump(config, f, indent=4)

        m_api = ModelApi(config="./config_quadratic.json", dataset=dataset, folder_output=result_folder)
        m_api.build_model()

        m_api.solve_model()

        key = f"lambda_{str(weight)}"

        # Fix variables
        m_api.fix_design_variables()
        m_api.fix_operational_variables()
        m_api.delete_not_required_constraints()

        for index, row in sample.iterrows():
            m_api.optimization_setup.solver.solver_options["Method"] = 0
            m_api.optimization_setup.solver.solver_options["NumericFocus"] = 3
            m_api.optimization_setup.solver.solver_options["FeasibilityTol"] = 1e-3
            m_api.reconstruct_cost_constraints(row)

            try:
                m_api.solve_model(skip_postprocess=True, skip_scaling=True)

                total_cost = m_api.optimization_setup.model.objective.value


            except:
                total_cost = -1

            objective.loc[index, key] = total_cost
            pd.DataFrame(objective).to_csv(f"./outputs_{time_str}_{dir_extension}/objective_samples.csv",
                                           index=False)

        # no_delta = sample.iloc[0] * 0
        # m_api.reconstruct_cost_constraints(no_delta)

        #
        #
        # else:
        #     result_folder = f"./outputs_{time_str}/{variance_inclusion}/lambda_{str(weight)}"
        #
        #     if not os.path.exists(result_folder):
        #         os.makedirs(result_folder)
        #
        #     plugin_config["weighting_factor"] = weight
        #     m_api.optimization_setup.solver.solver_options["LogFile"] =  f"{result_folder}/solver.log"
        #     m_api.optimization_setup.solver.solver_dir =  result_folder
        #     m_api.optimization_setup.analysis.folder_output =  result_folder
        #
        #     EventPublisher.trigger(Event.after_model_construction, optimization_setup=m_api.optimization_setup)
        #
        #     m_api.solve_model()
        #
        #     key = f"lambda_{str(weight)}"
        #
        #     # Fix variables
        #     m_api.fix_design_variables()
        #     m_api.fix_operational_variables()
        #
        #     for index, row in sample.iterrows():
        #         m_api.reconstruct_cost_constraints(row, demand_shedding_allowed=True)
        #
        #         m_api.solve_model(skip_postprocess = True, skip_scaling=True)
        #
        #         total_cost = m_api.optimization_setup.model.objective.value
        #
        #         objective[key].append(total_cost)
        #         objective.loc[index, key] = total_cost
        #         pd.DataFrame(objective).to_csv(f"./outputs_{time_str}_{dir_extension}/objective_samples.csv",
        #                                        index=False)
        #
        #     no_delta = sample.iloc[0] * 0
        #     m_api.reconstruct_cost_constraints(no_delta)


