from datetime import datetime
import json
from pathlib import Path
import os
import pandas as pd
from scipy import sparse
import argparse

from zen_garden.plugins.mean_variance_optimization.plugin import config, construct_mean_variance_objective
from preprocessing.helpers import ModelApi, construct_model, generate_samples

def get_parameter_grid():
    base = 10e-6
    weights = [x * base for x in [0, 25, 50, 75, 100]]
    # weights = [x * base for x in [0, 25]]
    return {
        "weights": weights
    }


def main(task_id: int):
    params = get_parameter_grid()

    weights = params["weights"]

    if task_id < 0 or task_id >= len(weights):
        raise ValueError(f"Invalid task_id {task_id}")

    weight = weights[task_id]

    sample_path = Path("./sample.pkl")

    now = datetime.now()
    time_str = now.strftime("%Y%m%d-%H%M%S")


    print(f"Running task_id={task_id}, weight={weight}")

    sample = pd.read_pickle(sample_path)

    # SETTINGS
    run_on = "euler" #epse_server, euler, local
    example_dataset = False

    # PATHS
    # load_settings
    with open('run_settings.json') as json_file:
        settings = json.load(json_file)

    root_path = Path(settings[run_on]['root_path'])
    if example_dataset:
        dataset = "8_yearly_variation"
        os.chdir(Path(f"C:\ZenGardenInput\example_datasets"))
    else:
        dataset = "Crystal_Ball"
        os.chdir(root_path / "data")

    # generate result folder

    # generate sample
    # m_api = ModelApi(config="./config.json", dataset=dataset, folder_output=results_root + "/sampling")
    # m_api.build_model()
    #
    # n_samples = 1000
    # covariance_map, covariance_matrix_upper = generate_covariance_matrix(m_api.optimization_setup)
    # sparse.save_npz(f"{results_root}/covariance_matrix_upper.npz", covariance_matrix_upper)
    # mapping_serializable = {
    #     "|".join(k): v for k, v in covariance_map.items()
    # }
    # with open(f"{results_root}/covariance_map.json", "w") as f:
    #     json.dump(mapping_serializable, f)
    # sample = generate_samples(covariance_matrix_upper, covariance_map, n_samples=n_samples)
    # sample.to_pickle(f"./outputs_{time_str}_{dir_extension}/sample.pkl")
    # sample.to_csv(f"./outputs_{time_str}_{dir_extension}/sample.csv", index=False)


    # Main run with weighting factor
    dir_extension = "fixed_cross_terms_1periods_with_operation"
    results_root = f"./outputs_{time_str}_{dir_extension}_weighting_factor"
    if not os.path.exists(results_root):
        os.makedirs(results_root)
    #
    include_variances_for = "technology_capex"
    result_folder = f"{results_root}/lambda_{str(weight)}"
    m_api = construct_model(weight, task_id, dataset, result_folder, include_variances_for)
    m_api.solve_model()
    objective_value = float(m_api.optimization_setup.model.variables["net_present_cost"].solution.sum("set_time_steps_yearly"))
    # m_api.reevaluate_objective(result_folder, sample, include_variances_for)

    # Main run with cost limit
    dir_extension = "fixed_cross_terms_1periods_with_operation"
    results_root = f"./outputs_{time_str}_{dir_extension}_cost_limit"
    if not os.path.exists(results_root):
        os.makedirs(results_root)
    result_folder = f"{results_root}/lambda_{str(weight)}"


    config["method"] = "cost_constraint"
    config["cost_constraint"] = objective_value
    m_api = construct_model(weight, task_id, dataset, result_folder, include_variances_for)
    m_api.solve_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, required=True)
    args = parser.parse_args()

    main(args.task_id)
