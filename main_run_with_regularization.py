from datetime import datetime
import json
from pathlib import Path
import os
import pandas as pd
from scipy import sparse
import argparse

from zen_garden.plugins.mean_variance_optimization.plugin import config, construct_mean_variance_objective
from preprocessing.helpers import ModelApi, construct_model, generate_samples

# SETTINGS
run_on = "local"  # epse_server, euler, local
nr_timesteps = 1
example_dataset = True
with_diagonal_variance_only = False
method = "regularization_only"


def get_parameter_grid():
    base = 1e-10
    weights = [x * base for x in [1e1, 1e2, 1e3]]
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


    # Main run with weighting factor
    dir_extension = f"CrystalBall_{nr_timesteps}periods_snapshot_regularization"
    results_root = f"./outputs_{time_str}_{dir_extension}_weighting_factor"
    if not os.path.exists(results_root):
        os.makedirs(results_root)
    #
    include_variances_for = "technology_capex"
    result_folder = f"{results_root}/lambda_{str(weight)}"
    config["method"] = method
    config["regularization_factor"] = weight
    m_api = construct_model(weight, task_id, dataset, result_folder, include_variances_for)
    m_api.solve_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, required=True)
    args = parser.parse_args()

    main(args.task_id)
