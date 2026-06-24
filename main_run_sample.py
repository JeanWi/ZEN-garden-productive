from datetime import datetime
import json
from pathlib import Path
import os
import pandas as pd
from scipy import sparse
import argparse

from zen_garden.plugins.mean_variance_optimization.plugin import config, construct_mean_variance_objective
from zen_garden.plugins.mean_variance_optimization.helpers import CovarianceImports, CovarianceTechnologies
from preprocessing.helpers import ModelApi, construct_model, generate_samples

# SETTINGS
example_dataset = False

def main(nr_timesteps: int, run_on: str):

    now = datetime.now()
    time_str = now.strftime("%Y%m%d-%H%M%S")

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
    dir_extension = f"CrystalBall_{nr_timesteps}periods_snapshot_full_covariance_matrix"
    results_root = f"./outputs_{time_str}_{dir_extension}_weighting_factor"
    if not os.path.exists(results_root):
        os.makedirs(results_root)

    sample = {}
    m_api = ModelApi(config="./config.json", dataset=dataset, folder_output=results_root + "/sampling")
    m_api.build_model()

    n_samples = 1000

    for item in ["technology_capex", "imports"]:
        if item == "imports":
            covariance_calculation = CovarianceImports(m_api.optimization_setup)
        elif item == "technology_capex":
            covariance_calculation = CovarianceTechnologies(m_api.optimization_setup)

        covariance_map, covariance_matrix_upper = covariance_calculation.generate_covariance_matrix()
        sparse.save_npz(f"{results_root}/covariance_matrix_{item}_upper.npz", covariance_matrix_upper)
        mapping_serializable = {
            "|".join(map(str, k)): v
            for k, v in covariance_map.items()
        }
        with open(f"{results_root}/covariance_map_{item}.json", "w") as f:
            json.dump(mapping_serializable, f)
        sample[item] = generate_samples(covariance_matrix_upper, covariance_map, n_samples=n_samples)

    sample = pd.concat(sample, names=["VarianceType"], axis=1)
    sample.to_csv(Path(f"./sample_T{str(nr_timesteps)}.csv"), index=False)
    sample.to_pickle(Path(f"./sample_T{str(nr_timesteps)}.pkl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nr_time_steps", type=int, default=1)
    parser.add_argument("--run_on", type=str, default="local", choices=["local", "epse_server", "euler"])
    args = parser.parse_args()

    main(args.nr_time_steps, args.run_on)
