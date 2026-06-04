import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from zen_garden import run
from zen_garden.plugin_system.events import Event, EventPublisher
from zen_garden.postprocess.postprocess import Postprocess
from zen_garden.wrapper.utils import load_results

# SETTINGS
on_server = False
example_dataset = True


# PATHS
if on_server:
    root_path = Path("/home/jwiegner/ZEN-models")
    os.chdir(root_path / "data")
    dataset = "Crystal_Ball"
    result_folder = "C:\ZenGardenInput\ZEN-models\data\outputs\linear"
    os.chdir(root_path / "data")

else:
    if example_dataset:
        root_path = Path("C:/ZenGardenInput/example_datasets")
        dataset = "1_base_case"
        now = datetime.now()
        time_str = now.strftime("%Y%m%d-%H%M%S")
        result_folder = f"./outputs_{time_str}_design"
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        os.chdir(root_path)

    else:
        root_path = Path("C:/ZenGardenInput/ZEN-models")
        os.chdir(root_path / "data")
        dataset = "Crystal_Ball"
        result_folder = "C:\ZenGardenInput\ZEN-models\data\outputs\linear"
        os.chdir(root_path / "data")

with open("./config.json") as f:
    config = json.load(f)
config["solver"]["solver_options"]["LogFile"] = f"{result_folder}/solver_design.log"

with open("./config.json", "w") as f:
    json.dump(config, f, indent=4)

optimization_setup = run(
    config="./config.json", dataset=dataset, folder_output=result_folder
)

# fix capacity expansions
results = load_results(result_folder + "/" + dataset, None)
capacity_addition = optimization_setup.model.variables["capacity_addition"]

for (tech, loc, year), row in results["capacity_addition"].iterrows():
    for cap_type in ["energy", "power"]:

        value = row[cap_type]

        if pd.notna(value):

            sel = dict(
                set_technologies=tech,
                set_capacity_types=cap_type,
                set_location=loc,
                set_time_steps_yearly=0,  # adapt if needed
            )

            capacity_addition.lower.loc[sel] = value
            capacity_addition.upper.loc[sel] = value


# adapt capex parameters

# rerun
result_folder = f"./outputs_{time_str}_operation"

if not os.path.exists(result_folder):
    os.makedirs(result_folder)

optimization_setup.solver.solver_options["LogFile"] = (
    f"{result_folder}/solver_operation.log"
)
optimization_setup.solver.solver_dir = result_folder
optimization_setup.analysis.folder_output = result_folder

EventPublisher.trigger(
    Event.after_model_construction, optimization_setup=optimization_setup
)

if optimization_setup.solver.use_scaling:
    optimization_setup.scaling.run_scaling()
elif (
    optimization_setup.solver.analyze_numerics
    or optimization_setup.solver.run_diagnostics
):
    optimization_setup.scaling.analyze_numerics()
# SOLVE THE OPTIMIZATION PROBLEM
optimization_setup.solve()

if optimization_setup.solver.use_scaling:
    optimization_setup.scaling.re_scale()

# EVALUATE RESULTS
scenarios = {"": {}}
subfolder = Path(".")
model_name = dataset
scenario_name = None
param_map = None
# write results
Postprocess(
    optimization_setup,
    scenarios=scenarios,
    subfolder=subfolder,
    model_name=model_name,
    scenario_name=scenario_name,
    param_map=param_map,
)
