import os
import json

from zen_garden import run
from datetime import datetime
#
# # Example dataset
# os.chdir("C:/ZenGardenInput/example_datasets")
# # run(config="./config.json", dataset="10_brown_field")
# for weight in [0.01, 0.1, 1]:
#     with open("./config_quadratic.json") as f:
#         config = json.load(f)
#
#     config["plugins"]["mean_variance_optimization"]["weighting_factor"] = weight
#
#     with open("./config_quadratic.json", "w") as f:
#         json.dump(config, f, indent=4)
#
#     run(
#         config="./config_quadratic.json",
#         dataset="10_brown_field",
#         folder_output="./outputs_quadratic" + str(weight),
#     )
# Crystal Ball
# current time
os.chdir("C:/ZenGardenInput/ZEN-models/data")

now = datetime.now()
time_str = now.strftime("%Y%m%d-%H%M%S")
result_folder = f"./outputs_{time_str}/linear"
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

with open("./config.json") as f:
    config = json.load(f)
config["solver"]["solver_options"]["LogFile"] = f"{result_folder}/solver.log"

with open("./config.json", "w") as f:
    json.dump(config, f, indent=4)

run(config="./config.json", dataset="Crystal_Ball", folder_output=result_folder)
# for weight in [1, 0.1, 0.01]:

include_var_for = {
    "capex": ["capex"],
    "capex_opex": ["capex", "opex"],
    "capex_opex_import_export": ["capex", "opex", "import", "export"],
    "all": ["capex", "opex", "import", "export", "demand_shedding"]
}


for variance_inclusion in include_var_for.keys():

    for weight in [1000, 100, 10, 1, 0.1, 0.01]:

        result_folder = f"./outputs_{time_str}/{variance_inclusion}/lambda_{str(weight)}"

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

        run(
            config="./config_quadratic.json",
            dataset="Crystal_Ball",
            folder_output=result_folder,
        )