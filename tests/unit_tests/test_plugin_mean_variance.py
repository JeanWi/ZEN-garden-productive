import os
import pytest
import linopy as lp

from zen_garden import run
import zen_garden.default_config as default_config
from zen_garden.plugin_system.loader import register_plugins
from zen_garden.utils import InputDataChecks
from zen_garden.optimization_setup import OptimizationSetup
from zen_garden.plugin_system.events import EventPublisher, Event

def create_config(config_dict):
    root_dir = os.path.dirname(__file__)
    test_dataset_dir = os.path.join(root_dir, "test_input", "test_dataset")

    config = default_config.Config(**config_dict)
    config.analysis.dataset = test_dataset_dir
    return config

def create_input_data_checks_from_config(config):
    input_data_checks = InputDataChecks(config=config, optimization_setup=None)
    input_data_checks.read_system_file(config)
    return input_data_checks

def create_optimization_setup(config):
    config = create_config(config)
    register_plugins(config.plugins)
    EventPublisher.trigger(Event.on_preprocessing, config)
    input_data_checks = create_input_data_checks_from_config(config)
    optimization_setup = OptimizationSetup(config=config, scenario_dict={}, input_data_checks=input_data_checks)
    return optimization_setup

class TestMeanVariancePlugin:
    def test_mean_variance_plugin(self):

        config = {
          "solver": {
            "keep_files": False,
            "use_scaling": False
          },
          "plugins": {
              "mean_variance_optimization": {
                  "weighting_factor": 1,
              }
          }
        }
        optimization_setup = create_optimization_setup(config)

        optimization_setup.construct_optimization_problem()
        EventPublisher.trigger(Event.after_model_construction, optimization_setup=optimization_setup)

        assert isinstance(optimization_setup.model.objective.expression, lp.QuadraticExpression)
