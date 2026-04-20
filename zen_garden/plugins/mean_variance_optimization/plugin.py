import linopy as lp
import pandas as pd
import numpy as np
import json
from pathlib import Path

from zen_garden.plugin_system.events import Event, EventPublisher
from zen_garden.model.element import GenericRule, Element
from zen_garden.preprocess.extract_input_data import DataInput
from zen_garden.preprocess.unit_handling import UnitHandling

# Todo: add types for type checking
config = {
    "weighting_factor": None,
}

def _update_carrier_attributes(variance_attributes, config):
    carrier_path = Path(config.analysis.dataset) / "set_carriers"

    for carrier_dir in carrier_path.iterdir():
        attr_file = carrier_dir / "attributes.json"

        with open(attr_file, "r", encoding="utf-8") as f:
            carrier_data = json.load(f)

        carrier_attributes_to_update = ["variance_price_export", "variance_price_import", "variance_price_shed_demand"]
        updated_carrier_data = _update_attributes(carrier_data, variance_attributes, carrier_attributes_to_update)

        with open(attr_file, "w", encoding="utf-8") as f:
            json.dump(updated_carrier_data, f, indent=2)

def _update_conversion_technology_attributes(variance_attributes, config):

    tech_path = Path(config.analysis.dataset) / "set_technologies" / "set_conversion_technologies"
    for tech_dir in tech_path.iterdir():
        if not "set_retrofitting_technologies" in tech_dir.parts:
            attr_file = tech_dir / "attributes.json"

            with open(attr_file, "r", encoding="utf-8") as f:
                tech_data = json.load(f)

            tech_attributes_to_update = []
            tech_attributes_to_update.append("variance_capex_specific_conversion")

            updated_tech_data = _update_attributes(tech_data, variance_attributes, tech_attributes_to_update)

            with open(attr_file, "w", encoding="utf-8") as f:
                json.dump(updated_tech_data, f, indent=2)

def _update_retrofitting_technology_attributes(variance_attributes, config):

    tech_path = Path(config.analysis.dataset) / "set_technologies" / "set_conversion_technologies" / "set_retrofitting_technologies"
    for tech_dir in tech_path.iterdir():
        attr_file = tech_dir / "attributes.json"

        with open(attr_file, "r", encoding="utf-8") as f:
            tech_data = json.load(f)

        tech_attributes_to_update = []
        tech_attributes_to_update.append("variance_capex_specific_conversion")

        updated_tech_data = _update_attributes(tech_data, variance_attributes, tech_attributes_to_update)

        with open(attr_file, "w", encoding="utf-8") as f:
            json.dump(updated_tech_data, f, indent=2)



def _update_attributes(carrier_data, variance_attributes, field_to_update):

    for key in field_to_update:
        if key not in carrier_data and key in variance_attributes:
            carrier_data[key] = variance_attributes[key]
    return carrier_data

@EventPublisher.register(Event.on_preprocessing)
def add_variance_to_attribute_jsons(config):
    # Add variance fields to attribute.json of carriers
    variance_attribute_path = Path(config.analysis.dataset) / "variances" / "attributes.json"

    with open(variance_attribute_path, "r") as f:
        variance_attributes = json.load(f)

    _update_carrier_attributes(variance_attributes, config)
    _update_conversion_technology_attributes(variance_attributes, config)
    _update_retrofitting_technology_attributes(variance_attributes, config)

@EventPublisher.register(Event.on_carrier_store_input_data)
def add_variance_to_carrier(carrier):
    carrier.raw_time_series["variance_price_export"] = carrier.data_input.extract_input_data(
        "variance_price_export",
        index_sets=["set_nodes", "set_time_steps"],
        time_steps="set_base_time_steps_yearly",
        unit_category={},
    )

    carrier.raw_time_series["variance_price_import"] = carrier.data_input.extract_input_data(
        "variance_price_import",
        index_sets=["set_nodes", "set_time_steps"],
        time_steps="set_base_time_steps_yearly",
        unit_category={},
    )

    carrier.variance_price_shed_demand = carrier.data_input.extract_input_data(
        "variance_price_shed_demand",
        index_sets=[],
        unit_category={},
    )


@EventPublisher.register(Event.on_conversion_technology_store_input_data)
def add_variance_to_conversion_technology(technology):
    technology.variance_capex_specific_conversion = technology.data_input.extract_input_data(
        "variance_capex_specific_conversion",
        index_sets=["set_nodes", "set_time_steps_yearly"],
        time_steps="set_time_steps_yearly",
        unit_category={},
    )

@EventPublisher.register(Event.on_carrier_construct_params)
def add_variance_parameters_to_carrier(optimization_setup, carrier_cls):
    optimization_setup.parameters.add_parameter(
        name="variance_price_export",
        index_names=["set_carriers", "set_nodes", "set_time_steps_operation"],
        doc="Variance of price for export",
        calling_class=carrier_cls,
    )

    optimization_setup.parameters.add_parameter(
        name="variance_price_import",
        index_names=["set_carriers", "set_nodes", "set_time_steps_operation"],
        doc="Variance of price for import",
        calling_class=carrier_cls,
    )

    optimization_setup.parameters.add_parameter(
        name="variance_price_shed_demand",
        index_names=["set_carriers"],
        doc="Variance of price to shed demand",
        calling_class=carrier_cls,
    )

@EventPublisher.register(Event.on_conversion_technology_construct_params)
def add_variance_parameters_to_conversion_technology(optimization_setup, technology_cls):
    optimization_setup.parameters.add_parameter(
            name="variance_capex_specific_conversion",
            index_names=[
                "set_conversion_technologies",
                "set_nodes",
                "set_time_steps_yearly",
            ],
            doc="Variance of specific capex of conversion technologies",
            calling_class=technology_cls,
        )


#
#
# class Variance(Element):
#     """Class defining a generic variance as a parameter."""
#
#     label = "variances"
#
#     def __init__(self, name, optimization_setup):
#         """Initialization of a generic variance object.
#
#         :param carrier: placeholder for variance that is added to the model
#         :param optimization_setup: The OptimizationSetup the element is part of
#         """
#         super().__init__(name, optimization_setup)
#         self.raw_time_series = dict()
#
#     def get_input_path(self):
#         """Get input path where input data is stored input_path."""
#         # get technology type
#         class_label = self.label
#         # get path dictionary
#         paths = self.optimization_setup.paths
#         # get input path for current class_label
#         self.input_path = Path(paths[class_label]["folder"])
#
#     def store_input_data(self):
#         """Retrieves and stores input data for element as attributes. Each Child class
#         overwrites method to store different attributes.
#         """
#         # store scenario dict
#         super().store_scenario_dict()
#
#         self.raw_time_series = dict()
#         self._read_variances()
#
#     @classmethod
#     def _construct_param(cls, optimization_setup, par_name, par_index_names, par_doc, corresponding_parameter):
#         """Constructs a parameter for the optimization model. Each Child class overwrites
#         method to construct different parameters.
#         """
#         parameters = optimization_setup.parameters
#
#         def get_parameter_data(par_name, par_index_names):
#             if len(par_index_names) > 0:
#                 custom_set, index_list = cls.create_custom_set(par_index_names, optimization_setup)
#             else:
#                 index_list = []
#             component_data, dict_of_units, attribute_is_series = (
#                 optimization_setup.get_attribute_of_all_elements(
#                     cls,
#                     par_name,
#                     capacity_types=False,
#                     return_attribute_is_series=True,
#                 )
#             )
#             return component_data["Variance"], index_list
#
#         data, index_list = get_parameter_data(par_name, par_index_names)
#         # data_temp = copy.copy(data)
#         relative_variance_xr = optimization_setup.parameters.convert_to_xarr(data, index_list)
#
#         par_values = (relative_variance_xr * corresponding_parameter).where(
#             ~np.isinf(corresponding_parameter),
#             np.inf
#         )
#         parameters.add_parameter(
#             name=par_name,
#             index_names=par_index_names,
#             doc=par_doc,
#             data=par_values,
#             calling_class=cls,
#         )
#
#
#     @classmethod
#     def construct_params(cls, optimization_setup):
#         """Constructs parameters for the optimization model. Each Child class overwrites
#         method to construct different parameters.
#         """
#
#         par_name = "variance_price_export"
#         par_index_names = ["set_carriers", "set_nodes", "set_time_steps_operation"]
#         par_doc = "Variance of price for export"
#         corresponding_parameter = optimization_setup.parameters.price_export
#         cls._construct_param(optimization_setup, par_name, par_index_names, par_doc, corresponding_parameter)
#
#         par_name = "variance_price_import"
#         par_index_names = ["set_carriers", "set_nodes", "set_time_steps_operation"]
#         par_doc = "Variance of price for import"
#         corresponding_parameter = optimization_setup.parameters.price_import
#         cls._construct_param(optimization_setup, par_name, par_index_names, par_doc, corresponding_parameter)
#
#         par_name = "variance_price_shed_demand"
#         par_index_names = ["set_carriers"]
#         par_doc = "Variance of price to shed demand"
#         corresponding_parameter = optimization_setup.parameters.price_shed_demand
#         cls._construct_param(optimization_setup, par_name, par_index_names, par_doc, corresponding_parameter)
#
#         par_name = "variance_price_carbon_emissions"
#         par_index_names = ["set_time_steps_yearly"]
#         par_doc = "variance of price for carbon emissions"
#         corresponding_parameter = optimization_setup.parameters.price_carbon_emissions
#         cls._construct_param(optimization_setup, par_name, par_index_names, par_doc, corresponding_parameter)
#
#         if len(optimization_setup.sets["set_conversion_technologies"]) > 0:
#             par_name = "variance_capex_specific_conversion"
#             par_index_names = ["set_conversion_technologies", "set_nodes", "set_time_steps_yearly"]
#             par_doc = "variance of capex of conversion technologies"
#             corresponding_parameter = optimization_setup.parameters.capex_specific_conversion
#             cls._construct_param(optimization_setup, par_name, par_index_names, par_doc, corresponding_parameter)
#
#         if len(optimization_setup.sets["set_storage_technologies"]) > 0:
#             par_name = "variance_capex_specific_storage"
#             par_index_names = ["set_storage_technologies", "set_nodes", "set_time_steps_yearly"]
#             par_doc = "variance of capex of storage technologies"
#             corresponding_parameter = optimization_setup.parameters.capex_specific_storage
#             cls._construct_param(optimization_setup, par_name, par_index_names, par_doc, corresponding_parameter)
#
#         if len(optimization_setup.sets["set_transport_technologies"]) > 0:
#             par_name = "variance_capex_specific_transport"
#             par_index_names = ["set_transport_technologies", "set_edges", "set_time_steps_yearly"]
#             par_doc = "variance of capex of transport technologies"
#             corresponding_parameter = optimization_setup.parameters.capex_specific_transport
#             cls._construct_param(optimization_setup, par_name, par_index_names, par_doc, corresponding_parameter)
#
#         # par_name = "variance_opex_specific_variable"
#         # par_index_names = ["set_technologies", "set_location", "set_time_steps_operation"]
#         # par_doc = "variance of capex of technologies"
#         # corresponding_parameter = optimization_setup.parameters.opex_specific_variable
#         # cls._construct_param(optimization_setup, par_name, par_index_names, par_doc, corresponding_parameter)
#
#     def _read_variances(self):
#         self.variance_price_export = self.data_input.extract_input_data(
#             "variance_price_export",
#             index_sets=[],
#             unit_category={},
#         )
#
#         self.variance_price_import = self.data_input.extract_input_data(
#             "variance_price_import",
#             index_sets=[],
#             unit_category={},
#         )
#
#         self.variance_price_shed_demand = self.data_input.extract_input_data(
#             "variance_price_shed_demand",
#             index_sets=[],
#             unit_category={},
#         )
#
#         self.variance_price_carbon_emissions = self.data_input.extract_input_data(
#             "variance_price_carbon_emissions",
#             index_sets=[],
#             unit_category={},
#         )
#
#         self.variance_capex_specific_conversion = self.data_input.extract_input_data(
#             "variance_capex_specific_conversion",
#             index_sets=[],
#             unit_category={},
#         )
#
#         self.variance_capex_specific_storage = self.data_input.extract_input_data(
#             "variance_capex_specific_storage",
#             index_sets=[],
#             unit_category={},
#         )
#
#         self.variance_capex_specific_transport = self.data_input.extract_input_data(
#             "variance_capex_specific_transport",
#             index_sets=[],
#             unit_category={},
#         )
        #
        # self.raw_time_series["variance_price_export"] = self.data_input.extract_input_data(
        #     "variance_price_export",
        #     index_sets=["set_carriers", "set_nodes", "set_time_steps"],
        #     time_steps="set_base_time_steps_yearly",
        #     unit_category={},
        # )
        #
        # self.raw_time_series["variance_price_import"] = self.data_input.extract_input_data(
        #     "variance_price_import",
        #     index_sets=["set_carriers", "set_nodes", "set_time_steps"],
        #     time_steps="set_base_time_steps_yearly",
        #     unit_category={},
        # )
        #
        # self.variance_price_shed_demand = self.data_input.extract_input_data(
        #     "variance_price_shed_demand",
        #     index_sets=["set_carriers"],
        #     time_steps="set_time_steps_yearly",
        #     unit_category={},
        # )
        #
        # self.variance_price_carbon_emissions = self.data_input.extract_input_data(
        #     "variance_price_carbon_emissions",
        #     index_sets=["set_time_steps_yearly"],
        #     time_steps="set_time_steps_yearly",
        #     unit_category={},
        # )
        #
        # self.variance_capex_specific_conversion = self.data_input.extract_input_data(
        #     "variance_capex_specific_conversion",
        #     index_sets=["set_conversion_technologies", "set_nodes", "set_time_steps_yearly"],
        #     time_steps="set_time_steps_yearly",
        #     unit_category={},
        # )
        #
        # self.variance_capex_specific_storage = self.data_input.extract_input_data(
        #     "variance_capex_specific_storage",
        #     index_sets=["set_storage_technologies", "set_nodes", "set_time_steps_yearly"],
        #     time_steps="set_time_steps_yearly",
        #     unit_category={},
        # )
        #
        # self.variance_capex_specific_transport = self.data_input.extract_input_data(
        #     "variance_capex_specific_transport",
        #     index_sets=["set_transport_technologies", "set_edges", "set_time_steps_yearly"],
        #     time_steps="set_time_steps_yearly",
        #     unit_category={},
        # )

        # self.variance_opex_specific_variable = self.data_input.extract_input_data(
        #     "variance_opex_specific_variable",
        #     index_sets=["set_technologies", "set_location", "set_time_steps_operation"],
        #     unit_category={"money": 1, "energy_quantity": -1},
        # )

class VarianceRules(GenericRule):
    """This class takes care of the rules for the mean-variance optimizatoin."""

    def __init__(self, optimization_setup):
        """Inits the constraints for a given energy system.

        :param optimization_setup: The OptimizationSetup of the EnergySystem class
        """
        super().__init__(optimization_setup)

    def constraint_variance_term(self):
        """
        Defines an objective function optimizing the mean-variance formulation.

        Todo:
            - Implement covariances between variables

        """
        weighting_factor = config.get("weighting_factor")

        #Import/export variances
        term_variance_import = (self.parameters.variance_price_import * self.parameters.price_import * self.variables["flow_import"] * self.variables["flow_import"]).sum(["set_carriers", "set_nodes", "set_time_steps_operation"])
        term_variance_export = (self.parameters.variance_price_import * self.parameters.price_export * self.variables["flow_export"] * self.variables["flow_export"]).sum(["set_carriers", "set_nodes", "set_time_steps_operation"])

        # Shed demand variance
        # replace inf with large number
        param = self.parameters.price_shed_demand
        price_shed_demand = param.where(
            np.isfinite(param), 1e6
        )
        term_variance_demand_shedding = (self.parameters.variance_price_shed_demand * price_shed_demand * self.variables["shed_demand"] * self.variables["shed_demand"]).sum(["set_carriers", "set_nodes", "set_time_steps_operation"])

        # Carbon emission variance
        # term_variance_carbon_emissions = self.parameters.variance_price_carbon_emissions * self.variables["carbon_emissions_annual"] * self.variables["carbon_emissions_annual"]

        # Capex variance conversion technologies
        capex_specific_conversion = self.parameters.capex_specific_conversion
        capex_specific_conversion = capex_specific_conversion.rename(
            {
                old: new
                for old, new in zip(
                    list(capex_specific_conversion.dims),
                    [
                        "set_conversion_technologies",
                        "set_nodes",
                        "set_time_steps_yearly",
                    ],
                    strict=False,
                )
            }
        )
        term_variance_capex_conversion = (self.parameters.variance_capex_specific_conversion * capex_specific_conversion * self.variables["capacity_approximation"] * self.variables["capacity_approximation"]).sum(["set_conversion_technologies", "set_nodes"])

        # if len(self.sets["set_storage_technologies"]) > 0:
        #     term_variance_capex_storage = (self.parameters.variance_capex_specific_storage * self.variables["capacity_addition"] * self.variables["capacity_addition"]).sum(["set_storage_technologies", "set_nodes", "set_time_steps_yearly"])
        # else:
        #     term_variance_capex_storage = 0
        #
        # if len(self.sets["set_transport_technologies"]) > 0:
        #     term_variance_capex_transport = (self.parameters.variance_capex_specific_transport * self.variables["capacity_addition"] * self.variables["capacity_addition"]).sum(["set_transport_technologies", "set_edges", "set_time_steps_yearly"])
        # else:
        #     term_variance_capex_transport = 0

        variance_term = (
                       term_variance_import +
                       term_variance_export +
                       term_variance_demand_shedding +
                       # term_variance_carbon_emissions +
                       term_variance_capex_conversion
                       # term_variance_capex_storage +
                       # term_variance_capex_transport
               ).sum("set_time_steps_yearly")

        npv_term = self.variables["net_present_cost"].sum("set_time_steps_yearly")

        return npv_term + weighting_factor * variance_term


@EventPublisher.register(Event.after_model_construction)
def construct_mean_variance_objective(optimization_setup=None):

    optimization_setup.model.remove_objective()
    rule = VarianceRules(optimization_setup)
    objective = rule.constraint_variance_term()
    sense = "min"
    optimization_setup.model.add_objective(objective, sense=sense)

