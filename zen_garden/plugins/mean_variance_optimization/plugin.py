import linopy as lp
import pandas as pd
import numpy as np
import json
from pathlib import Path

from zen_garden.plugin_system.events import Event, EventPublisher
from zen_garden.model.element import GenericRule, Element
from zen_garden.model.component import IndexSet
from zen_garden.preprocess.extract_input_data import DataInput
from zen_garden.preprocess.unit_handling import UnitHandling


config = {
    "weighting_factor": None,
    "include_variances_for": ["capex", "opex", "import", "export", "demand_shedding"],
}

def _update_dict(dict, dict_to_add, fields_to_update):

    for key in fields_to_update:
        if key not in dict and key in dict_to_add:
            dict[key] = dict_to_add[key]
    return dict

def _update_attribute_in_json(dir, attributes_to_update: list, attribute_value):
    attr_file = dir / "attributes.json"

    with open(attr_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    updated_data = _update_dict(data, attribute_value, attributes_to_update)

    with open(attr_file, "w", encoding="utf-8") as f:
            json.dump(updated_data, f, indent=2)

def _update_carrier_attributes(variance_attributes, config):
    carrier_path = Path(config.analysis.dataset) / "set_carriers"

    for carrier_dir in carrier_path.iterdir():
        update_attributes = ["variance_price_export", "variance_price_import", "variance_price_shed_demand"]
        _update_attribute_in_json(carrier_dir, update_attributes, variance_attributes)


def _update_conversion_technology_attributes(variance_attributes, config):

    tech_path = Path(config.analysis.dataset) / "set_technologies" / "set_conversion_technologies"
    for tech_dir in tech_path.iterdir():
        if not "set_retrofitting_technologies" in tech_dir.parts:
            update_attributes = ["variance_capex_specific_conversion", "variance_opex_specific_variable"]
            _update_attribute_in_json(tech_dir, update_attributes, variance_attributes)

def _update_storage_technology_attributes(variance_attributes, config):

    tech_path = Path(config.analysis.dataset) / "set_technologies" / "set_storage_technologies"
    for tech_dir in tech_path.iterdir():
        update_attributes = ["variance_capex_specific_storage", "variance_opex_specific_variable"]
        _update_attribute_in_json(tech_dir, update_attributes, variance_attributes)

def _update_transport_technology_attributes(variance_attributes, config):

    tech_path = Path(config.analysis.dataset) / "set_technologies" / "set_transport_technologies"
    for tech_dir in tech_path.iterdir():
        update_attributes = ["variance_capex_specific_transport", "variance_opex_specific_variable"]
        _update_attribute_in_json(tech_dir, update_attributes, variance_attributes)

def _update_retrofitting_technology_attributes(variance_attributes, config):

    tech_path = Path(config.analysis.dataset) / "set_technologies" / "set_conversion_technologies" / "set_retrofitting_technologies"
    if tech_path.exists():
        for tech_dir in tech_path.iterdir():
            update_attributes = ["variance_capex_specific_retrofitting", "variance_opex_specific_variable"]
            _update_attribute_in_json(tech_dir, update_attributes, variance_attributes)

@EventPublisher.register(Event.on_preprocessing)
def add_variance_to_attribute_jsons(config):
    # Add variance fields to attribute.json of carriers
    variance_attribute_path = Path(config.analysis.dataset) / "variances" / "attributes.json"

    with open(variance_attribute_path, "r") as f:
        variance_attributes = json.load(f)

    _update_carrier_attributes(variance_attributes, config)
    _update_conversion_technology_attributes(variance_attributes, config)
    _update_storage_technology_attributes(variance_attributes, config)
    _update_transport_technology_attributes(variance_attributes, config)
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


@EventPublisher.register(Event.on_technology_store_input_data)
def add_variance_to_technology(technology):
    set_location = technology.location_type

    technology.variance_opex_specific_variable = technology.data_input.extract_input_data(
        "variance_opex_specific_variable",
        index_sets=[set_location, "set_time_steps"],
        time_steps="set_base_time_steps_yearly",
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

@EventPublisher.register(Event.on_storage_technology_store_input_data)
def add_variance_to_storage_technology(technology):
    technology.variance_capex_specific_storage = technology.data_input.extract_input_data(
        "variance_capex_specific_storage",
        index_sets=["set_nodes", "set_time_steps_yearly"],
        time_steps="set_time_steps_yearly",
        unit_category={},
    )

@EventPublisher.register(Event.on_transport_technology_store_input_data)
def add_variance_to_transport_technology(technology):
    technology.variance_capex_specific_transport = technology.data_input.extract_input_data(
        "variance_capex_specific_transport",
        index_sets=["set_edges", "set_time_steps_yearly"],
        time_steps="set_time_steps_yearly",
        unit_category={},
    )

@EventPublisher.register(Event.on_retrofitting_technology_store_input_data)
def add_variance_to_retrofitting_technology(technology):
    technology.variance_capex_specific_retrofitting = technology.data_input.extract_input_data(
        "variance_capex_specific_retrofitting",
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


@EventPublisher.register(Event.on_technology_construct_params)
def add_variance_parameters_to_technology(optimization_setup, technology_cls):
    optimization_setup.parameters.add_parameter(
            name="variance_opex_specific_variable",
            index_names=[
                "set_technologies",
                "set_location",
                "set_time_steps_operation",
            ],
            doc="Variance of specific opex of technologies",
            calling_class=technology_cls,
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

@EventPublisher.register(Event.on_storage_technology_construct_params)
def add_variance_parameters_to_storage_technology(optimization_setup, technology_cls):
    optimization_setup.parameters.add_parameter(
            name="variance_capex_specific_storage",
            index_names=[
                "set_storage_technologies",
                "set_nodes",
                "set_time_steps_yearly",
            ],
            doc="Variance of specific capex of storage technologies",
            calling_class=technology_cls,
        )

@EventPublisher.register(Event.on_transport_technology_construct_params)
def add_variance_parameters_to_transport_technology(optimization_setup, technology_cls):
    optimization_setup.parameters.add_parameter(
            name="variance_capex_specific_transport",
            index_names=[
                "set_transport_technologies",
                "set_edges",
                "set_time_steps_yearly",
            ],
            doc="Variance of specific capex of transport technologies",
            calling_class=technology_cls,
        )

# @EventPublisher.register(Event.on_retrofit_technology_construct_params)
# def add_variance_parameters_to_retrofitting_technology(optimization_setup, technology_cls):
#     optimization_setup.parameters.add_parameter(
#             name="variance_capex_specific_retrofitting",
#             index_names=[
#                 "set_retrofitting_technologies",
#                 "set_nodes",
#                 "set_time_steps_yearly",
#             ],
#             doc="Variance of specific capex of retrofitting technologies",
#             calling_class=technology_cls,
#         )




#         self.variance_price_carbon_emissions = self.data_input.extract_input_data(
#             "variance_price_carbon_emissions",
#             index_sets=[],
#             unit_category={},
#         )
#
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

    def _construct_import_term(self):
        return (self.parameters.variance_price_import * self.parameters.price_import * self.variables["flow_import"] * self.variables["flow_import"]).sum(["set_carriers", "set_nodes", "set_time_steps_operation"])

    def _construct_export_term(self):
        return (self.parameters.variance_price_import * self.parameters.price_export * self.variables["flow_export"] * self.variables["flow_export"]).sum(["set_carriers", "set_nodes", "set_time_steps_operation"])

    def _construct_demand_shedding_term(self):
        # replace inf with large number
        param = self.parameters.price_shed_demand
        price_shed_demand = param.where(
            np.isfinite(param), 1e6
        )

        return (self.parameters.variance_price_shed_demand * price_shed_demand * self.variables["shed_demand"] * self.variables["shed_demand"]).sum(["set_carriers", "set_nodes", "set_time_steps_operation"])

    def _construct_conversion_capex_technology_term(self):
        # Capex variance
        techs = self.sets["set_conversion_technologies"]
        nodes = self.sets["set_nodes"]
        capacity_addition = self.variables["capacity_addition"].rename(
            {
                "set_technologies": "set_conversion_technologies",
                "set_location": "set_nodes",
            }
        )
        capacity_addition = capacity_addition.sel({"set_nodes": nodes, "set_conversion_technologies": techs})



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
        term_variance_capex = (
                    self.parameters.variance_capex_specific_conversion * capex_specific_conversion * capacity_addition * capacity_addition).sum(
            ["set_conversion_technologies", "set_nodes", "set_capacity_types"])

        return term_variance_capex


    def _construct_conversion_opex_technology_term(self):
        # Opex variance
        techs = self.sets["set_conversion_technologies"]
        nodes = self.sets["set_nodes"]
        opex_parameter = self.parameters.opex_specific_variable.rename(
            {
                "set_technologies": "set_conversion_technologies",
                "set_location": "set_nodes",
            }
        )
        variance_opex = self.parameters.variance_opex_specific_variable.rename(
            {
                "set_technologies": "set_conversion_technologies",
                "set_location": "set_nodes",
            }
        )


        terms = []
        for t in techs:
            rc = self.sets["set_reference_carriers"][t][0]
            if rc in self.sets["set_input_carriers"][t]:
                terms.append(
                    opex_parameter.loc[t, nodes] * variance_opex.loc[t, nodes] * self.variables["flow_conversion_input"].loc[t, rc, nodes, :] * self.variables["flow_conversion_input"].loc[t, rc, nodes, :]
                )
            else:
                terms.append(
                    opex_parameter.loc[t, nodes] * variance_opex.loc[t, nodes] * self.variables["flow_conversion_output"].loc[t, rc, nodes, :] * self.variables["flow_conversion_output"].loc[t, rc, nodes, :]
                )
        expression = lp.merge(
            terms,
            dim="set_conversion_technologies",
            join="outer",
            coords="minimal",
            compat="override",
        )

        term_variance_opex = (expression).sum(["set_conversion_technologies", "set_nodes", "set_time_steps_operation"])

        return term_variance_opex

    def _construct_storage_capex_technology_term(self):
        techs = self.sets["set_storage_technologies"]
        nodes = self.sets["set_nodes"]
        if len(techs) == 0:
            return 0
        else:
            # Capex variance
            capacity_addition = self.variables["capacity_addition"].rename(
                {
                    "set_technologies": "set_storage_technologies",
                    "set_location": "set_nodes",
                }
            )
            capacity_addition = capacity_addition.sel({"set_nodes": nodes, "set_storage_technologies": techs})
            term_variance_capex = (self.parameters.variance_capex_specific_storage * self.parameters.capex_specific_storage * capacity_addition * capacity_addition).sum(["set_storage_technologies", "set_nodes", "set_capacity_types"])
            return term_variance_capex


    def _construct_storage_opex_technology_term(self):
        techs = self.sets["set_storage_technologies"]
        nodes = self.sets["set_nodes"]
        if len(techs) == 0:
            return 0
        else:
            # Opex variance
            opex_parameter = self.parameters.opex_specific_variable.sel({"set_technologies": techs, "set_location": nodes}).rename(
                {
                    "set_technologies": "set_storage_technologies",
                    "set_location": "set_nodes",
                }
            )
            variance_opex = self.parameters.variance_opex_specific_variable.sel({"set_technologies": techs, "set_location": nodes}).rename(
                {
                    "set_technologies": "set_storage_technologies",
                    "set_location": "set_nodes",
                }
            )
            flow_charge = self.variables["flow_storage_charge"].sel({"set_storage_technologies": techs, "set_nodes": nodes})
            flow_discharge = self.variables["flow_storage_discharge"].sel({"set_storage_technologies": techs, "set_nodes": nodes})

            expression = variance_opex * opex_parameter * flow_charge * flow_charge +  variance_opex * opex_parameter * flow_discharge * flow_discharge
            term_variance_opex = expression.sum(["set_storage_technologies", "set_nodes", "set_time_steps_operation"])
            return term_variance_opex

    def _construct_transport_capex_technology_term(self):
        techs = self.sets["set_transport_technologies"]
        edges = self.sets["set_edges"]
        if len(techs) == 0:
            return 0
        else:
            # Capex variance
            capacity_type = "power"
            capacity_addition = self.variables["capacity_addition"].rename(
                {
                    "set_technologies": "set_transport_technologies",
                    "set_location": "set_edges",
                }
            )
            capacity_addition = capacity_addition.sel(
                {"set_edges": edges, "set_transport_technologies": techs, "set_capacity_types": capacity_type})

            term_variance_capex = (self.parameters.variance_capex_specific_transport * self.parameters.capex_specific_transport * capacity_addition * capacity_addition).sum(["set_transport_technologies", "set_edges"])
            return term_variance_capex

    def _construct_transport_opex_technology_term(self):
        techs = self.sets["set_transport_technologies"]
        edges = self.sets["set_edges"]
        if len(techs) == 0:
            return 0
        else:
            # Opex variance
            opex_parameter = self.parameters.opex_specific_variable.sel({"set_technologies": techs, "set_location": edges}).rename(
                {
                    "set_technologies": "set_transport_technologies",
                    "set_location": "set_edges",
                }
            )
            variance_opex = self.parameters.variance_opex_specific_variable.sel({"set_technologies": techs, "set_location": edges}).rename(
                {
                    "set_technologies": "set_transport_technologies",
                    "set_location": "set_edges",
                }
            )
            flow = self.variables["flow_transport"].sel({"set_transport_technologies": techs, "set_edges": edges})
            expression = variance_opex * opex_parameter * flow * flow
            term_variance_opex = expression.sum(["set_transport_technologies", "set_edges", "set_time_steps_operation"])

            return term_variance_opex

    def _construct_capex_term(self):
        term_capex_variance_conversion_techs = self._construct_conversion_capex_technology_term()
        term_capex_variance_storage_techs = self._construct_storage_capex_technology_term()
        term_capex_variance_transport_techs = self._construct_transport_capex_technology_term()

        return term_capex_variance_conversion_techs + term_capex_variance_storage_techs + term_capex_variance_transport_techs

    def _construct_opex_term(self):
        term_opex_variance_conversion_techs = self._construct_conversion_opex_technology_term()
        term_opex_variance_storage_techs = self._construct_storage_opex_technology_term()
        term_opex_variance_transport_techs = self._construct_transport_opex_technology_term()

        return term_opex_variance_conversion_techs + term_opex_variance_storage_techs + term_opex_variance_transport_techs

    def constraint_variance_term(self):
        """
        Defines an objective function optimizing the mean-variance formulation.

        Todo:
            - Implement covariances between variables
            - Implement retrofitting technologies

        """
        weighting_factor = config.get("weighting_factor")

        #Import/export variances
        if "import" in config.get("include_variances_for"):
            term_variance_import = self._construct_import_term()
        else:
            term_variance_import = 0

        if "export" in config.get("include_variances_for"):
            term_variance_export = self._construct_export_term()
        else:
            term_variance_export = 0

        # Shed demand variances
        if "demand_shedding" in config.get("include_variances_for"):
            term_variance_demand_shedding = self._construct_demand_shedding_term()
        else:
            term_variance_demand_shedding = 0

        # technology capex variances
        if "capex" in config.get("include_variances_for"):
            term_variances_capex = self._construct_capex_term()
        else:
            term_variances_capex = 0

        # technology opex variances
        if "opex" in config.get("include_variances_for"):
            term_variances_opex = self._construct_capex_term()
        else:
            term_variances_opex = 0


        # Carbon emission variance
        # term_variance_carbon_emissions = self.parameters.variance_price_carbon_emissions * self.variables["carbon_emissions_annual"] * self.variables["carbon_emissions_annual"]

        variance_term = (
                       term_variance_import +
                       term_variance_export +
                       term_variance_demand_shedding +
                       term_variances_capex +
                       term_variances_opex
                       # term_variance_carbon_emissions +
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

