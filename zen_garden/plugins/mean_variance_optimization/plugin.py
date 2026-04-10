import linopy as lp
import pandas as pd

from zen_garden.plugin_system.events import Event, EventPublisher
from zen_garden.model.element import GenericRule

# Todo: add types for type checking
config = {
    "weighting_factor": None
}

def remove_objective_from_model(model):
    """
    Removes the existing objective from the optimization model.
    """
    model.remove_objective()

def read_variances(optimization_setup):
    """
    Reads the variances and returns them as a dictionary with an entry for each variable.

    Todo:
        - make path to variances file more flexible
        - implement indexing of variables
    """
    dict = pd.read_excel("C:/Users/jwiegner/ZEN_universe/ZEN-garden-productive/VarianceFactors.xlsx", index_col=0).to_dict()
    return dict["Variance"]

class MeanVarianceRules(GenericRule):
    """This class takes care of the rules for the mean-variance optimizatoin."""

    def __init__(self, optimization_setup):
        """Inits the constraints for a given energy system.

        :param optimization_setup: The OptimizationSetup of the EnergySystem class
        """
        super().__init__(optimization_setup)

    def define_mean_variance_objective(self, variances, weighting_factor):
        """
        Defines an objective function optimizing the mean-variance formulation.

        """
        quad_terms = []
        model = self.optimization_setup.model

        for name in model.variables:
            var = model.variables[name].stack(flat=model.variables[name].dims)
            # convert variable to a linear expression and form its elementwise square
            le = var.to_linexpr()
            quad = variances[name] * le * le
            # sum over all indices of the quadratic expression to get a scalar term
            quad_terms.append(quad.sum())

        return lp.expressions.merge(quad_terms) + model.variables[
            "net_present_cost"
        ].sum("set_time_steps_yearly")


@EventPublisher.register(Event.after_model_construction)
def construct_mean_variance_objective(optimization_setup=None):
    remove_objective_from_model(optimization_setup.model)
    variances = read_variances(optimization_setup)
    weighting_factor = config.get("weighting_factor")

    rules = MeanVarianceRules(optimization_setup)
    objective = rules.define_mean_variance_objective(variances, weighting_factor)
    sense = "min"
    optimization_setup.model.add_objective(objective, sense=sense)

