# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import Callable, Optional, Union

import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from pyhgf.model import Network
from pyhgf.response import first_level_binary_surprise, first_level_gaussian_surprise


class HGF(Network):
    """The two-level and three-level Hierarchical Gaussian Filters (HGF).

    This class uses pre-made node structures that correspond to the most widely used
    HGF. The inputs can be continuous or binary.

    Attributes
    ----------
    model_type :
        The model implemented (can be `"continuous"`, `"binary"` or `"custom"`).
    n_levels :
        The number of hierarchies in the model, including the input vector. It cannot be
        less than 2.
    .. note::
        The parameter structure also incorporates the value and volatility coupling
        strength with children and parents (i.e. `"value_coupling_parents"`,
        `"value_coupling_children"`, `"volatility_coupling_parents"`,
        `"volatility_coupling_children"`).

    """

    def __init__(
        self,
        n_levels: Optional[int] = 2,
        model_type: str = "continuous",
        initial_mean: dict = {
            "1": 0.0,
            "2": 0.0,
            "3": 0.0,
        },
        initial_precision: dict = {
            "1": 1.0,
            "2": 1.0,
            "3": 1.0,
        },
        continuous_precision: Union[float, np.ndarray, ArrayLike] = 1e4,
        tonic_volatility: dict = {
            "1": -3.0,
            "2": -3.0,
            "3": -3.0,
        },
        volatility_coupling: dict = {"1": 1.0, "2": 1.0},
        eta0: Union[float, np.ndarray, ArrayLike] = 0.0,
        eta1: Union[float, np.ndarray, ArrayLike] = 1.0,
        binary_precision: Union[float, np.ndarray, ArrayLike] = jnp.inf,
        tonic_drift: dict = {
            "1": 0.0,
            "2": 0.0,
            "3": 0.0,
        },
    ) -> None:
        r"""Parameterization of the HGF model.

        Parameters
        ----------
        n_levels :
            The number of hierarchies in the perceptual model (can be `2` or `3`). If
            `None`, the nodes hierarchy is not created and might be provided afterwards.
            Defaults to `2` for a 2-level HGF.
        model_type : str
            The model type to use (can be `"continuous"` or `"binary"`).
        initial_mean :
            A dictionary containing the initial values for the initial mean at
            different levels of the hierarchy. Defaults set to `0.0`.
        initial_precision :
            A dictionary containing the initial values for the initial precision at
            different levels of the hierarchy. Defaults set to `1.0`.
        continuous_precision :
            The expected precision of the continuous input node. Default to `1e4`. Only
            relevant if `model_type="continuous"`.
        tonic_volatility :
            A dictionary containing the initial values for the tonic volatility
            at different levels of the hierarchy. This represents the tonic
            part of the variance (the part that is not affected by the parent node).
            Defaults are set to `-3.0`.
        volatility_coupling :
            A dictionary containing the initial values for the volatility coupling
            at different levels of the hierarchy. This represents the phasic part of
            the variance (the part that is affected by the parent nodes) and will
            define the strength of the connection between the node and the parent
            node. Defaults set to `1.0`.
        eta0 :
            The first categorical value of the binary node. Defaults to `0.0`. Only
            relevant if `model_type="binary"`.
        eta1 :
            The second categorical value of the binary node. Defaults to `0.0`. Only
            relevant if `model_type="binary"`.
        binary_precision :
            The precision of the binary input node. Default to `jnp.inf`. Only relevant
            if `model_type="binary"`.
        tonic_drift :
            A dictionary containing the initial values for the tonic drift
            at different levels of the hierarchy. This represents the drift of the
            random walk. Defaults set all entries to `0.0` (no drift).

        """
        Network.__init__(self)
        self.model_type = model_type
        self.n_levels = n_levels

        if model_type not in ["continuous", "binary"]:
            raise ValueError("Invalid model type.")
        else:
            if model_type == "continuous":
                # X - 0
                self.add_nodes(
                    precision=continuous_precision,
                )

                # X - 1
                self.add_nodes(
                    value_children=([0], [1.0]),
                    node_parameters={
                        "mean": initial_mean["1"],
                        "precision": initial_precision["1"],
                        "tonic_volatility": tonic_volatility["1"],
                        "tonic_drift": tonic_drift["1"],
                    },
                )

                # X - 2
                self.add_nodes(
                    volatility_children=([1], [volatility_coupling["1"]]),
                    node_parameters={
                        "mean": initial_mean["2"],
                        "precision": initial_precision["2"],
                        "tonic_volatility": tonic_volatility["2"],
                        "tonic_drift": tonic_drift["2"],
                    },
                )

                #########################
                # Meta volatility level #
                #########################
                if self.n_levels == 3:
                    self.add_nodes(
                        volatility_children=([2], [volatility_coupling["2"]]),
                        node_parameters={
                            "mean": initial_mean["3"],
                            "precision": initial_precision["3"],
                            "tonic_volatility": tonic_volatility["3"],
                            "tonic_drift": tonic_drift["3"],
                        },
                    )

            elif model_type == "binary":

                if binary_precision == jnp.inf:
                    # X - 0
                    self.add_nodes(
                        kind="binary-state",
                        node_parameters={
                            "mean": initial_mean["1"],
                            "precision": initial_precision["1"],
                        },
                    )

                # X - 1
                self.add_nodes(
                    kind="continuous-state",
                    value_children=([0], [1.0]),
                    node_parameters={
                        "mean": initial_mean["2"],
                        "precision": initial_precision["2"],
                        "tonic_volatility": tonic_volatility["2"],
                        "tonic_drift": tonic_drift["2"],
                    },
                )

                #########################
                # Meta volatility level #
                #########################
                if self.n_levels == 3:
                    self.add_nodes(
                        volatility_children=([1], [volatility_coupling["2"]]),
                        node_parameters={
                            "mean": initial_mean["3"],
                            "precision": initial_precision["3"],
                            "tonic_volatility": tonic_volatility["3"],
                            "tonic_drift": tonic_drift["3"],
                        },
                    )

            # initialize the model so it is ready to receive new observations
            self.create_belief_propagation_fn()

    def surprise(
        self,
        response_function: Optional[Callable] = None,
        response_function_inputs: Optional[Union[ArrayLike, tuple]] = None,
        response_function_parameters: Optional[
            Union[np.ndarray, ArrayLike, float]
        ] = None,
    ) -> float:
        """Surprise of the model conditioned by the response function.

        The surprise (negative log probability) depends on the input data, the model
        parameters, the response function, its inputs and its additional parameters
        (optional).

        Parameters
        ----------
        response_function :
            The response function to use to compute the model surprise. If `None`
            (default), return the sum of Gaussian surprise if `model_type=="continuous"`
            or the sum of the binary surprise if `model_type=="binary"`.
        response_function_inputs :
            A list of tuples with the same length as the number of models. Each tuple
            contains additional data and parameters that can be accessible to the
            response functions.
        response_function_parameters :
            A list of additional parameters that will be passed to the response
            function. This can include values over which inferece is performed in a
            PyMC model (e.g. the inverse temperature of a binary softmax).

        Returns
        -------
        surprise :
            The model's surprise given the input data and the response function.

        """
        if response_function is None:
            response_function = (
                first_level_gaussian_surprise
                if self.model_type == "continuous"
                else first_level_binary_surprise
            )

        return response_function(
            hgf=self,
            response_function_inputs=response_function_inputs,
            response_function_parameters=response_function_parameters,
        )
