# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import Callable, Optional, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import random
from jax.lax import scan, switch
from jax.tree_util import Partial
from jax.typing import ArrayLike

from pyhgf.model import (
    add_binary_state,
    add_categorical_state,
    add_continuous_state,
    add_dp_state,
    add_ef_state,
    get_couplings,
)
from pyhgf.plots import graphviz, matplotlib, networkx
from pyhgf.typing import (
    Attributes,
    Edges,
    LearningSequence,
    NetworkParameters,
    Sequence,
    UpdateSequence,
)
from pyhgf.updates.learning import learning_weights_dynamic, learning_weights_fixed
from pyhgf.utils import (
    add_edges,
    beliefs_propagation,
    get_input_idxs,
    get_update_sequence,
    learning,
    sample,
    to_pandas,
)


class Network:
    """A predictive coding neural network.

    This is the core class to define and manipulate neural networks, that consists in
    1. attributes, 2. structure and 3. update sequences.

    Attributes
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.AdjacencyLists`. The tuple has the same length as the
        node number. For each node, the index lists the value/volatility
        parents/children.
    inputs :
        Information on the input nodes.
    node_trajectories :
        The dynamic of the node's beliefs after updating.
    update_sequence :
        The sequence of update functions that are applied during the belief propagation
        step.
    scan_fn :
        The function that is passed to :py:func:`jax.lax.scan`. This is a pre-
        parametrized version of :py:func:`pyhgf.networks.beliefs_propagation`.

    """

    def __init__(self, update_type: str = "eHGF") -> None:
        """Initialize an empty neural network.

        Parameters
        ----------
        update_type :
            The type of update to perform for volatility coupling. Can be `"eHGF"`
            (defaults), `"unbounded"` or `"standard"`. The unbounded approximation was
            recently introduced to avoid negative precisions updates, which greatly
            improve sampling performance. The eHGF update step was proposed as an
            alternative to the original definition in that it starts by updating the
            mean and then the precision of the parent node, which generally reduces the
            errors associated with impossible parameter space and improves sampling.

            .. note:
              The different update steps only apply to nodes having at least one
              volatility parents. In other cases, the regular HGF updates are applied.

        """
        self.edges: Edges = ()
        self.n_nodes: int = 0  # number of nodes in the network
        self.node_trajectories: dict = {}
        self.predictions: dict = {}
        self.attributes: Attributes = {-1: {"time_step": 0.0}}
        self.update_sequence: Optional[UpdateSequence] = None
        self.scan_fn: Optional[Callable] = None
        self.scan_fn_sample: Optional[Callable] = None
        self.additional_parameters: dict = {}
        self.input_dim: list = []
        self.action_steps: Optional[Sequence] = None
        self.last_attributes: Optional[Attributes] = None
        self.update_type = update_type

    @property
    def input_idxs(self):
        """Idexes of state nodes that can observe new data points by default."""
        input_idxs = get_input_idxs(self.edges)

        # check if the input indexes have been set manually
        if hasattr(self, "_input_idxs") and self._input_idxs is not None:
            return self._input_idxs

        # set the autoconnection strength and tonic volatility to 0
        for idx in input_idxs:
            if self.edges[idx].node_type == 2:
                self.attributes[idx]["autoconnection_strength"] = 0.0
                self.attributes[idx]["tonic_volatility"] = 0.0

        return input_idxs

    @input_idxs.setter
    def input_idxs(self, value):
        self._input_idxs = value

    def create_belief_propagation_fn(
        self,
        overwrite: bool = True,
        sampling_fn: bool = False,
    ) -> "Network":
        """Create the belief propagation function.

        .. note:
           This step is called by default when using py:meth:`input_data`.

        Parameters
        ----------
        overwrite :
            If `True` (default), create a new belief propagation function and ignore
            preexisting values. Otherwise, do not create a new function if the attribute
            `scan_fn` is already defined.
        sampling_fn :
            If `True`, also create a generative sampling function. This is used for
            generative sampling of the network. Defaults to `False`.

        """
        # get the dimension of the input nodes
        if not self.input_dim:
            self.get_input_dimension()

        # create the update sequence if it does not already exist
        if self.update_sequence is None:
            self.update_sequence = get_update_sequence(
                network=self, update_type=self.update_type
            )

        # create the belief propagation function
        # this function is used by scan to loop over observations
        if (self.scan_fn is None) or overwrite:
            self.scan_fn = Partial(
                beliefs_propagation,
                update_sequence=self.update_sequence,
                edges=self.edges,
                input_idxs=self.input_idxs,
            )

        # Create the generative scan function if it doesn't exist, and if requested.
        if (self.scan_fn_sample is None) and sampling_fn:
            self.sample_scan_fn = Partial(
                beliefs_propagation,
                update_sequence=self.update_sequence,
                edges=self.edges,
                input_idxs=self.input_idxs,
                observations="generative",
            )

        return self

    def create_learning_propagation_fn(
        self,
        inputs_x_idxs: tuple[int],
        inputs_y_idxs: tuple[int],
        overwrite: bool = True,
        lr: Union[str, float] = 0.2,
    ) -> "Network":
        """Create the belief propagation function.

        .. note:
           This step is called by default when using py:meth:`input_data`.

        Parameters
        ----------
        inputs_x_idxs :
            The indexes of the nodes receiving the predictors (x).
        inputs_y_idxs :
            The indexes of the nodes receiving the predictions (y).
        overwrite :
            If `True` (default), create a new belief propagation function and ignore
            preexisting values. Otherwise, do not create a new function if the attribute
            `scan_fn` is already defined.
        lr :
            The kind of learning to use for the coupling strengths.

        """
        # get the dimension of the input nodes
        if not self.input_dim:
            self.get_input_dimension()

        # create the update sequence if it does not already exist
        if self.update_sequence is None:
            self.update_sequence = get_update_sequence(
                network=self, update_type=self.update_type
            )
        # create the learning sequence
        # all nodes except the prediction nodes should update their coupling strengths
        if lr == "dynamic":
            learning_weights = learning_weights_dynamic
        elif isinstance(lr, float):
            learning_weights = Partial(learning_weights_fixed, lr=lr)
        else:
            raise ValueError("Invalid lr value. Should be 'dynamic' or a float value.")

        learning_steps = [
            (node_idx, learning_weights)
            for node_idx, _ in self.update_sequence.prediction_steps
            if (node_idx not in inputs_x_idxs)
            and (self.edges[node_idx].volatility_children is None)
        ]

        # do not update the last layer
        update_steps = tuple([
            step
            for step in self.update_sequence.update_steps
            if step[0] not in inputs_x_idxs
        ])
        # do not predict on the last layer
        prediction_steps = tuple([
            step
            for step in self.update_sequence.prediction_steps
            if step[0] not in inputs_x_idxs
        ])

        self.learning_sequence = LearningSequence(
            prediction_steps=prediction_steps,
            update_steps=update_steps,
            learning_steps=tuple(learning_steps),  # type: ignore
        )

        # create the learning propagation function
        # this function is used by scan to loop over predictors (x) and predictions (y)
        if (self.scan_fn is None) or overwrite:
            self.scan_fn = Partial(
                learning,
                learning_sequence=self.learning_sequence,
                edges=self.edges,
                inputs_x_idxs=inputs_x_idxs,
                inputs_y_idxs=inputs_y_idxs,
            )

        return self

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        inputs_x_idxs: tuple[int],
        inputs_y_idxs: tuple[int],
        lr: Union[str, float] = 0.2,
        overwrite: bool = True,
    ):
        """Add new observations.

        Parameters
        ----------
        x :
            A tuple of n arrays containing the new predictors (x). Predictors values are
            set to the obervation nodes defined by `inputs_x_idxs` before the prediction
            steps.
        y :
            A tuple of n arrays containing the resulting predictions (y). Predictions
            are observed in the observation steps in the nodes defined by
            `inputs_y_idxs`.
        inputs_x_idxs :
            The indexes of the nodes receiving the predictors (x).
        inputs_y_idxs :
            The indexes of the nodes receiving the predictions (y).
        lr :
            The kind of learning to use for the coupling strengths.

        """
        if x.ndim == 1:
            x = x[:, jnp.newaxis]
        if y.ndim == 1:
            y = y[:, jnp.newaxis]

        # generate the belief propagation function
        if (self.scan_fn is None) and overwrite:
            self = self.create_learning_propagation_fn(
                inputs_x_idxs=inputs_x_idxs, inputs_y_idxs=inputs_y_idxs, lr=lr
            )

        # wrap the inputs
        inputs = x, y

        # this is where the model loops over the whole input time series
        # at each time point, the node structure is traversed and beliefs are updated
        # using precision-weighted prediction errors
        last_attributes, node_trajectories = scan(self.scan_fn, self.attributes, inputs)

        # belief trajectories
        self.node_trajectories = node_trajectories
        self.last_attributes = last_attributes

        return self

    def input_data(
        self,
        input_data: np.ndarray,
        time_steps: Optional[np.ndarray] = None,
        observed: Optional[tuple[np.ndarray, ...]] = None,
        input_idxs: Optional[tuple[int]] = None,
        rng_keys: Optional[random.PRNGKey] = None,
    ):
        """Add new observations.

        Parameters
        ----------
        input_data :
            A 2d array of new observations (time x observation). Input nodes can receive
            floats or vectors according to their dimensions. This matrix is further
            split into tuple of columns accordingly.
        time_steps :
            Time steps vector (optional). If `None`, this will default to
            `np.ones(len(input_data))`.
        observed :
            A 2d array of mask (time x number of input nodes). In case of missing
            inputs, (i.e. `observed` is `0`), the input node will have value and
            volatility set to `0.0`. If the parent(s) of this input receive prediction
            error from other children, they simply ignore this one. If they are not
            receiving other prediction errors, they are updated by keeping the same
            mean by decreasing the precision as a function of time to reflect the
            evolution of the underlying Gaussian Random Walk.
        .. warning::
            Missing inputs are missing observations from the agent's perspective and
            should not be used to handle missing data points that were observed (e.g.
            missing in the event log, or rejected trials).
        input_idxs :
            Indexes on the state nodes receiving observations.
        rng_keys :
            Optional. A random key for the random number generator. This is only used
            when an action function is provided.

        """
        if rng_keys is not None:
            # get one key for each time step
            rng_keys = random.split(rng_keys, num=input_data.shape[0])

        if input_data.ndim == 1:
            input_data = input_data[:, jnp.newaxis]

        # set the input nodes indexes
        if input_idxs is not None:
            self.input_idxs = input_idxs

        # generate the belief propagation function
        if self.scan_fn is None:
            self = self.create_belief_propagation_fn()

        # time steps vector
        if time_steps is None:
            time_steps = np.ones(input_data.shape[0])

        # observation mask
        if observed is None:
            observed = tuple([
                np.ones(input_data.shape[0], dtype=int)
                for _ in range(len(self.input_idxs))
            ])
        elif isinstance(observed, np.ndarray):
            if observed.ndim == 1:
                observed = (observed,)
            else:
                observed = tuple(observed[:, i] for i in range(observed.shape[1]))

        # format input_data according to the input nodes dimension
        split_indices = np.cumsum(self.input_dim[:-1])
        values = tuple(np.split(input_data, split_indices, axis=1))

        # wrap the inputs
        inputs = values, observed, time_steps, rng_keys

        # this is where the model loops over the whole input time series
        # at each time point, the node structure is traversed and beliefs are updated
        # using precision-weighted prediction errors
        last_attributes, node_trajectories = scan(self.scan_fn, self.attributes, inputs)

        # belief trajectories
        self.node_trajectories = node_trajectories
        self.last_attributes = last_attributes

        return self

    def sample(
        self,
        n_predictions: int,
        time_steps: Optional[np.ndarray] = None,
        rng_key: ArrayLike = random.key(0),
    ):
        """Generate predictions using the utility predict function.

        Parameters
        ----------
        n_predictions :
            Number of predictions to generate.
        time_steps :
            Array of time steps.
        rng_key :
            Random number generator key, by default PRNGKey(0).


        """
        self.samples = sample(
            self, time_steps=time_steps, n_predictions=n_predictions, rng_key=rng_key
        )

        return self

    def input_custom_sequence(
        self,
        update_branches: tuple[UpdateSequence],
        branches_idx: np.ndarray,
        input_data: np.ndarray,
        time_steps: Optional[np.ndarray] = None,
        observed: Optional[tuple[np.ndarray, ...]] = None,
        input_idxs: Optional[tuple[int, ...]] = None,
        rng_keys: Optional[random.PRNGKey] = None,
    ):
        """Add new observations with custom update sequences.

        This method should be used when the update sequence is function of the input
        data. (e.g. in the case of missing/null observations that should not trigger
        node update).

        .. note::
           When the dynamic adaptation of the update sequence is not required, it is
           recommended to use :py:meth:`pyhgf.model.HGF.input_data` instead as this
           might result in performance improvement.

        Parameters
        ----------
        update_branches :
            A tuple of UpdateSequence listing the possible update sequences.
        branches_idx :
            The branches indexes (integers). Should have the same length as the input
            data.
        input_data :
            2d array of new observations (time x features).
        time_steps :
            Time vector (optional). If `None`, the time vector will default to
            `np.ones(len(input_data))`. This vector is automatically transformed
            into a time steps vector.
        observed :
            A 2d boolean array masking `input_data`. In case of missing inputs, (i.e.
            `observed` is `0`), the input node will have value and volatility set to
            `0.0`. If the parent(s) of this input receive prediction error from other
            children, they simply ignore this one. If they are not receiving other
            prediction errors, they are updated by keeping the same mean be decreasing
            the precision as a function of time to reflect the evolution of the
            underlying Gaussian Random Walk.
        .. warning::
            Missing inputs are missing observations from the agent's perspective and
            should not be used to handle missing data points that were observed (e.g.
            missing in the event log, or rejected trials).
        input_idxs :
            Indexes on the state nodes receiving observations.
        rng_keys :
            Optional. A random key for the random number generator. This is only used
            when an action function is provided.

        """
        if rng_keys is not None:
            # get one key for each time step
            rng_keys = random.split(rng_keys, num=input_data.shape[0])

        if input_data.ndim == 1:
            input_data = input_data[:, jnp.newaxis]

        # set the input nodes indexes
        if input_idxs is not None:
            self.input_idxs = input_idxs

        # get the dimension of the input nodes
        if not self.input_dim:
            self.get_input_dimension()

        # generate the belief propagation function
        if self.scan_fn is None:
            self = self.create_belief_propagation_fn()

        # time steps vector
        if time_steps is None:
            time_steps = np.ones(input_data.shape[0])

        # observation mask
        if observed is None:
            observed = tuple([
                np.ones(input_data.shape[0], dtype=int)
                for _ in range(len(self.input_idxs))
            ])

        # format input_data according to the input nodes dimension
        split_indices = np.cumsum(self.input_dim[:-1])
        values = tuple(np.split(input_data, split_indices, axis=1))

        # wrap the inputs
        inputs = values, observed, time_steps, rng_keys

        # create the update functions that will be scanned
        branches_fn = [
            Partial(
                beliefs_propagation,
                update_sequence=seq,
                edges=self.edges,
                input_idxs=self.input_idxs,
            )
            for seq in update_branches
        ]

        # create the function that will be scanned
        def switching_propagation(attributes, scan_input):
            data, idx = scan_input
            return switch(idx, branches_fn, attributes, data)

        # wrap the inputs
        scan_input = (inputs, branches_idx)

        # scan over the input data and apply the switching belief propagation functions
        _, node_trajectories = scan(switching_propagation, self.attributes, scan_input)

        # the node structure at each value updates
        self.node_trajectories = node_trajectories

        return self

    def get_network(self) -> NetworkParameters:
        """Return the attributes, edges and update sequence defining the network."""
        if self.scan_fn is None:
            self = self.create_belief_propagation_fn()

        assert self.update_sequence is not None

        return self.attributes, self.edges, self.update_sequence

    def add_nodes(
        self,
        kind: str = "continuous-state",
        n_nodes: int = 1,
        node_parameters: dict = {},
        value_children: Optional[Union[list, tuple, int]] = None,
        value_parents: Optional[Union[list, tuple, int]] = None,
        volatility_children: Optional[Union[list, tuple, int]] = None,
        volatility_parents: Optional[Union[list, tuple, int]] = None,
        coupling_fn: tuple[Optional[Callable], ...] = (None,),
        **additional_parameters,
    ):
        """Add new input/state node(s) to the neural network.

        Parameters
        ----------
        kind :
            The kind of node to create. If `"continuous-state"` (default), the node will
            be a regular state node that can have value and/or volatility
            parents/children. If `"binary-state"`, the node should be the
            value parent of a binary input. State nodes filtering distribution from the
            exponential family can be created using `"ef-state"`.

        .. note::
            When using a categorical state node, the `binary_parameters` can be used to
            parametrize the implied collection of binary HGFs.

        .. note:
            When using `categorical-state`, the implied `n` binary HGFs are
            automatically created with a shared volatility parent at the third level,
            resulting in a network with `3n + 2` nodes in total.

        n_nodes :
            The number of nodes to create (defaults to `1`).
        node_parameters :
            Dictionary of parameters. The default values are automatically inferred
            from the node type. Different values can be provided by passing them in the
            dictionary, which will overwrite the defaults.
        value_children :
            Indexes to the node's value children. The index can be passed as an integer
            or a list of integers, in case of multiple children. The coupling strength
            can be controlled by passing a tuple, where the first item is the list of
            indexes, and the second item is the list of coupling strengths.
        value_parents :
            Indexes to the node's value parents. The index can be passed as an integer
            or a list of integers, in case of multiple children. The coupling strength
            can be controlled by passing a tuple, where the first item is the list of
            indexes, and the second item is the list of coupling strengths.
        volatility_children :
            Indexes to the node's volatility children. The index can be passed as an
            integer or a list of integers, in case of multiple children. The coupling
            strength can be controlled by passing a tuple, where the first item is the
            list of indexes, and the second item is the list of coupling strengths.
        volatility_parents :
            Indexes to the node's volatility parents. The index can be passed as an
            integer or a list of integers, in case of multiple children. The coupling
            strength can be controlled by passing a tuple, where the first item is the
            list of indexes, and the second item is the list of coupling strengths.
        coupling_fn :
            Coupling function(s) between the current node and its value children.
            It has to be provided as a tuple. If multiple value children are specified,
            the coupling functions must be stated in the same order of the children.
            Note: if a node has multiple parents nodes with different coupling
            functions, a coupling function should be indicated for all the parent nodes.
            If no coupling function is stated, the relationship between nodes is assumed
            linear.
        **kwargs :
            Additional keyword parameters will be passed and overwrite the node
            attributes.

        """
        if kind not in [
            "dp-state",
            "ef-state",
            "categorical-state",
            "continuous-state",
            "binary-state",
        ]:
            raise ValueError(
                (
                    "Invalid node type. Should be one of the following: "
                    "'dp-state', 'continuous-state', 'binary-state', "
                    "'ef-state', or 'categorical-state'"
                )
            )

        # turn coupling indexes of various kinds
        # into tuples of indexes and coupling strength
        value_parents, volatility_parents, value_children, volatility_children = (
            get_couplings(
                value_parents=value_parents,
                volatility_parents=volatility_parents,
                value_children=value_children,
                volatility_children=volatility_children,
            )
        )

        # create the default parameters set according to the node type
        if kind == "continuous-state":
            self = add_continuous_state(
                network=self,
                n_nodes=n_nodes,
                value_parents=value_parents,
                volatility_parents=volatility_parents,
                value_children=value_children,
                volatility_children=volatility_children,
                node_parameters=node_parameters,
                additional_parameters=additional_parameters,
                coupling_fn=coupling_fn,
            )
        elif kind == "binary-state":
            self = add_binary_state(
                network=self,
                n_nodes=n_nodes,
                value_parents=value_parents,
                volatility_parents=volatility_parents,
                value_children=value_children,
                volatility_children=volatility_children,
                node_parameters=node_parameters,
                additional_parameters=additional_parameters,
            )
        elif kind == "ef-state":
            self = add_ef_state(
                network=self,
                n_nodes=n_nodes,
                node_parameters=node_parameters,
                additional_parameters=additional_parameters,
                value_children=value_children,
            )
        elif kind == "categorical-state":
            self = add_categorical_state(
                network=self,
                n_nodes=n_nodes,
                node_parameters=node_parameters,
                additional_parameters=additional_parameters,
            )
        elif kind == "dp-state":
            self = add_dp_state(
                network=self,
                n_nodes=n_nodes,
                node_parameters=node_parameters,
                additional_parameters=additional_parameters,
            )

        return self

    def plot_nodes(self, node_idxs: Union[int, list[int]], **kwargs):
        """Plot the node(s) beliefs trajectories."""
        return matplotlib.plot_nodes(network=self, node_idxs=node_idxs, **kwargs)

    def plot_trajectories(self, **kwargs):
        """Plot the parameters trajectories."""
        return matplotlib.plot_trajectories(network=self, **kwargs)

    def plot_samples(self, **kwargs):
        """Plot the parameters trajectories."""
        return matplotlib.plot_samples(network=self, **kwargs)

    def plot_correlations(self):
        """Plot the heatmap of cross-trajectories correlation."""
        return matplotlib.plot_correlations(network=self)

    def plot_network(self, backend: str = "graphviz"):
        """Visualization of node network using GraphViz or Networkx."""
        if backend == "graphviz":
            return graphviz.plot_network(network=self)
        elif backend == "networkx":
            return networkx.plot_network(network=self)
        else:
            raise ValueError(
                (
                    "Invalid backend."
                    " Should be one of the following: 'graphviz' or 'networkx'",
                )
            )
        

    def to_pandas(self) -> pd.DataFrame:
        """Export the nodes trajectories and surprise as a Pandas data frame.

        Returns
        -------
        structure_df :
            Pandas data frame with the time series of sufficient statistics and
            the surprise of each node in the structure.

        """
        return to_pandas(self)

    def surprise(
        self,
        response_function: Callable,
        response_function_inputs: tuple = (),
        response_function_parameters: Optional[Union[tuple, ArrayLike, float]] = None,
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
        return response_function(
            hgf=self,
            response_function_inputs=response_function_inputs,
            response_function_parameters=response_function_parameters,
        )
        return self

    def add_edges(
        self,
        kind="value",
        parent_idxs=Union[int, list[int]],
        children_idxs=Union[int, list[int]],
        coupling_strengths: Union[float, list[float], tuple[float]] = 1.0,
        coupling_fn: tuple[Optional[Callable], ...] = (None,),
    ) -> "Network":
        """Add a value or volatility coupling link between a set of nodes.

        Parameters
        ----------
        kind :
            The kind of coupling, can be `"value"` or `"volatility"`.
        parent_idxs :
            The index(es) of the parent node(s).
        children_idxs :
            The index(es) of the children node(s).
        coupling_strengths :
            The coupling strength betwen the parents and children.
        coupling_fn :
            Coupling function(s) between the current node and its value children.
            It has to be provided as a tuple. If multiple value children are specified,
            the coupling functions must be stated in the same order of the children.
            Note: if a node has multiple parents nodes with different coupling
            functions, a coupling function should be indicated for all the parent nodes.
            If no coupling function is stated, the relationship between nodes is assumed
            linear.

        """
        attributes, edges = add_edges(
            attributes=self.attributes,
            edges=self.edges,
            kind=kind,
            parent_idxs=parent_idxs,
            children_idxs=children_idxs,
            coupling_strengths=coupling_strengths,
            coupling_fn=coupling_fn,
        )

        self.attributes = attributes
        self.edges = edges

        return self

    def get_input_dimension(
        self,
    ) -> "Network":
        """Get input node dimensions.

        All nodes have dimension 1, except exponential family state nodes and
        categorical state node.
        """
        for idx in self.input_idxs:
            if self.edges[idx].node_type == 3:
                dim = self.attributes[idx]["dimension"]
            elif self.edges[idx].node_type == 5:
                dim = self.attributes[idx]["n_categories"]
            else:
                dim = 1
            self.input_dim.append(dim)
        return self