# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd
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
from pyhgf.plots import plot_correlations, plot_network, plot_nodes, plot_trajectories
from pyhgf.typing import Attributes, Edges, NetworkParameters, UpdateSequence
from pyhgf.utils import (
    add_edges,
    beliefs_propagation,
    get_input_idxs,
    get_update_sequence,
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

    def __init__(self) -> None:
        """Initialize an empty neural network."""
        self.edges: Edges = ()
        self.n_nodes: int = 0  # number of nodes in the network
        self.node_trajectories: Dict = {}
        self.attributes: Attributes = {-1: {"time_step": 0.0}}
        self.update_sequence: Optional[UpdateSequence] = None
        self.scan_fn: Optional[Callable] = None
        self.additional_parameters: Dict = {}
        self.input_dim: list = []

    @property
    def input_idxs(self):
        """Idexes of state nodes that can observe new data points by default."""
        input_idxs = get_input_idxs(self.edges)

        # set the autoconnection strength and tonic volatility to 0
        for idx in input_idxs:
            if self.edges[idx].node_type == 2:
                self.attributes[idx]["autoconnection_strength"] = 0.0
                self.attributes[idx]["tonic_volatility"] = 0.0
        return input_idxs

    @input_idxs.setter
    def input_idxs(self, value):
        self.input_idxs = value

    def create_belief_propagation_fn(
        self, overwrite: bool = True, update_type: str = "eHGF"
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
        update_type :
            The type of update to perform for volatility coupling. Can be `"eHGF"`
            (defaults) or `"standard"`. The eHGF update step was proposed as an
            alternative to the original definition in that it starts by updating the
            mean and then the precision of the parent node, which generally reduces the
            errors associated with impossible parameter space and improves sampling.

        """
        # get the dimension of the input nodes
        if not self.input_dim:
            self.get_input_dimension()

        # create the update sequence if it does not already exist
        if self.update_sequence is None:
            self.update_sequence = get_update_sequence(
                network=self, update_type=update_type
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

        return self

    def input_data(
        self,
        input_data: np.ndarray,
        time_steps: Optional[np.ndarray] = None,
        observed: Optional[np.ndarray] = None,
        input_idxs: Optional[Tuple[int]] = None,
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

        """
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
            observed = tuple(
                [
                    np.ones(input_data.shape[0], dtype=int)
                    for _ in range(len(self.input_idxs))
                ]
            )
        elif isinstance(observed, np.ndarray):
            if observed.ndim == 1:
                observed = (observed,)
            else:
                observed = tuple(observed[:, i] for i in range(observed.shape[1]))

        # format input_data according to the input nodes dimension
        split_indices = np.cumsum(self.input_dim[:-1])
        values = tuple(np.split(input_data, split_indices, axis=1))

        # wrap the inputs
        inputs = values, observed, time_steps

        # this is where the model loops over the whole input time series
        # at each time point, the node structure is traversed and beliefs are updated
        # using precision-weighted prediction errors
        last_attributes, node_trajectories = scan(self.scan_fn, self.attributes, inputs)

        # belief trajectories
        self.node_trajectories = node_trajectories
        self.last_attributes = last_attributes

        return self

    def input_custom_sequence(
        self,
        update_branches: Tuple[UpdateSequence],
        branches_idx: np.array,
        input_data: np.ndarray,
        time_steps: Optional[np.ndarray] = None,
        observed: Optional[np.ndarray] = None,
        input_idxs: Optional[Tuple[int]] = None,
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

        """
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
            observed = tuple(
                [
                    np.ones(input_data.shape[0], dtype=int)
                    for _ in range(len(self.input_idxs))
                ]
            )

        # format input_data according to the input nodes dimension
        split_indices = np.cumsum(self.input_dim[:-1])
        values = tuple(np.split(input_data, split_indices, axis=1))

        # wrap the inputs
        inputs = values, observed, time_steps

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
        node_parameters: Dict = {},
        value_children: Optional[Union[List, Tuple, int]] = None,
        value_parents: Optional[Union[List, Tuple, int]] = None,
        volatility_children: Optional[Union[List, Tuple, int]] = None,
        volatility_parents: Optional[Union[List, Tuple, int]] = None,
        coupling_fn: Tuple[Optional[Callable], ...] = (None,),
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
                value_parents, volatility_parents, value_children, volatility_children
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

    def plot_nodes(self, node_idxs: Union[int, List[int]], **kwargs):
        """Plot the node(s) beliefs trajectories."""
        return plot_nodes(network=self, node_idxs=node_idxs, **kwargs)

    def plot_trajectories(self, **kwargs):
        """Plot the parameters trajectories."""
        return plot_trajectories(network=self, **kwargs)

    def plot_correlations(self):
        """Plot the heatmap of cross-trajectories correlation."""
        return plot_correlations(network=self)

    def plot_network(self):
        """Visualization of node network using GraphViz."""
        return plot_network(network=self)

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
        response_function_inputs: Tuple = (),
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
        return response_function(
            hgf=self,
            response_function_inputs=response_function_inputs,
            response_function_parameters=response_function_parameters,
        )
        return self

    def add_edges(
        self,
        kind="value",
        parent_idxs=Union[int, List[int]],
        children_idxs=Union[int, List[int]],
        coupling_strengths: Union[float, List[float], Tuple[float]] = 1.0,
        coupling_fn: Tuple[Optional[Callable], ...] = (None,),
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
