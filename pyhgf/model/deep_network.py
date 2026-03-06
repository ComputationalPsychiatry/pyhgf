# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np

from pyhgf.model.network import Network
from pyhgf.utils import set_coupling
from pyhgf.utils.weight_initialisation import (
    he_init,
    orthogonal_init,
    sparse_init,
    xavier_init,
)


class DeepNetwork(Network):
    """A deep network with fully connected layers for predictive coding.

    This class extends Network with automatic layer tracking and convenience methods
    for building, visualizing, and training deep hierarchical networks.

    Unlike the base Network class, DeepNetwork automatically tracks layer structure
    as you add layers, providing simplified interfaces for plotting and fitting.

    Examples
    --------
    >>> # Build a deep network with method chaining
    >>> net = (
    ...     DeepNetwork()
    ...     .add_nodes(kind="continuous-state", n_nodes=10)
    ...     .add_layer(size=8, precision=1.0)
    ...     .add_layer(size=6, precision=1.0)
    ...     .add_layer(size=4, precision=1.0)
    ... )
    >>>
    >>> # Or use add_layer_stack for multiple layers at once
    >>> net = (
    ...     DeepNetwork()
    ...     .add_nodes(kind="continuous-state", n_nodes=10)
    ...     .add_layer_stack(layer_sizes=[8, 6, 4], precision=1.0)
    ... )
    >>>
    >>> # Plot without specifying layer indices
    >>> net.plot_layers()
    >>>
    >>> # Fit with automatic input/output detection
    >>> net.fit(x_train, y_train)

    """

    def __init__(
        self,
        kind: str = "volatile-state",
        coupling_fn: tuple[Optional[Callable], ...] = (None,),
        weight_init: Optional[str] = None,
    ):
        """Initialize a DeepNetwork with layer tracking.

        Parameters
        ----------
        kind :
            The default node type for hidden layers.
        coupling_fn :
            Default coupling function(s) for hidden layers.
        weight_init :
            Weight initialisation strategy applied to hidden layers when
            :meth:`weight_initialisation` is called.  Must be one of
            ``"xavier"``, ``"he"``, ``"orthogonal"``, ``"sparse"``, or
            *None* (no automatic initialisation).

        """
        super().__init__()
        self.layers: list = []  # Track layer structure: list of lists of node indices
        self.kind = kind
        self.coupling_fn = coupling_fn
        self.weight_init = weight_init

    def add_layer(
        self,
        size: int = 1,
        kind: Optional[str] = None,
        value_children: Optional[Union[int, list[int], tuple[int, ...]]] = None,
        coupling_strengths: Union[float, list[float], tuple[float, ...]] = 0.0,
        coupling_fn: Optional[tuple[Optional[Callable], ...]] = None,
        **node_parameters,
    ) -> DeepNetwork:
        """Add a fully connected layer and track it.

        Each new parent connects to all nodes in the provided list of children,
        analogous to a dense layer in deep learning. By default, connects to all
        orphan nodes (nodes without value parents).

        Parameters
        ----------
        size :
            Number of parent nodes to create in this layer.
        kind :
            The type of nodes to add (e.g., "continuous-state", "volatile-state"). If
            None, defaults to the type of node declared in the class.
        value_children :
            Index or list of indices for the child nodes below this layer.
            If None, automatically connects to all orphan nodes in the network.
        coupling_strengths :
            Coupling strength(s) to the child nodes. Can be a single float
            (applied to all connections) or a list/tuple of floats.
        coupling_fn :
            Coupling function(s) between the new nodes and their value children.
        **node_parameters
            Additional keyword parameters for node configuration (e.g., precision,
            mean, tonic_volatility, etc...). These will be passed to add_nodes.

        Returns
        -------
        self :
            The updated network for method chaining.

        """
        # Track the number of nodes before adding
        n_nodes_before = self.n_nodes

        # define the type of node to use in the hidden layers
        if kind is None:
            kind = self.kind

        # define the coupling functions to use in the hidden layers
        if coupling_fn is None:
            coupling_fn = self.coupling_fn

        # Auto-detect orphan nodes if no children specified
        if value_children is None:
            children = []
            for idx in range(len(self.edges)):
                # A node is orphan if it has no value parents
                if self.edges[idx].value_parents is None:
                    children.append(idx)
            if not children:
                raise ValueError(
                    "No orphan nodes found. Please specify value_children explicitly."
                )
        else:
            # Normalize children to list
            if isinstance(value_children, int):
                children = [value_children]
            else:
                children = list(value_children)

        # Add node parameters that shouldn't be overridden
        node_params = {
            "coupling_fn": coupling_fn,
            **node_parameters,
        }

        # Add all parent nodes (one per unit in the new layer)
        for _ in range(size):
            self.add_nodes(
                kind=kind,
                value_children=(children, [coupling_strengths] * len(children)),
                **node_params,
            )

        # Record the new layer indices
        new_layer_idxs = list(range(n_nodes_before, self.n_nodes))
        self.layers.append(new_layer_idxs)

        return self

    def add_layer_stack(
        self,
        layer_sizes: list[int],
        kind: Optional[str] = None,
        value_children: Optional[Union[int, list[int], tuple[int, ...]]] = None,
        coupling_strengths: Union[float, list[float], tuple[float, ...]] = 1.0,
        coupling_fn: Optional[tuple[Optional[Callable], ...]] = None,
        **node_parameters,
    ) -> DeepNetwork:
        """Add multiple fully connected layers and track them.

        Builds multiple layers sequentially, with each layer automatically tracking
        its node indices.

        Parameters
        ----------
        layer_sizes :
            Number of parent nodes to create in each new layer.
        kind :
            The type of nodes to add (e.g., "continuous-state", "volatile-state"). If
            None, defaults to the type of node declared in the class.
        value_children :
            Index or list of indices for the bottom layer.
            If None, automatically connects to all orphan nodes for the first layer.
        coupling_strengths :
            Coupling strength(s) to apply across all layers.
        coupling_fn :
            Coupling function(s) to apply across all layers.
        **node_parameters
            Additional keyword parameters for node configuration.

        Returns
        -------
        self :
            The updated network for method chaining.

        """
        # define the type of node to use in the hidden layers
        if kind is None:
            kind = self.kind

        # define the coupling functions to use in the hidden layers
        if coupling_fn is None:
            coupling_fn = self.coupling_fn

        # Build layer by layer using the overridden add_layer
        # which automatically tracks each layer
        for i, size in enumerate(layer_sizes):
            if i == 0:
                self.add_layer(
                    size=size,
                    kind=kind,
                    value_children=value_children,
                    coupling_strengths=coupling_strengths,
                    coupling_fn=coupling_fn,
                    **node_parameters,
                )
            else:
                self.add_layer(
                    size=size,
                    value_children=None,  # Auto-detect orphans
                    coupling_strengths=coupling_strengths,
                    coupling_fn=coupling_fn,
                    **node_parameters,
                )

        return self

    def add_nodes(self, *args, **kwargs) -> DeepNetwork:
        """Add nodes and track as a layer if they form the base.

        This overrides Network.add_nodes to track base/input layers.
        """
        n_nodes_before = self.n_nodes

        # Call parent class method
        super().add_nodes(*args, **kwargs)

        # If this is creating multiple nodes at once, track as a layer
        n_nodes = kwargs.get("n_nodes", 1)
        if n_nodes > 1:
            new_layer_idxs = list(range(n_nodes_before, self.n_nodes))
            # Only add to layers if this is the first layer (base layer)
            if len(self.layers) == 0:
                self.layers.append(new_layer_idxs)

        return self

    def plot_deep_network(self, backend: str = "graphviz", **kwargs):
        """Plot the network in a deep-learning style (layer-by-layer).

        Parameters
        ----------
        backend :
            The plotting backend to use. Only 'graphviz' is supported.
        **kwargs
            Additional arguments passed to the plotting function.

        Returns
        -------
        The plot object (depends on backend).

        """
        if backend == "graphviz":
            from pyhgf.plots import graphviz

            return graphviz.plot_deep_network(deep_network=self, **kwargs)
        else:
            raise ValueError(
                "Invalid backend. Currently supports only backend='graphviz'."
            )

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        inputs_x_idxs: Optional[tuple[int, ...]] = None,
        inputs_y_idxs: Optional[tuple[int, ...]] = None,
        lr: Union[str, float] = 0.2,
        overwrite: bool = True,
    ):
        """Fit the deep network to predictors (X, top layer) and outcomes (Y, bottom).

        If inputs_x_idxs and inputs_y_idxs are not provided, this method automatically
        detects them based on the tracked layer structure:
        - inputs_x_idxs: the first (top) layer
        - inputs_y_idxs: the last (bottom) layer

        Parameters
        ----------
        x :
            Predictor values (input features).
        y :
            Target values (predictions/labels).
        inputs_x_idxs :
            Node indices receiving the predictors. If None, uses the top layer.
        inputs_y_idxs :
            Node indices receiving the predictions. If None, uses the bottom layer.
        lr :
            Learning rate for coupling strength updates. Either "dynamic" or a float.
            If a float is provided, the value will be used as learning rate.
        overwrite
            Whether to recreate the propagation function if it already exists.

        Returns
        -------
        self :
            The network with updated parameters.

        """
        # Auto-detect input/output layers if not specified
        if inputs_x_idxs is None:
            if not self.layers:
                raise ValueError(
                    "No layers tracked. Either provide inputs_x_idxs explicitly "
                    "or use add_layer() to build the network."
                )
            inputs_x_idxs = tuple(self.layers[-1])  # First layer

        if inputs_y_idxs is None:
            if not self.layers:
                raise ValueError(
                    "No layers tracked. Either provide inputs_y_idxs explicitly "
                    "or use add_layer() to build the network."
                )
            inputs_y_idxs = tuple(self.layers[0])  # Last layer

        # Call parent fit method
        return super().fit(
            x=x,
            y=y,
            inputs_x_idxs=inputs_x_idxs,
            inputs_y_idxs=inputs_y_idxs,
            lr=lr,
            overwrite=overwrite,
        )

    def weight_initialisation(
        self,
        strategy: Optional[str] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> DeepNetwork:
        """Initialise coupling weights for every layer except the input layer.

        For each targeted layer the method generates a weight matrix whose dimensions
        are ``(n_parents, n_current)`` — the number of parent nodes in the layer above
        and the number of nodes in the current layer, and applies the values via
        :func:`pyhgf.utils.set_coupling`.

        The input (top) layer is skipped because it has no parents to
        initialise.

        Parameters
        ----------
        strategy :
            Initialisation strategy.  One of ``"xavier"``, ``"he"``, ``"orthogonal"``,
            or ``"sparse"``.  If *None*, falls back to ``self.weight_init`` set during
            construction.
        seed :
            Optional random seed passed to the initialisation function.
        **kwargs
            Extra keyword arguments forwarded to the initialisation function (e.g.
            ``gain`` for orthogonal, ``sparsity`` / ``std`` for sparse).

        Returns
        -------
        self :
            The network with updated coupling weights (for method chaining).

        Raises
        ------
        ValueError
            If no strategy is provided (neither as argument nor at init time), or if the
            network has fewer than 2 tracked layers.

        """
        # If strategy is not provided here, use the default values from the constructor
        if strategy is None:
            return self
        elif strategy not in ["xavier", "he", "orthogonal", "sparse"]:
            raise ValueError(
                "Invalid weight initialisation strategy. Pass strategy='xavier' (or "
                "'he', 'orthogonal', 'sparse') in the DeepNetwork constructor."
            )

        _init_fns: dict[str, Callable[..., np.ndarray]] = {
            "xavier": xavier_init,
            "he": he_init,
            "orthogonal": orthogonal_init,
            "sparse": sparse_init,
        }
        init_fn: Callable[..., np.ndarray] = _init_fns[strategy]

        if len(self.layers) < 2:
            raise ValueError(
                "weight_initialisation requires at least 2 tracked layers. "
                f"The network currently has {len(self.layers)} layer(s)."
            )

        # All layers except the top/input layer (layers[-1]).
        # For each layer at index `layer_idx`, the parents live in the layer
        # above at `layer_idx + 1`.
        for layer_idx in range(0, len(self.layers) - 1):
            current_nodes = self.layers[layer_idx]
            parent_nodes = self.layers[layer_idx + 1]

            n_parents = len(parent_nodes)
            n_current = len(current_nodes)

            # Generate weight matrix (flat, row-major: parent × current)
            weights = init_fn(n_parents, n_current, seed=seed, **kwargs)

            # Apply weights to each parent→current-node connection
            for p_local, parent_idx in enumerate(parent_nodes):
                for c_local, child_idx in enumerate(current_nodes):
                    w = float(weights[p_local * n_current + c_local])
                    self.attributes = set_coupling(
                        attributes=self.attributes,
                        edges=self.edges,
                        parent_idx=parent_idx,
                        child_idx=child_idx,
                        coupling=w,
                    )

        return self

    def get_layer_sizes(self) -> list[int]:
        """Get the size of each tracked layer.

        Returns
        -------
        layer_sizes :
            List of integers representing the number of nodes in each layer.

        """
        return [len(layer) for layer in self.layers]

    def get_layer(self, layer_idx: int) -> list[int]:
        """Get the node indices for a specific layer.

        Parameters
        ----------
        layer_idx :
            The index of the layer (0 = first/bottom layer).

        Returns
        -------
        node_indices :
            List of node indices in that layer.

        """
        if layer_idx < 0 or layer_idx >= len(self.layers):
            raise IndexError(
                (
                    f"Layer index {layer_idx} out of range. "
                    "Network has {len(self.layers)} layers."
                )
            )
        return self.layers[layer_idx]

    def __repr__(self) -> str:
        """Print string representation of layer structure."""
        if not self.layers:
            return f"DeepNetwork(nodes={self.n_nodes}, layers=[])"

        layer_sizes = self.get_layer_sizes()
        return f"DeepNetwork(nodes={self.n_nodes}, layers={layer_sizes})"
