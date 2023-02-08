from typing import *

from pathlib import Path
import logging
from copy import deepcopy
import sys
import yaml
import json

import networkx as nx
from networkx.drawing.nx_agraph import to_agraph

from new_haystack.actions._utils import DEFAULT_EDGE_NAME
from new_haystack.pipeline._utils import (
    PipelineRuntimeError,
    merge,
    find_actions,
    validate as validate,
    is_warm,
    is_cold,
    warm_up,
    cool_down,
)


logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(
        self,
        path: Optional[Path] = None,
        search_actions_in: Optional[List[str]] = None,
        extra_actions: Optional[Dict[str, Callable[..., Any]]] = None,
        validation: bool = True,
    ):
        """
        Loads the pipeline from `path`, or creates an empty pipeline if no path is given.

        Searches for actions into the modules listed by `search_actions_in`. To narrow down the scope of the action search,
        set `search_actions_in=[<only the modules I want to look into for actions>]`.
        """
        self.stores = {}
        self.extra_actions = extra_actions or {}
        self.available_actions = extra_actions or {}
        self.search_actions_in = search_actions_in
        
        self.graph: nx.DiGraph
        if not path:
            logger.debug("Loading an empty pipeline")
            self.graph = nx.DiGraph()
        else:
            logger.debug("Loading pipeline from %s...", path)
            with open(path, "r") as f:
                self.graph = nx.node_link_graph(yaml.safe_load(f))
            logger.debug(
                "Pipeline edge list:\n - %s",
                "\n - ".join([str(edge) for edge in nx.to_edgelist(self.graph)]),
            )
            if validation:
                validate(self.graph, self.available_actions)
            else:
                logger.info("Skipping pipeline validation.")

    @property
    def search_actions_in(self):
        return self._search_actions_in

    @search_actions_in.setter
    def search_actions_in(self, search_modules):
        """
        Assigning a value to this attribute triggers a `find_actions` call,
        which will import any path in the `self._search_actions_in` attribute,
        and overwrite the content of `self.available_actions`.
        """
        self._search_actions_in = (
            search_modules
            if search_modules is not None
            else list(sys.modules.keys())
        )
        self.available_actions = {**find_actions(self._search_actions_in), **self.extra_actions}

    def validate(self):
        """
        Shorthand to call validate() on this pipeline.
        """
        validate(self.graph, self.available_actions)

    def warm_up(self):
        """
        Shorthand to call warm_up() on this pipeline.
        """
        warm_up(self.graph, self.available_actions)

    def cool_down(self):
        """
        Shorthand to call cool_down() on this pipeline.
        """
        cool_down(self.graph)

    def save(self, path: Path) -> None:
        """
        Saves a pipeline to YAML.
        """
        _graph = deepcopy(self.graph)
        if is_cold(self.graph):
            logger.debug("Pipeline is cold: no need to serialize actions.")
        else:
            cool_down(_graph)

        with open(path, "w") as f:
            yaml.dump(nx.node_link_data(_graph), f)
        logger.debug("Pipeline saved to %s.", path)

    def connect_store(self, name: str, store: Any) -> None:
        self.stores[name] = store

    def list_stores(self) -> Iterable[str]:
        return self.stores.keys()

    def disconnect_store(self, name: str) -> None:
        del self.stores[name]

    def add_node(
        self,
        name: str,
        action: Callable[..., Any],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Create a node for the given action. Nodes are not connected to anything by default:
        use `Pipeline.connect()` to connect nodes together.

        Node names must be unique, but actions can be reused from other nodes.
        """
        # Action names are unique
        if name in self.graph.nodes:
            raise ValueError(f"Node named {name} already exists: choose another name.")

        # Params must be a dict
        if parameters and not isinstance(parameters, dict):
            raise ValueError("'parameters' must be a dictionary.")

        # Add action to the graph, disconnected
        logger.debug("Adding node %s (%s)", name, action)
        self.graph.add_node(name, action=action, parameters=parameters)

    def connect(self, nodes: List[str], weights: Optional[List[int]] = None) -> None:
        """
        Connect nodes together. All nodes to connect must exist in the pipeline,
        while new edges are created on the fly.

        For example, `pipeline.connect(["node_1.output_1", "node_2", "node_3.output_2", "node_4"])`
        generates 3 edges across these nodes in the order they're given.

        If connecting to an node that has several output edges, specify its name with 'node_name.edge_name'.
        If the node will return outputs that needs to be merged, you can speficy the priority of each edge
        in case of conflicts by assigning a weight to each edge. Note that iterators (lists, dictionaries, etc...)
        are merged instead of replaced, but in case of internal conflicts the same priority order applies.
        """
        # Check weights
        if not weights:
            weights = [1] * len(nodes)
        elif len(weights) != len(nodes) - 1:
            raise ValueError(
                "You must give as many weights as the edges to create, or no weights at all."
            )

        # Connect in pairs
        for position in range(len(nodes) - 1):
            input_node = nodes[position]
            output_node = nodes[position + 1]
            weight = weights[position]

            # Find out if the edge is named
            if "." in input_node:
                input_node, edge_name = input_node.split(".", maxsplit=1)
            else:
                edge_name = DEFAULT_EDGE_NAME

            # Remove edge name from output_node
            output_node = output_node.split(".", maxsplit=2)[0]

            # All nodes names must be in the pipeline already
            if input_node not in self.graph.nodes:
                raise ValueError(f"{input_node} is not present in the pipeline.")
            if output_node not in self.graph.nodes:
                raise ValueError(f"{output_node} is not present in the pipeline.")

            # Check if the edge already exists
            if any(
                edge[1] == output_node
                for edge in self.graph.edges.data(nbunch=input_node)
            ):
                logger.debug(
                    "An edge connecting node %s and node %s already exists: skipping.",
                    input_node,
                    output_node,
                )

            else:
                # Create the edge
                logger.debug(
                    "Connecting node %s to node %s along edge %s (weight: %s)",
                    input_node,
                    output_node,
                    edge_name,
                    weight,
                )
                self.graph.add_edge(
                    input_node, output_node, label=edge_name, weight=weight
                )

    def get_node(self, name: str) -> Dict[str, Any]:
        """
        Returns all the data associated with a node.
        """
        candidates = [node for node in self.graph.nodes if node == name]
        if not candidates:
            raise ValueError(f"Node named {name} not found.")
        return self.graph.nodes[candidates[0]]

    # TODO later
    def concatenate(self, pipeline, input_edge, output_edge):
        # Watch out, Pipelines might have N input actions and M output actions!
        # One might need to specify to-from which edge to concatenate!
        pass

    def draw(self, path: Path) -> None:
        try:
            import pygraphviz
        except ImportError:
            raise ImportError(
                "Could not import `pygraphviz`. Please install via: \n"
                "pip install pygraphviz\n"
                "(You might need to run this first: apt install libgraphviz-dev graphviz )"
            )
        graphviz = to_agraph(self.graph)
        graphviz.layout("dot")
        graphviz.draw(path)
        logger.debug(f"Pipeline diagram saved at {path}")

    def run(
        self,
        data: Dict[str, Any],
        parameters: Optional[Dict[str, Dict[str, Any]]] = None,
        with_debug_info: bool = False,
    ) -> Dict[str, Any]:
        """
        Runs the pipeline

        """
        if not parameters:
            parameters = {}

        # Make sure the pipeline is valid
        self.validate()

        # Validate the parameters
        if any(node not in self.graph.nodes for node in parameters.keys()):
            logging.warning(
                "You passed parameters for one or more node(s) that do not exist in the pipeline: %s",
                [node for node in parameters.keys() if node not in self.graph.nodes],
            )

        # Warm up the pipeline if necessary
        if not is_warm(self.graph):
            logger.info(
                "Pipeline hasn't been warmed up before calling run(): warming it up now. This operation can take some time."
            )
            warm_up(graph=self.graph, available_actions=self.available_actions)

        # Prepare the actions input buffers
        inputs_buffer = {node_name: [] for node_name in self.graph.nodes}

        # Locate merge nodes (nodes with more than one input edge) and store their input edges info
        # Merge nodes must wait for all its inputs to be ready before running.
        merge_nodes = {
            node: self.graph.in_edges(node, data=True) for node in self.graph.nodes
        }
        merge_nodes = {
            node: {in_edge[0] for in_edge in inputs}
            for node, inputs in merge_nodes.items()
            if len(inputs) > 1
        }

        # Collect the nodes taking no input edges and pass them the pipeline input data
        node_names = [
            node
            for node in self.graph.nodes
            if not any(edge[1] == node for edge in self.graph.edges.data())
        ]
        for node_name in node_names:
            inputs_buffer[node_name] = [
                {"data": data, "parameters": parameters, "weight": 1}
            ]

        # Execution loop
        output_data = {}
        logger.info("Pipeline execution started.")
        while node_names:
            node_name, node_names = node_names[0], node_names[1:]

            # Make sure all expected input nodes have run. Sometimes with branched pipelines where a
            # branch is longer than the other, the merging node might be called before the long
            # branch had time to run. In this case we skip the merge node.
            # NOTE: the merge node can be removed from node_names, because it will be added again
            # then the longer branch's last node is called.
            if node_name in merge_nodes.keys() and merge_nodes[node_name] != {
                buffer["from"] for buffer in inputs_buffer[node_name]
            }:
                logging.debug(
                    "Skipping %s, not all input nodes have run yet.", node_name
                )
                continue

            # Read the inputs in the buffer and merge them into a single dict if necessary
            # ASSUMPTION: input values are already sorted by weight!
            input_data = {}
            input_params = {}
            for node_input_values in inputs_buffer[node_name]:
                input_data = merge(input_data, node_input_values["data"])
                input_params = merge(input_params, node_input_values["parameters"])

            # Find out where the output will go
            outgoing_data = self.graph.edges.data(nbunch=[node_name])
            if outgoing_data:
                # Regular node with output edges
                outgoing_nodes, outgoing_edges = zip(
                    *[(data[1], data[2]) for data in outgoing_data]
                )
                outgoing_edges_names = [data["label"] for data in outgoing_edges]
            else:
                # Terminal node - add a default edge, so the node itself doesn't need to know it's terminal
                outgoing_nodes, outgoing_edges, outgoing_edges_names = (
                    [],
                    [],
                    [DEFAULT_EDGE_NAME],
                )

            # Get the node's callable
            node_action = self.graph.nodes[node_name]["action"]

            # Check for default parameters and add them (lowest priority)
            default_params = self.graph.nodes[node_name]["parameters"]
            if default_params:
                input_params[node_name] = {
                    **default_params,
                    **input_params.get(node_name, {}),
                }

            # Call the node
            try:
                logger.info("* Running %s", node_name)
                out_dict: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]]
                out_dict = node_action(
                    name=node_name,
                    data=input_data,
                    parameters=input_params,
                    outgoing_edges=outgoing_edges_names,
                    stores=self.stores,
                )
            except Exception as e:
                logger.debug(
                    "%s failed! See the error below. Here is the current input buffer:\n%s",
                    node_name,
                    json.dumps(inputs_buffer, indent=4, default=str),
                )
                raise PipelineRuntimeError(
                    f"{node_name} raised '{e.__class__.__name__}: {e}' \n\ndata={input_data}\n\nparameters={input_params}\n\noutgoing_edges={outgoing_edges}\n\n"
                    "See the stacktrace above for more information."
                ) from e

            # Store the output
            if outgoing_nodes:

                # Store in the buffer to be used by following nodes
                for outgoing_node, outgoing_edge in zip(outgoing_nodes, outgoing_edges):
                    node_data, node_params = out_dict.get(
                        outgoing_edge["label"], ({}, {})
                    )
                    inputs_buffer[outgoing_node].append(
                        {
                            "data": node_data,
                            "parameters": node_params,
                            "weight": outgoing_edge["weight"],
                            "from": node_name,
                        }
                    )
                    inputs_buffer[outgoing_node].sort(
                        key=lambda x: x["weight"]
                    )  # Sort the outputs by weight

                # Add the outgoing nodes into the list of nodes to run.
                # Take care of avoiding duplicates (happens with nodes with multiple inputs)
                for outgoing_node in outgoing_nodes:
                    if not outgoing_node in node_names:
                        node_names.append(outgoing_node)

            else:
                # Store in the output dict to be returned by the pipeline
                node_data, node_params = out_dict.get(DEFAULT_EDGE_NAME, ({}, {}))
                output_data[node_name] = node_data

        logger.info("Pipeline executed successfully.")

        # Simplify output for single output pipelines
        if len(output_data.keys()) == 1:
            output_data = output_data[list(output_data.keys())[0]]

        if with_debug_info:
            return {"output": output_data, "_debug": inputs_buffer}

        return output_data
