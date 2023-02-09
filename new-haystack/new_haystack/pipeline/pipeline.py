from typing import *

from pathlib import Path
import logging
from copy import deepcopy
import sys
import yaml
from collections import OrderedDict

import networkx as nx
from networkx.drawing.nx_agraph import to_agraph

from new_haystack.actions._utils import DEFAULT_EDGE_NAME
from new_haystack.pipeline._utils import (
    PipelineRuntimeError,
    PipelineError,
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
        self.stores: Dict[str, object] = {}
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

        #
        # NOTES on the Pipeline.run() algorithm
        #
        # Nodes are run as soon as an input for them appears in the inputs buffer.
        # When there's more than a node at once  in the buffer (which means some 
        # branches are running in parallel or that there are loops) they are selected to 
        # run in FIFO order by the `inputs_buffer` OrderedDict.
        #
        # Inputs are labeled with the name of the node they're aimed for, plus the name 
        # of the edge they're coming from: 
        # 
        #   ````
        #   inputs_buffer[target_node] = {
        #       'source_node.output_edge': <node's input from this edge>,  
        #       ... 
        #   }
        #   ```
        #
        # Nodes should wait until all the necessary input data has arrived before running. 
        # If they're popped from the input_buffer before they're ready, they're put back in.
        # If the pipeline has branches of different length, it's possible that a node
        # might have to wait a bit and "let other nodes pass" before having all the 
        # input data it needs.
        #
        # However, if the node in question is in a loop, some node in the "waiting for"
        # list might appear to be _downstream_: therefore it can't be waited for!
        #
        # So: nodes should wait only for all edges coming from **strictly upstream nodes**.
        # If there exists a path in the directed graph between the current node
        # and the expected input edge, do not wait for that input edge.
        #
        # Data access:
        # - Name of the node       # self.graph.nodes  (List[str])
        # - Action of the node     # self.graph.nodes[node]["action"]
        # - Input nodes            # [e[0] for e in self.graph.in_edges(node, data=True)]
        # - Weight of input nodes  # [e[2]["weight"] for e in self.graph.in_edges(node, data=True)]
        # - Output nodes           # [e[1] for e in self.graph.out_edges(node, data=True)]
        # - Output edges           # [e[2]["label"] for e in self.graph.out_edges(node, data=True)]

        logger.info("Pipeline execution started.")
        inputs_buffer: OrderedDict = OrderedDict()

        # Collect the nodes taking no input edges: these are the entry points.
        # They receive directly the pipeline inputs.
        node_names: List[str] = [node for node in self.graph.nodes if not self.graph.in_edges(node)]
        for node_name in node_names:
            inputs_buffer[node_name] = {"": {"data": data, "parameters": parameters, "weight": 1}}

        # Execution loop. We select the nodes to run by checking which keys are set in the
        # inputs buffer. If the key exists, the node might be ready to run.
        pipeline_results = {}
        while inputs_buffer:
            node_name, node_inputs = inputs_buffer.popitem(last=False)  # FIFO
                
            # *** LOOPS DETECTION ***
            # Let's first list all the edges the current node should be waiting for. As said above,
            # we should be wait on all edges, except for the downstream ones.
            nodes_to_wait_for = {
                f"{e[0]}.{e[2]['label']}"  # the first element of the edge (input node) with its relative label
                for e in self.graph.in_edges(node_name, data=True)  # for all input edges
                # if there's no path in the graph leading back from the current node to the input one
                if not nx.has_path(self.graph, node_name, e[0]) 
            }

            # For each of them, let's verify that all inputs that we should be waiting for
            # are actually there. We're going to check the "from" label of every block of
            # data contained in the node_inputs buffer
            # Note the exception: if the node has no input nodes, for sure it's ready to run.
            if self.graph.in_edges(node_name) and nodes_to_wait_for > node_inputs.keys():
                # We are missing some inputs. Let's put this node back in the queue
                # and go to the next node.
                inputs_buffer[node_name] = node_inputs
                continue
            
            # We have all the input data we need to proceed.
            # If there are multiple entries in the node_inputs buffer, let's merge them here.
            # Note that entries are sorted by weight because `merge()` lets the first parameter's
            # values dominate in case of conflict. See merge() docstrings for details.
            input_data: Dict[str, Any] = {}
            input_params: Dict[str, Any] = {}
            node_inputs = sorted(node_inputs.values(), key=lambda x: x["weight"], reverse=True)
            for node_input in node_inputs:
                input_data = merge(input_data, node_input["data"])
                input_params = merge(input_params, node_input["parameters"])

            # Check for default parameters and add them to the parameter's dictionary
            # Default parameters are the one passed with the `pipeline.add_node()` method
            # and have lower priority with respect to parameters passed through `pipeline.run()`.
            default_params = self.graph.nodes[node_name]["parameters"]
            if default_params:
                input_params[node_name] = {
                    **default_params,
                    **parameters.get(node_name, {}),
                }

            # List the output edges to pass to the action
            output_edges = {e[2]["label"] for e in self.graph.out_edges(node_name, data=True)} or {DEFAULT_EDGE_NAME}

            # *** PRUNING ***
            # If the input data is an empty dictionary, that means that the node has been pruned.
            # Note the difference: if there's no key for this node in the buffer, or not all the expected
            # input nodes are present in it, then the node needs to wait. Instead, if a key
            # is present for all expected input edges but all of them are empty, that means that some
            # decision node upstream has selected a branch that does not include this node, and therefore
            # this node should not run. It should be "skipped" without blocking any other node downstream that
            # might be waiting for their output.
            #
            # The typical case of pruning is a classifier. In indexing, classifiers might send a specific file 
            # to a specific converter and later, all converters would send the documents to a PreProcessor.
            # We want all converters that did not receive documents to:
            # - not run
            # - signal PreProcessor not to wait for them.
            #
            # This is called pruning.
            # 
            # TL:DR; Pruned nodes should:
            # - not run
            # - prune all their outgoing edges by propagating the empty input data
            node_results: Dict[str, Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]]
            if not input_data:
                # Prune the node and all the downstream ones
                node_results = {output_edge: ({}, {}) for output_edge in output_edges}
            else:
                # Call the node
                node_action = self.graph.nodes[node_name]["action"]
                try:
                    logger.info("* Running %s", node_name)
                    node_results = node_action(
                        name=node_name,
                        data=input_data,
                        parameters=input_params,
                        outgoing_edges=output_edges,
                        stores=self.stores,
                    )
                except Exception as e:
                    raise PipelineRuntimeError(
                        f"{node_name} raised '{e.__class__.__name__}: {e}' \n\ndata={input_data}\n\nparameters={input_params}\n\n"
                        "See the stacktrace above for more information."
                    ) from e

            if not self.graph.out_edges(node_name):
                # If there are no output edges, the output of this node is the output of the pipeline
                # Store it in pipeline_results
                # NOTE: we're assuming nodes can't output mode than once! If the assumption doesn't hold
                # anymore, fix this code by using merge()
                pipeline_results[node_name] = node_results[DEFAULT_EDGE_NAME][0]
            else:
                # Find out where the output will go: which nodes and along which edges
                # This data helps us find:
                #  - Where to store the data (output_node)
                #  - What to put in the "from" fiels (input_node.label)
                for edge_data in self.graph.out_edges(node_name, data=True):
                    source_node = edge_data[0]
                    source_edge = edge_data[2]['label']
                    target_node = edge_data[1]

                    # Corner case: pruning by passing an empty dict doesn't play well in loops.
                    # Such nodes must be removed from the input buffer completely.
                    if not source_edge in node_results.keys() and nx.has_path(self.graph, target_node, node_name):
                        continue
                    
                    # In all other cases, either populate the buffer or prune the downstream node by adding an empty dict
                    if not target_node in inputs_buffer:
                        inputs_buffer[target_node] = {}
                    if source_edge in node_results.keys():
                        inputs_buffer[target_node][f"{source_node}.{source_edge}"] = {
                            "data": node_results[source_edge][0], 
                            "parameters": node_results[source_edge][1], 
                            "weight": edge_data[2]['weight']
                        }
                    else:
                        inputs_buffer[target_node][f"{source_node}.{source_edge}"] = {
                            "data": {},
                            "parameters": {},
                            "weight": 0
                        }

        logger.info("Pipeline executed successfully.")

        # Simplify output for single output pipelines
        if len(pipeline_results.keys()) == 1:
            pipeline_results = pipeline_results[list(pipeline_results.keys())[0]]

        return pipeline_results
