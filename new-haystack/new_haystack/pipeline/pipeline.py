from typing import *

from pathlib import Path
import logging
from copy import deepcopy
import sys
import yaml
from collections import OrderedDict

import networkx as nx
from networkx.drawing.nx_agraph import to_agraph

from new_haystack.pipeline._utils import (
    PipelineRuntimeError,
    PipelineConnectError,
    PipelineError,
    NoSuchStoreError,
    find_actions,
    validate_graph as validate_graph,
    load_nodes,
    serialize,
)


logger = logging.getLogger(__name__)


class Pipeline:
    """
    Core loop of a Haystack application.
    """
    
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
            load_nodes(self.graph, self.available_actions)

            if validation:
                validate_graph(self.graph, self.available_actions)
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

    def save(self, path: Path) -> None:
        """
        Saves a pipeline to YAML.
        """
        _graph = deepcopy(self.graph)
        serialize(_graph)
        with open(path, "w") as f:
            # FIXME we should dump the actual serialized graph, not just its node link data
            yaml.dump(nx.node_link_data(_graph), f)
        logger.debug("Pipeline saved to %s.", path)

    def connect_store(self, name: str, store: object) -> None:
        self.stores[name] = store

    def list_stores(self) -> Iterable[str]:
        return self.stores.keys()

    def get_store(self, name: str) -> object:
        try:
            return self.stores[name]
        except KeyError as e:
            raise NoSuchStoreError(f"No store named '{name}' is connected to this pipeline.") from e

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
            raise ValueError(f"Node named '{name}' already exists: choose another name.")

        # Params must be a dict
        if parameters and not isinstance(parameters, dict):
            raise ValueError("'parameters' must be a dictionary.")

        # Add action to the graph, disconnected
        logger.debug("Adding node '%s' (%s)", name, action)
        self.graph.add_node(name, action=action, parameters=parameters)

    def connect(self, nodes: List[str]) -> None:
        """
        Connect nodes together. All nodes to connect must exist in the pipeline,
        while new edges are created on the fly.

        For example, `pipeline.connect(["node_1.output_1", "node_2", "node_3.output_2", "node_4"])`
        generates 3 edges across these nodes in the order they're given.

        If connecting to an node that has several output edges, specify its name with 'node_name.edge_name'.
        If the node will return outputs that needs to be merged. Note that iterators (lists, dictionaries, etc...)
        are merged instead of replaced, but in case of internal conflicts the same priority order applies.
        """
        # Connect in pairs
        for position in range(len(nodes) - 1):
            upstream_node_name = nodes[position]
            downstream_node_name = nodes[position + 1]

            # Find out the name of the edge
            edge_name = None
            # Edges may be named explicitly by passing 'node_name.edge_name' to connect(). 
            # Specify the edge name for the upstream node only.
            if "." in upstream_node_name:
                upstream_node_name, edge_name = upstream_node_name.split(".", maxsplit=1)
                upstream_node = self.graph.nodes[upstream_node_name]["action"]
            else:
                # If the edge had no explicit name and the upstream node has multiple outputs, raise an exception            
                upstream_node = self.graph.nodes[upstream_node_name]["action"]
                if len(upstream_node.expected_outputs) != 1:
                    raise PipelineConnectError(
                        f"Please specify which output of node '{upstream_node_name}' node "
                        f"'{downstream_node_name}' should connect to. Node '{upstream_node_name}' has the following "
                        f"outputs: {upstream_node.expected_outputs}"
                    )
                edge_name = upstream_node.expected_outputs[0]

            # Remove edge name from downstream_node name (it's needed only when the node is upstream)
            downstream_node_name = downstream_node_name.split(".", maxsplit=2)[0]
            downstream_node = self.graph.nodes[downstream_node_name]["action"]

            # All nodes names must be in the pipeline already
            if upstream_node_name not in self.graph.nodes:
                raise PipelineConnectError(f"'{upstream_node_name}' is not present in the pipeline.")
            if downstream_node_name not in self.graph.nodes:
                raise PipelineConnectError(f"'{downstream_node_name}' is not present in the pipeline.")

            # Check if the edge with that name already exists between those two nodes
            if any(
                edge[1] == downstream_node_name and edge[2]["label"] == edge_name
                for edge in self.graph.edges.data(nbunch=upstream_node_name)
            ):
                logger.info(
                    "An edge called '%s' connecting node '%s' and node '%s' already exists: skipping.",
                    edge_name,
                    upstream_node_name,
                    downstream_node_name,
                )
                return

            # Find all empty slots in the upstream and downstream nodes
            free_downstream_inputs = deepcopy(downstream_node.expected_inputs)
            for _, __, data in self.graph.in_edges(downstream_node_name, data=True):
                position = free_downstream_inputs.index(data["label"])
                free_downstream_inputs.pop(position)

            free_upstream_outputs = deepcopy(upstream_node.expected_outputs)
            for _, __, data in self.graph.out_edges(upstream_node_name, data=True):
                position = free_upstream_outputs.index(data["label"])
                free_upstream_outputs.pop(position)

            # Make sure the edge is connecting one free input to one free output
            if edge_name not in free_downstream_inputs or edge_name not in free_upstream_outputs:
                expected_inputs_string = "\n".join(
                    [" - " + edge[2]["label"] + f" (taken by {edge[0]})" for edge in self.graph.in_edges(downstream_node_name, data=True)] + \
                    [f" - {free_in_edge} (free)" for free_in_edge in free_downstream_inputs]
                )
                expected_outputs_string = "\n".join(
                    [" - " + edge[2]["label"] + f" (taken by {edge[1]})" for edge in self.graph.out_edges(upstream_node_name, data=True)] + \
                    [f" - {free_out_edge} (free)" for free_out_edge in free_upstream_outputs] 
                )
                raise PipelineConnectError(
                    f"Cannot connect '{upstream_node_name}' with '{downstream_node_name}' with an edge named '{edge_name}': "
                    f"their declared inputs and outputs do not match.\n"
                    f"Upstream node '{upstream_node_name}' declared these outputs:\n{expected_outputs_string}\n"
                    f"Downstream node '{downstream_node_name}' declared these inputs:\n{expected_inputs_string}\n"
                )
            # Create the edge
            logger.debug(
                "Connecting node '%s' to node '%s' along edge '%s'",
                upstream_node_name,
                downstream_node_name,
                edge_name,
            )
            self.graph.add_edge(
                upstream_node_name, downstream_node_name, label=edge_name
            )

    def get_node(self, name: str) -> Dict[str, Any]:
        """
        Returns all the data associated with a node.
        """
        candidates = [node for node in self.graph.nodes if node == name]
        if not candidates:
            raise ValueError(f"Node named {name} not found.")
        return self.graph.nodes[candidates[0]]

    def draw(self, path: Optional[Path] = None, graphviz: bool = True) -> None:
        """
        Draws the pipeline. If path is not given, shows an interactive plot
        in a matplotlib window.
        """

        if graphviz:
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

        else:
            try:
                import matplotlib.pyplot as plt
                from netgraph import InteractiveGraph
            except (ImportError, ModuleNotFoundError) as e:
                raise PipelineError("Failed to import some of the drawing libraries! Can't draw this pipeline.") from e

            plot_instance = InteractiveGraph(
                self.graph, 
                node_size=2, 
                node_layout="dot",
                node_color="#ffffff00",
                node_edge_color="#ffffff00",
                node_labels=True, 
                node_label_offset=0.0001,
                node_label_fontdict={"backgroundcolor": "lightgrey"},
                node_shape="o",
                edge_width=1.,
                edge_layout="curved",
                edge_label_rotate=False,
                edge_labels={(e[0], e[1]): e[2]["label"] for e in self.graph.edges(data=True)},            
                edge_label_fontdict={"fontstyle": "italic", "color": "grey", "backgroundcolor": "#ffffff00"},
                arrows=True, 
                ax=None,
                #scale=(10, 10)
            )

            if not path:
                plt.show()
            else:
                plt.savefig(path, bbox_inches='tight')
                logger.debug(f"Pipeline diagram saved at {path}")

    def run(
        self,
        data: Union[Dict[str, Any], List[Tuple[str, Any]]],
        parameters: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Runs the pipeline
        """
        #
        # Idea for the future
        #
        # Right now, pipelines allow for loops. Loops make sense if any of the involved nodes
        # is stateful, or if it loops over the same values in the pipeline context (like adding 1 
        # to a value until it passes over a threshold). However, if the work is stateless, we should
        # add the possibility to unwrap these loops and transforms them into a arbitrary number of replicas
        # of the same function. For example, loops consuming a queue would be spread over N nodes, one for each
        # item of the queue.
        #
        if not parameters:
            parameters = {}

        # Validate the parameters
        if any(node not in self.graph.nodes for node in parameters.keys()):
            logging.warning(
                "You passed parameters for one or more node(s) that do not exist in the pipeline: %s",
                [node for node in parameters.keys() if node not in self.graph.nodes],
            )
        #
        # NOTES on the Pipeline.run() algorithm
        #
        # Nodes are run as soon as an input for them appears in the inputs buffer.
        # When there's more than a node at once  in the buffer (which means some 
        # branches are running in parallel or that there are loops) they are selected to 
        # run in FIFO order by the `inputs_buffer` OrderedDict.
        #
        # Inputs are labeled with the name of the node they're aimed for: 
        # 
        #   ````
        #   inputs_buffer[target_node] = [(input_edge, input_value), ...]
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
        # - Output nodes           # [e[1] for e in self.graph.out_edges(node, data=True)]
        # - Output edges           # [e[2]["label"] for e in self.graph.out_edges(node, data=True)]

        logger.info("Pipeline execution started.")
        inputs_buffer: OrderedDict = OrderedDict()

        # Collect the nodes taking no input edges: these are the entry points.
        # They receive directly the pipeline inputs.
        node_names: List[str] = [node for node in self.graph.nodes if not self.graph.in_edges(node)]
        for node_name in node_names:
            # NOTE: We allow users to pass dictionaries just for convenience.
            # The real input format is List[Tuple[str, Any]], to allow several input edges to have the same name.
            if isinstance(data, dict):
                data = [(key, value) for key, value in data.items()]
            inputs_buffer[node_name] = {"data": data, "parameters": parameters}

        # Execution loop. We select the nodes to run by checking which keys are set in the
        # inputs buffer. If the key exists, the node might be ready to run.
        pipeline_results = {}
        while inputs_buffer:
            node_name, node_inputs = inputs_buffer.popitem(last=False)  # FIFO
                
            # *** LOOPS DETECTION ***
            # Let's first list all the inputs the current node should be waiting for. As said above,
            # we should be wait on all edges except for the downstream ones.
            inputs_to_wait_for = [
                e[2]['label']  # the first element of the edge (input node) with its relative label
                for e in self.graph.in_edges(node_name, data=True)  # for all input edges
                # if there's no path in the graph leading back from the current node to the input one
                if not nx.has_path(self.graph, node_name, e[0]) 
            ]

            # Let's verify that all inputs edges that we should be waiting for are in the graph. 
            # Note the exception: if the node has no input nodes, for sure it's ready to run.
            input_names_received = [i[0] for i in node_inputs["data"]]
            if self.graph.in_edges(node_name) and sorted(inputs_to_wait_for) != sorted(input_names_received):
                # We are missing some inputs. Let's put this node back in the queue
                # and go to the next node.
                logger.debug(
                    "Skipping '%s', some inputs are missing (inputs to wait for: %s, inputs_received: %s)", 
                    node_name, 
                    inputs_to_wait_for, 
                    input_names_received
                )
                inputs_buffer[node_name] = node_inputs
                continue
            
            # Check for default parameters and add them to the parameter's dictionary
            # Default parameters are the one passed with the `pipeline.add_node()` method
            # and have lower priority with respect to parameters passed through `pipeline.run()`
            # Or the modifications made by other nodes along the pipeline.
            if self.graph.nodes[node_name]["parameters"]:
                node_inputs["parameters"][node_name] = {
                    **(self.graph.nodes[node_name]["parameters"] or {}),
                    **parameters.get(node_name, {}),
                }
            
            node_results: Tuple[Dict[str, Any], Optional[Dict[str, Dict[str, Any]]]]
            
            # Call the node
            node_action = self.graph.nodes[node_name]["action"]
            try:
                logger.info("* Running %s", node_name)
                logger.debug("   '%s' inputs: %s", node_name, node_inputs)
                node_results = node_action.run(
                    name=node_name,
                    data=node_inputs["data"],
                    parameters=node_inputs["parameters"],
                    stores=self.stores,
                )
                logger.debug("   '%s' outputs: %s\n", node_name, node_results)
            except Exception as e:
                raise PipelineRuntimeError(
                    f"{node_name} raised '{e.__class__.__name__}: {e}' \ninputs={node_inputs['data']}\nparameters={node_inputs.get('parameters', None)}\n\n"
                    "See the stacktrace above for more information."
                ) from e

            if not self.graph.out_edges(node_name):
                # If there are no output edges, the output of this node is the output of the pipeline
                # Store it in pipeline_results
                # We use append() to account for the case in which a node outputs several times 
                # (for example, it can happen if there's a loop upstream). The list gets unwrapped before
                # returning it if there's only one output.
                if not node_name in pipeline_results.keys():
                    pipeline_results[node_name] = []
                pipeline_results[node_name].append(node_results[0])
            else:
                # Find out where the output will go: which nodes and along which edges
                for edge_data in self.graph.out_edges(node_name, data=True):
                    source_edge = edge_data[2]['label']
                    target_node = edge_data[1]

                    # Corner case: skipping by returining an empty dict doesn't play well in loops.
                    # Such nodes must be removed from the input buffer completely.
                    if not source_edge in node_results[0].keys() and nx.has_path(self.graph, target_node, node_name):
                        continue
                    
                    # In all other cases, populate the inputs buffer
                    if not target_node in inputs_buffer:
                        inputs_buffer[target_node] = {"data": []}  # Create the buffer for the downstream node if it's not there yet
                    if source_edge in node_results[0].keys():
                        inputs_buffer[target_node]["data"].append((source_edge, node_results[0][source_edge]))
                    inputs_buffer[target_node]["parameters"] = node_results[1] if len(node_results) == 2 else node_inputs["parameters"]

        logger.info("Pipeline executed successfully.")

        # Simplify output for single edge, single output pipelines
        if len(pipeline_results.keys()) == 1:
            pipeline_results = pipeline_results[list(pipeline_results.keys())[0]]
            if len(pipeline_results) == 1:
                pipeline_results = pipeline_results[0]

        return pipeline_results
