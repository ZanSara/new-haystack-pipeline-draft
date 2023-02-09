from typing import Dict, Any, Callable, List, Iterable, Union, Set

from itertools import chain
import sys
import json
import logging
from inspect import getmembers, isclass, isfunction

import networkx as nx

from new_haystack.actions._utils import ActionValidationError


logger = logging.getLogger(__name__)


class PipelineError(Exception):
    pass


class PipelineRuntimeError(Exception):
    pass


class PipelineValidationError(PipelineError):
    pass


class PipelineSerializationError(PipelineError):
    pass


class PipelineDeserializationError(PipelineError):
    pass


def find_actions(modules_to_search: List[str]) -> Dict[str, Callable[..., Any]]:
    """
    Finds all functions decorated with `haystack_action` or derivatives (like `haystack_simple_action`)
    in all the modules listed in `modules_to_search`.

    WARNING: will attempt to import any module listed for search.

    Returns a dictionary with the action name and the action itself.
    """
    # # no-op if the module was already imported
    # for module in modules_to_search:
    #     import_module(module)

    actions = {}
    for search_module in modules_to_search:
        logger.debug("Searching for Haystack actions under %s...", search_module)

        duplicate_names = []
        for _, entity in getmembers(
            sys.modules[search_module], lambda x: isfunction(x) or isclass(x)
        ):
            # It's a Haystack action
            if hasattr(entity, "__haystack_action__"):

                # Two actions were discovered with the same name - namespace them
                if entity.__haystack_action__ in actions:
                    other_entity = actions[entity.__haystack_action__]
                    other_source_module = other_entity.__module__
                    logger.info(
                        "An action with the same name was found in two separate modules!\n"
                        " - Action name: %s\n - Found in modules: '%s' and '%s'\n"
                        "They both are going to be loaded, but you will need to use a namespace "
                        "path (%s.%s and %s.%s respectively) to use them in your Pipeline YAML definitions.",
                        entity.__haystack_action__,
                        other_source_module,
                        search_module,

                        other_source_module,
                        entity.__haystack_action__,
                        search_module,
                        entity.__haystack_action__,
                    )
                    duplicate_names.append(entity.__haystack_action__)

                    # Add both actions as namespaced
                    actions[f"{other_source_module}.{entity.__haystack_action__}"] = other_entity
                    actions[f"{search_module}.{entity.__haystack_action__}"] = entity
                    # Do not remove the non-namespaced one, so in the case of a third collision it geta detected properly

                actions[entity.__haystack_action__] = entity
                logger.debug(" * Found action: %s", entity)

    # Now delete all remaining duplicates
    for duplicate in duplicate_names:
        del actions[duplicate]

    return actions


def merge(first_dict: Dict[str, Any], second_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merges two dictionaries. In case of collision, the first dictionary values dominate.

    If the collision happens on a dictionary, the contained dictionaries are
    merged recursively.

    If the collision happens on another Iterable, the two
    Iterables are chained. Should work for lists, sets, and most other iterables.
    Iterables will be merged only if they're of the same type.
    """
    # Merge the dictionaries
    merge_dictionary = {**second_dict, **first_dict}

    # Check for dicts and iterables
    for key in merge_dictionary.keys():
        # if both have this key and it contains a dict, merge recursively
        if isinstance(first_dict.get(key, None), dict) and isinstance(
            second_dict.get(key, None), dict
        ):
            merge_dictionary[key] = merge(first_dict[key], second_dict[key])

        # if both have this key and it contains the same type of iterable, merge them
        elif (
            isinstance(first_dict.get(key, None), Iterable)
            and not isinstance(first_dict.get(key, None), str)
            and isinstance(second_dict.get(key, None), Iterable)
            and type(first_dict[key]) == type(second_dict[key])
        ):
            type_of_iterable = type(
                first_dict[key]
            )  # Find out if it's a list, a set, a tuple, ...
            merged_iterable = chain.from_iterable(
                [first_dict[key], second_dict[key]]
            )  # Returns an iterable
            merge_dictionary[key] = type_of_iterable(
                merged_iterable
            )  # Cast it to the same type as the merging dicts's content

    return merge_dictionary


def validate(
    graph: nx.DiGraph, available_actions: Dict[str, Dict[str, Union[str, Callable[..., Any]]]]
) -> None:
    """
    Makes sure the pipeline can run. Useful especially for pipelines loaded from file.

    NOTE: Does NOT warm up the pipeline if it's cold.
    """
    # Check that there are no isolated nodes or groups of nodes
    if not nx.is_weakly_connected(graph):
        raise PipelineValidationError(
            "The graph is not fully connected. Make sure all the nodes are connected to the same graph. "
            "You can use Pipeline.draw() to visualize the graph, or inspect the Pipeline.graph object."
        )

    # Check that the graph has starting nodes (nodes that take no input edges)
    # FIXME This just needs very good documentation imho.
    input_nodes = [
        node
        for node in graph.nodes
        if not any(edge[1] == node for edge in graph.edges.data())
    ]
    if not input_nodes:
        raise PipelineValidationError(
            "This pipeline doesn't seem to have starting nodes. "
            "Haystack checks for nodes that receive no input as starting nodes. "
            "If the first node of your pipeline is involved in a loop, please add a small no-op node "
            "in front of it to point Haystack towards the correct start of your graph."
        )

    for node in graph.nodes:
        action = graph.nodes[node]["action"]

        # Check that all actions in the graph are actually registered actions
        registered_function = (
            isfunction(action) and action in available_actions.values()
        )
        registered_class = type(action) in available_actions.values()
        name_of_registered_action = (
            isinstance(action, str) and action in available_actions.keys()
        )
        if not (registered_function or registered_class or name_of_registered_action):
            raise PipelineValidationError(
                f"Action {action} not found. Are you sure it is a Haystack action?"
            )

        # Class Actions might implement a cls.validate() method to customize validation
        if isinstance(action, str) and isclass(available_actions[action]):
            # Cold node
            # Remember that this type of validation does not occurr on warm nodes 
            # because they're already instantiated (so they "must" be valid I assume)
            # Correct the code if this assumption stops to hold
            if (
                hasattr(available_actions[action], "validate")
                and "init" in graph.nodes[node].keys()
            ):
                if isinstance(graph.nodes[node]["init"], str):
                    try:
                        parameters = json.loads(graph.nodes[node]["init"])
                    except Exception as e:
                        raise ActionValidationError(
                            f"Can't deserialize the init parameters for action {node}"
                        )
                else:
                    parameters = graph.nodes[node]["init"]
                available_actions[action].validate(init_parameters=parameters)

    logger.debug("Pipeline is valid")


def is_cold(graph: nx.DiGraph) -> bool:
    """
    Checks if all nodes in the graph are "cold", i.e. they're strings, ready for serialization
    """
    return all(isinstance(graph.nodes[node]["action"], str) for node in graph.nodes)


def is_warm(graph: nx.DiGraph) -> bool:
    """
    Checks if all nodes in the graph are "warm", i.e. they're strings, ready tu run
    """
    return all(
        isinstance(graph.nodes[node]["action"], Callable) for node in graph.nodes
    )


def warm_up(
    graph: nx.DiGraph, available_actions: Dict[str, Dict[str,Callable[..., Any]]]
) -> None:
    """
    Prepares the pipeline for the first execution. Instantiates all
    class nodes present in the pipeline, if they're not instantiated yet.
    """
    # Convert action names into actions and deserialize parameters
    for name in graph.nodes:
        try:
            if isinstance(graph.nodes[name]["action"], str):
                graph.nodes[name]["action"] = available_actions[
                    graph.nodes[name]["action"]
                ]
                # If it's a class, check if it's reusable or needs instantiation
                if isclass(graph.nodes[name]["action"]):
                    if "instance_id" in graph.nodes[name].keys():
                        # ReusableL fish it out from the graph
                        graph.nodes[name]["action"] = graph.nodes[
                            graph.nodes[name]["instance_id"]
                        ]["action"]
                    else:
                        # New: instantiate it
                        graph.nodes[name]["action"] = graph.nodes[name]["action"](
                            **graph.nodes[name]["init"] or {}
                        )
        except Exception as e:
            raise PipelineDeserializationError(
                "Couldn't deserialize this action: " + name
            ) from e

        try:
            if isinstance(graph.nodes[name]["parameters"], str):
                graph.nodes[name]["parameters"] = json.loads(
                    graph.nodes[name]["parameters"]
                )
        except Exception as e:
            raise PipelineDeserializationError(
                "Couldn't deserialize this action's parameters: " + name
            ) from e


def cool_down(graph: nx.DiGraph()) -> None:
    """
    Serializes all the nodes into a state that can be dumped to JSON or YAML.
    """
    reused_instances = {}
    for name in graph.nodes:
        # If the action is a reused instance, let's add the instance ID to the meta
        if graph.nodes[name]["action"] in reused_instances.values():
            graph.nodes[name]["instance_id"] = [
                key
                for key, value in reused_instances.items()
                if value == graph.nodes[name]["action"]
            ][0]

        elif hasattr(graph.nodes[name]["action"], "init_parameters"):
            # Class nodes need to have a self.init_parameters attribute (or property)
            # if they want their init params to be serialized.
            try:
                graph.nodes[name]["init"] = graph.nodes[name]["action"].init_parameters
            except Exception as e:
                raise PipelineSerializationError(
                    f"A class Action failed to provide its init parameters: {name}\n"
                    "If this is an action you wrote, you should save your init parameters into an instance "
                    "attribute called 'self.init_parameters' for this check to pass. Consider adding this "
                    "step into your class' '__init__' method."
                ) from e

            # This is a new action instance, so let's store it
            reused_instances[name] = graph.nodes[name]["action"]

        # Serialize the callable by name
        try:
            graph.nodes[name]["action"] = graph.nodes[name][
                "action"
            ].__haystack_action__
        except Exception as e:
            raise PipelineSerializationError(f"Couldn't serialize this action: {name}")

        # Serialize its default parameters with JSON
        try:
            if graph.nodes[name]["parameters"]:
                graph.nodes[name]["parameters"] = json.dumps(
                    graph.nodes[name]["parameters"]
                )
        except Exception as e:
            raise PipelineSerializationError(
                f"Couldn't serialize this action's parameters: {name}"
            )


def prune_branch(pruned_node: str, merge_nodes: Dict[str, Set[str]], graph: nx.DiGraph):

    outgoing_data = graph.edges.data(nbunch=[pruned_node])
    if outgoing_data:
        outgoing_nodes = [data[1] for data in outgoing_data]
        
        for node in outgoing_nodes:
            # Check if this node receives input on multiple edges.
            # If it receives input from many edges, prune only the edge coming from this node.
            if node in merge_nodes.keys():
                merge_nodes[node] = {edge for edge in merge_nodes[node] if edge != pruned_node}

                if merge_nodes[node]:
                    # the node still has incoming edges: don't prune, stop here
                    return merge_nodes

            merge_nodes = prune_branch(node, merge_nodes, graph)
    
    return merge_nodes