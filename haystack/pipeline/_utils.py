from typing import Dict, Any, Callable, List, Iterable

from itertools import chain
import sys
import json
import logging
from inspect import getmembers, isclass, isfunction
from collections import namedtuple
from importlib import import_module

import networkx as nx


logger = logging.getLogger(__name__)


DEFAULT_EDGE_NAME = "all"


class PipelineError(Exception):
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
    # no-op if the module was already imported
    for module in modules_to_search:
        import_module(module)

    actions = {}
    for search_module in modules_to_search:
        logger.debug("Searching for Haystack actions under %s...", search_module)
        for _, entity in getmembers(sys.modules[search_module], lambda x: isfunction(x) or isclass(x)):
            try:
                if hasattr(entity, "__haystack_action__"):
                    logger.debug(" * Found action: %s", entity)
                    actions[entity.__haystack_action__] = entity
            except AttributeError as e:
                pass
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
        if isinstance(first_dict.get(key, None), dict) and isinstance(second_dict.get(key, None), dict):
            merge_dictionary[key] = merge(first_dict[key], second_dict[key])

        # if both have this key and it contains the same type of iterable, merge them
        elif (
            isinstance(first_dict.get(key, None), Iterable)
            and isinstance(second_dict.get(key, None), Iterable)
            and type(first_dict[key]) == type(second_dict[key])
        ):
            type_of_iterable = type(first_dict[key])  # Find out if it's a list, a set, a tuple, ...
            merged_iterable = chain.from_iterable([first_dict[key], second_dict[key]])  # Returns an iterable
            merge_dictionary[key] = type_of_iterable(
                merged_iterable
            )  # Cast it to the same type as the merging dicts's content

    return merge_dictionary


def is_cold(graph: nx.DiGraph) -> bool:
    """
    Checks if all nodes in the graph are "cold", i.e. they're strings, ready for serialization
    """
    return all(isinstance(graph.nodes[node]["action"], str) for node in graph.nodes)


def is_warm(graph: nx.DiGraph) -> bool:
    """
    Checks if all nodes in the graph are "warm", i.e. they're strings, ready tu run
    """
    return all(isinstance(graph.nodes[node]["action"], Callable) for node in graph.nodes)


def warm_up(graph: nx.DiGraph, available_actions: Dict[str, Callable[..., Any]]) -> None:
    """
    Prepares the pipeline for the first execution. Instantiates all
    class nodes present in the pipeline, if they're not instantiated yet.
    """
    # Convert action names into actions and deserialize parameters
    for name in graph.nodes:
        try:
            if isinstance(graph.nodes[name]["action"], str):
                graph.nodes[name]["action"] = available_actions[graph.nodes[name]["action"]]

                # If it's a class, check if it's reusable or needs instantiation
                if isclass(graph.nodes[name]["action"]):
                    if "instance_id" in graph.nodes[name].keys():
                        # ReusableL fish it out from the pool
                        graph.nodes[name]["action"] = graph.nodes[graph.nodes[name]["instance_id"]]["action"]
                    else:
                        # New: instantiate it
                        graph.nodes[name]["action"] = graph.nodes[name]["action"](**graph.nodes[name]["init"] or {})

        except Exception as e:
            raise PipelineDeserializationError("Couldn't deserialize this action: " + name) from e

        try:
            if isinstance(graph.nodes[name]["parameters"], str):
                graph.nodes[name]["parameters"] = json.loads(graph.nodes[name]["parameters"])
        except Exception as e:
            raise PipelineDeserializationError("Couldn't deserialize this action's parameters: " + name) from e


def cool_down(graph: nx.DiGraph()) -> None:
    """
    Serializes all the nodes into a state that can be dumped to JSON or YAML.
    """
    reused_instances = {}
    for name in graph.nodes:

        # If the action is a reused instance, let's add the instance ID to the meta
        if graph.nodes[name]["action"] in reused_instances.values():
            graph.nodes[name]["instance_id"] = [key for key, value in reused_instances.items() if value == graph.nodes[name]["action"]][0]

        elif hasattr(graph.nodes[name]["action"], "init_parameters"):
            # Class nodes need to have a self.init_parameters attribute (or property)
            # if they want their init params to be serialized.
            try:
                graph.nodes[name]["init"] = graph.nodes[name]["action"].init_parameters
            except Exception as e:
                raise PipelineSerializationError(
                    f"A class Action failed to provide its init parameters: {name}"
                ) from e

            # This is a new action instance, so let's store it
            reused_instances[name] = graph.nodes[name]["action"]
        
        # Serialize the callable by name
        try:
            graph.nodes[name]["action"] = graph.nodes[name]["action"].__haystack_action__
        except Exception as e:
            raise PipelineSerializationError(f"Couldn't serialize this action: {name}")

        # Serialize its default parameters with JSON
        try:
            if graph.nodes[name]["parameters"]:
                graph.nodes[name]["parameters"] = json.dumps(graph.nodes[name]["parameters"])
        except Exception as e:
            raise PipelineSerializationError(f"Couldn't serialize this action's parameters: {name}")
