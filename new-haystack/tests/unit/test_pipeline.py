from typing import *

import logging

import pytest

from new_haystack.pipeline import (
    Pipeline,
)
from new_haystack.actions import haystack_action


def test_bare_node_stateless_with_run_parameter_one_in_edge_many_out_edge_one_input_one_output():
    @haystack_action
    def action(
        name: str,
        data: Dict[str, Any],
        parameters: Dict[str, Any],
        outgoing_edges: Set[str],
        stores: Dict[str, Any],
    ):
        return {
            "edge_1": (
                {**data, "new_value": data["value"] + parameters["test_node"]["add"]},
                parameters,
            ),
            "edge_2": (
                {
                    **data,
                    "new_value": data["value"] + parameters["test_node"]["add"] + 2,
                },
                parameters,
            ),
        }

    @haystack_action
    def empty(
        name: str,
        data: Dict[str, Any],
        parameters: Dict[str, Any],
        outgoing_edges: List[str],
        stores: Dict[str, Any],
    ):
        return {edge: (data, parameters) for edge in outgoing_edges}

    # NOTE: we specify extra_actions here only because action is now an inner function
    # and does not get picked up by the automatic discovery. I consider it a reasonable
    # limitation for now, and `extra_actions` can go around it.
    pipeline = Pipeline(extra_actions={"action": action, "empty": empty})
    pipeline.add_node("test_node", action)
    pipeline.add_node("empty1", empty)
    pipeline.add_node("empty2", empty)
    pipeline.connect(["test_node.edge_1", "empty1"])
    pipeline.connect(["test_node.edge_2", "empty2"])

    results = pipeline.run(data={"value": 1}, parameters={"test_node": {"add": 2}})

    assert results == {
        "empty1": {"value": 1, "new_value": 3},
        "empty2": {"value": 1, "new_value": 5},
    }
