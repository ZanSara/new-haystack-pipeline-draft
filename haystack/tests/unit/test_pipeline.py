from typing import *

import logging

import pytest

from haystack.pipeline import (
    Pipeline,
    PipelineDeserializationError,
    PipelineError,
    PipelineSerializationError,
    PipelineValidationError,
    validate,
)
from haystack.actions import haystack_action, ActionError, ActionValidationError


def test_bare_node_stateless_with_run_parameter_one_in_edge_many_out_edge_one_input_one_output():
    @haystack_action
    def action(
        name: str,
        data: Dict[str, Any],
        parameters: Dict[str, Any],
        outgoing_edges: List[str],
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
        return {outgoing_edges[0]: (data, parameters)}

    pipeline = Pipeline()
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
