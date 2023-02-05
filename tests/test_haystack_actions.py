from typing import *

import logging

import pytest

from haystack.pipeline import Pipeline, PipelineError
from haystack.actions import haystack_action, ActionError


def test_bare_node_stateless_missing_necessary_data():
    @haystack_action
    def action(name: str, data: Dict[str, Any], parameters: Dict[str, Any], outgoing_edges: List[str]):
        return {outgoing_edges[0]: ({**data, "value": data["value"] + parameters["test_node"]["add"]}, parameters)}

    pipeline = Pipeline()
    pipeline.add_node("test_node", action)

    with pytest.raises(PipelineError):
        pipeline.run(data={"wrong_value": 1}, parameters={"test_node": {"add": 2}})


def test_bare_node_stateless_missing_necessary_parameter(caplog):
    @haystack_action
    def action(name: str, data: Dict[str, Any], parameters: Dict[str, Any], outgoing_edges: List[str]):
        return {outgoing_edges[0]: ({**data, "value": data["value"] + parameters["test_node"]["add"]}, parameters)}

    pipeline = Pipeline()
    pipeline.add_node("test_node", action)

    with pytest.raises(PipelineError):
        pipeline.run(data={"value": 1})

    with caplog.at_level(logging.INFO):
        with pytest.raises(PipelineError):
            pipeline.run(data={"value": 1}, parameters={"wrong_node": {"add": 2}})
            assert "wrong_node" in caplog.text


def test_bare_node_stateless_parameterless_no_input_no_output():
    @haystack_action
    def action(name: str, data: Dict[str, Any], parameters: Dict[str, Any], outgoing_edges: List[str]):
        return {outgoing_edges[0]: (data, parameters)}

    pipeline = Pipeline()
    pipeline.add_node("test_node", action)

    results = pipeline.run(data={"value": 1}, parameters={"test_node": {"not_requested": 2}})

    assert results == {"value": 1}


def test_bare_node_stateless_parameterless_one_input_one_output():
    @haystack_action
    def action(name: str, data: Dict[str, Any], parameters: Dict[str, Any], outgoing_edges: List[str]):
        return {outgoing_edges[0]: ({**data, "value": data["value"] + 1}, parameters)}

    pipeline = Pipeline()
    pipeline.add_node("test_node", action)

    results = pipeline.run(data={"value": 1}, parameters={"test_node": {"not_requested": 2}})

    assert results == {"value": 2}


def test_bare_node_stateless_parameterless_two_input_one_output():
    @haystack_action
    def action(name: str, data: Dict[str, Any], parameters: Dict[str, Any], outgoing_edges: List[str]):
        return {outgoing_edges[0]: ({**data, "result": data["value"] + data["other_value"]}, parameters)}

    pipeline = Pipeline()
    pipeline.add_node("test_node", action)

    results = pipeline.run(data={"value": 1, "other_value": 5}, parameters={"test_node": {"not_requested": 2}})

    assert results == {"value": 1, "other_value": 5, "result": 6}


def test_bare_node_stateless_parameterless_one_input_two_outputs():
    @haystack_action
    def action(name: str, data: Dict[str, Any], parameters: Dict[str, Any], outgoing_edges: List[str]):
        return {outgoing_edges[0]: ({**data, "value+1": data["value"] + 1, "value+2": data["value"] + 2}, parameters)}

    pipeline = Pipeline()
    pipeline.add_node("test_node", action)

    results = pipeline.run(data={"value": 1}, parameters={"test_node": {"not_requested": 2}})

    assert results == {"value": 1, "value+1": 2, "value+2": 3}


def test_bare_node_stateless_with_run_parameter_one_input_one_output():
    @haystack_action
    def action(name: str, data: Dict[str, Any], parameters: Dict[str, Any], outgoing_edges: List[str]):
        return {outgoing_edges[0]: ({**data, "value": data["value"] + parameters["test_node"]["add"]}, parameters)}

    pipeline = Pipeline()
    pipeline.add_node("test_node", action)

    results = pipeline.run(data={"value": 1}, parameters={"test_node": {"add": 2}})

    assert results == {"value": 3}


def test_bare_node_stateless_with_run_parameter_one_input_one_output():
    @haystack_action
    def action(name: str, data: Dict[str, Any], parameters: Dict[str, Any], outgoing_edges: List[str]):
        return {outgoing_edges[0]: ({**data, "value": data["value"] + parameters["test_node"]["add"]}, parameters)}

    pipeline = Pipeline()
    pipeline.add_node("test_node", action)

    results = pipeline.run(data={"value": 1}, parameters={"test_node": {"add": 2}})

    assert results == {"value": 3}


def test_bare_node_stateful_must_have_run():
    with pytest.raises(ActionError, match="run"):

        @haystack_action
        class Action:
            def __init__(self):
                self.counter = 0
                self.init_parameters = {}


def test_bare_node_stateful_must_have_validate():
    with pytest.raises(ActionError, match="validate"):

        @haystack_action
        class Action:
            def __init__(self):
                self.counter = 0
                self.init_parameters = {}

            def run(self, name: str, data: Dict[str, Any], parameters: Dict[str, Any], outgoing_edges: List[str]):
                self.counter += 1
                return {outgoing_edges[0]: (data, parameters)}


def test_bare_node_stateful_dont_need_init():
    @haystack_action
    class Action:
        def run(self, name: str, data: Dict[str, Any], parameters: Dict[str, Any], outgoing_edges: List[str]):
            return {outgoing_edges[0]: (data, parameters)}

        @classmethod
        def validate(cls, init_parameters: Dict[str, Any]) -> None:
            pass

    action = Action()
    pipeline = Pipeline()
    pipeline.add_node("test_node", action)

    results = pipeline.run(data={"value": 1})

    assert results == {"value": 1}


def test_bare_node_stateful_no_init_no_input_no_output_no_validation():
    @haystack_action
    class Action:
        def __init__(self):
            self.counter = 0

        def run(self, name: str, data: Dict[str, Any], parameters: Dict[str, Any], outgoing_edges: List[str]):
            self.counter += 1
            return {outgoing_edges[0]: (data, parameters)}

        @classmethod
        def validate(cls, init_parameters: Dict[str, Any]) -> None:
            pass

    action = Action()
    pipeline = Pipeline()
    pipeline.add_node("test_node", action)

    results = pipeline.run(data={"value": 1})

    assert results == {"value": 1}
    assert pipeline.get_node("test_node")["action"].counter == 1

    results = pipeline.run(data={"value": 1})

    assert results == {"value": 1}
    assert pipeline.get_node("test_node")["action"].counter == 2


def test_bare_node_stateful_with_init_param_no_input_no_output_no_validation():
    @haystack_action
    class Action:
        def __init__(self, counter):
            self.counter = counter
            self.init_parameters = {"counter": counter}

        def run(self, name: str, data: Dict[str, Any], parameters: Dict[str, Any], outgoing_edges: List[str]):
            self.counter += 1
            return {outgoing_edges[0]: (data, parameters)}

        @classmethod
        def validate(cls, init_parameters: Dict[str, Any]) -> None:
            return True

    action = Action(10)
    pipeline = Pipeline()
    pipeline.add_node("test_node", action)

    results = pipeline.run(data={"value": 3})

    assert results == {"value": 3}
    assert pipeline.get_node("test_node")["action"].counter == 11

    results = pipeline.run(data={"value": 3})

    assert results == {"value": 3}
    assert pipeline.get_node("test_node")["action"].counter == 12


def test_bare_node_stateful_one_input_one_output():
    @haystack_action
    class Action:
        def __init__(self, counter):
            self.counter = counter
            self.init_parameters = {"counter": counter}

        def run(self, name: str, data: Dict[str, Any], parameters: Dict[str, Any], outgoing_edges: List[str]):
            self.counter += data["value"]
            return {outgoing_edges[0]: ({**data, "value": data["value"] + 3}, parameters)}

        @classmethod
        def validate(cls, init_parameters: Dict[str, Any]) -> None:
            pass

    action = Action(10)
    pipeline = Pipeline()
    pipeline.add_node("test_node", action)

    results = pipeline.run(data={"value": 2})

    assert results == {"value": 5}
    assert pipeline.get_node("test_node")["action"].counter == 12
