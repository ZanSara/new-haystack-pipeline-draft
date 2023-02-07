from typing import *

import logging

import pytest

from haystack.pipeline import Pipeline, PipelineError
from haystack.actions import haystack_simple_action, ActionError


def test_simple_node_stateless_missing_necessary_data():
    @haystack_simple_action
    def action(value, add):
        pass

    pipeline = Pipeline()
    pipeline.add_node("test_node", action)

    with pytest.raises(PipelineError):
        pipeline.run(data={"wrong_value": 1}, parameters={"test_node": {"add": 2}})


def test_simple_node_stateless_missing_necessary_parameter(caplog):
    @haystack_simple_action
    def action(value, add):
        pass

    pipeline = Pipeline()
    pipeline.add_node("test_node", action)

    with caplog.at_level(logging.INFO):
        with pytest.raises(PipelineError):
            pipeline.run(data={"value": 1}, parameters={"test_node": {"wrong_name": 2}})
            assert "wrong_name" in caplog.text

    with caplog.at_level(logging.INFO):
        with pytest.raises(PipelineError):
            pipeline.run(data={"value": 1}, parameters={"wrong_node": {"add": 2}})
            assert "wrong_node" in caplog.text


def test_simple_node_stateless_unexpected_parameter(caplog):
    @haystack_simple_action
    def action(value):
        pass

    pipeline = Pipeline()
    pipeline.add_node("test_node", action)

    results = pipeline.run(
        data={"value": 1}, parameters={"test_node": {"not_requested": 2}}
    )

    assert results == {"value": 1}
    assert "not_requested" in caplog.text


def test_simple_node_stateless_parameterless_no_input_no_output():
    @haystack_simple_action
    def action():
        pass

    pipeline = Pipeline()
    pipeline.add_node("test_node", action)

    results = pipeline.run(data={"value": 1})

    assert results == {"value": 1}


def test_simple_node_stateless_parameterless_no_input_one_new_output():
    @haystack_simple_action
    def action():
        return {"out": "a"}

    pipeline = Pipeline()
    pipeline.add_node("test_node", action)

    results = pipeline.run(data={"value": 1})

    assert results == {"value": 1, "out": "a"}


def test_simple_node_stateless_parameterless_no_input_one_existing_output():
    @haystack_simple_action
    def action():
        return {"value": "a"}

    pipeline = Pipeline()
    pipeline.add_node("test_node", action)

    results = pipeline.run(data={"value": 1})

    assert results == {"value": "a"}


def test_simple_node_stateless_parameterless_one_input_one_output():
    @haystack_simple_action
    def action(value):
        return {"value": value + 1}

    pipeline = Pipeline()
    pipeline.add_node("test_node", action)

    results = pipeline.run(data={"value": 1})

    assert results == {"value": 2}


def test_simple_node_stateless_parameterless_two_input_one_output():
    @haystack_simple_action
    def action(first, second):
        return {"sum": first + second}

    pipeline = Pipeline()
    pipeline.add_node("test_node", action)

    results = pipeline.run(data={"first": 1, "second": 5})

    assert results == {"first": 1, "second": 5, "sum": 6}


def test_simple_node_stateless_with_run_parameter_one_input_one_output():
    @haystack_simple_action
    def action(value, add):
        return {"value": value + add}

    pipeline = Pipeline()
    pipeline.add_node("test_node", action)

    results = pipeline.run(data={"value": 1}, parameters={"test_node": {"add": 2}})

    assert results == {"value": 3}


def test_simple_node_stateful_must_have_run():
    with pytest.raises(ActionError, match="run"):

        @haystack_simple_action
        class Action:
            def __init__(self):
                self.counter = 0


def test_simple_node_stateful_dont_need_validate():
    @haystack_simple_action
    class Action:
        def __init__(self):
            self.counter = 0

        def run(self):
            self.counter += 1

    action = Action()
    pipeline = Pipeline()
    pipeline.add_node("test_node", action)

    results = pipeline.run(data={"value": 1})

    assert results == {"value": 1}
    assert pipeline.get_node("test_node")["action"].counter == 1


def test_simple_node_stateful_dont_need_init():
    @haystack_simple_action
    class Action:
        def run(self):
            self.counter = 5

    action = Action()
    pipeline = Pipeline()
    pipeline.add_node("test_node", action)

    results = pipeline.run(data={"value": 1})

    assert results == {"value": 1}
    assert pipeline.get_node("test_node")["action"].counter == 5


def test_simple_node_stateful_no_init_no_input_no_output():
    @haystack_simple_action
    class Action:
        def __init__(self):
            self.counter = 0

        def run(self):
            self.counter += 1

    action = Action()
    pipeline = Pipeline()
    pipeline.add_node("test_node", action)

    results = pipeline.run(data={"value": 50})

    assert results == {"value": 50}
    assert pipeline.get_node("test_node")["action"].counter == 1

    results = pipeline.run(data={"value": 40})
    results = pipeline.run(data={"value": 30})
    results = pipeline.run(data={"value": 20})
    results = pipeline.run(data={"value": 10})

    assert results == {"value": 10}
    assert pipeline.get_node("test_node")["action"].counter == 5


def test_simple_node_stateful_with_init_param_no_input_no_output_no_validation():
    @haystack_simple_action
    class Action:
        def __init__(self, counter):
            self.counter = counter

        def run(self):
            self.counter += 1

    action = Action(counter=10)
    pipeline = Pipeline()
    pipeline.add_node("test_node", action)

    results = pipeline.run(data={"value": 3})

    assert results == {"value": 3}
    assert pipeline.get_node("test_node")["action"].counter == 11

    results = pipeline.run(data={"value": 3})
    results = pipeline.run(data={"value": 3})
    results = pipeline.run(data={"value": 3})
    results = pipeline.run(data={"value": 3})

    assert results == {"value": 3}
    assert pipeline.get_node("test_node")["action"].counter == 15


def test_simple_node_stateful_one_input_one_output():
    @haystack_simple_action
    class Action:
        def __init__(self, counter):
            self.counter = counter

        def run(self, increase):
            self.counter += increase
            return {"total": increase + 1}

    action = Action(counter=10)
    pipeline = Pipeline()
    pipeline.add_node("test_node", action)

    results = pipeline.run(data={"increase": 2})

    assert results == {"increase": 2, "total": 3}
    assert pipeline.get_node("test_node")["action"].counter == 12
