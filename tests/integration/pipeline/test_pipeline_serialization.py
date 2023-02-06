from typing import *
from pathlib import Path

import haystack
from haystack.pipeline import Pipeline
from haystack.actions import *

import logging

logging.basicConfig(level=logging.WARNING)


# No parameters
@haystack_simple_action
def plus_one(value):
    return {"value": value + 1}


# with parameters
@haystack_simple_action
def multiply(value, by=4):
    return {"value": value * by}


# stateful
@haystack_simple_action
class Count:
    def __init__(self, message="hello"):
        self.message = message
        self.counter = 0

    def run(self):
        self.counter += 1
        print(f"{self.message}! I was called {self.counter} times.")
        return {}

    @staticmethod
    def validate(init_parameters: Dict[str, Any]):
        if "message" in init_parameters.keys() and init_parameters["message"].lower() not in ["hello", "bye"]:
            raise ActionValidationError("'message' must be either 'hello' or 'bye'!")


def test_pipeline_serialization(tmp_path):
    pipe = Pipeline()

    counter = Count()  # Try message="Ciao" to see the validation kicking in
    counter2 = Count(message="bye")  # Try message="Ciao" to see the validation kicking in
    pipe.add_node(name="counter1", action=counter)
    pipe.add_node(name="multiplier", action=multiply, parameters={"by": 1})
    pipe.add_node(name="counter2", action=counter)
    pipe.add_node(name="plus_one", action=plus_one)
    pipe.add_node(name="counter3", action=counter2)
    pipe.connect(["counter1", "multiplier", "counter2", "plus_one", "counter3"])

    pipe.draw(tmp_path / "pipeline_to_serialize.png")

    results_no_params = pipe.run(data={"value": 2})
    print(results_no_params)
    results_with_params = pipe.run(data={"value": 3}, parameters={"multiplier": {"by": 2}})
    print(results_with_params)

    pipe.save(tmp_path / "pipeline.yaml")

    pipe2 = Pipeline(tmp_path / "pipeline.yaml")

    new_results_no_params = pipe2.run(data={"value": 2})
    print(new_results_no_params)
    new_results_with_params = pipe2.run(data={"value": 3}, parameters={"multiplier": {"by": 2}})
    print(new_results_with_params)

    assert results_no_params == new_results_no_params
    assert results_with_params == new_results_with_params


if __name__ == "__main__":
    test_pipeline_serialization(Path(__file__).parent)
