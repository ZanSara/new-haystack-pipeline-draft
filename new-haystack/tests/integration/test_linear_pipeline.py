from typing import Dict, Any, List, Tuple

from pathlib import Path
from pprint import pprint

from new_haystack.pipeline import Pipeline
from new_haystack.nodes import *
from new_haystack.nodes import haystack_node

import logging

logging.basicConfig(level=logging.DEBUG)


@haystack_node
class AddValue:
    def __init__(self, add: int = 1, input_name: str = "value", output_name: str = "value"):
        self.add = add

        # Contract
        self.init_parameters = {"add": add}
        self.expected_inputs = [input_name]
        self.expected_outputs = [output_name]

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        for _, value in data:
            value += self.add

        return ({"value": value}, )



@haystack_node
class Double:
    def __init__(self, expected_inputs_name: str = "value"):
        # Contract
        self.init_parameters = {"expected_inputs_name": expected_inputs_name}
        self.expected_inputs = [expected_inputs_name]
        self.expected_outputs = [expected_inputs_name]

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        for _, value in data:
            value *= 2

        return ({self.expected_outputs[0]: value}, )



def test_pipeline(tmp_path):
    pipeline = Pipeline(search_nodes_in=[__name__])
    pipeline.add_node("first_addition", AddValue(add=2))
    pipeline.add_node("second_addition", AddValue(add=1))
    pipeline.add_node("double", Double(expected_inputs_name="value"))
    pipeline.connect(["first_addition", "double", "second_addition"])
    pipeline.draw(tmp_path / "linear_pipeline.png")

    results = pipeline.run({"value": 1})
    pprint(results)

    assert results == {"value": 7}


if __name__ == "__main__":
    test_pipeline(Path(__file__).parent)
