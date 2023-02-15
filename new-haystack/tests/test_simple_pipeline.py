from typing import *
from pathlib import Path
from pprint import pprint

from new_haystack.pipeline import Pipeline
from new_haystack.actions import *

import logging

logging.basicConfig(level=logging.INFO)


from typing import Dict, Any

import logging

from tqdm import tqdm

from new_haystack.actions import haystack_node, DEFAULT_EDGE_NAME


logger = logging.getLogger(__name__)


@haystack_node
class AddValue:
    def __init__(self, add: int = 1):
        self.add = add

        # Contract
        self.init_parameters = {"add": add}
        self.expects_inputs = {DEFAULT_EDGE_NAME: {"value"}}
        self.produces_output = {DEFAULT_EDGE_NAME: {"value"}}

    def run(
        self,
        name: str,
        data: Dict[str, Any],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        value = data[DEFAULT_EDGE_NAME]["value"]
        value += self.add

        data["value"] = value
        return {DEFAULT_EDGE_NAME: (data, parameters)}



@haystack_node
class Sum:
    def __init__(self, expected_edges: List[str], expected_variable_name: str):
        self.expected_variable_name = expected_variable_name
        
        # Contract
        self.init_parameters = {"expected_edges": expected_edges, "expected_variable_name": expected_variable_name}
        self.expects_inputs = {expected_edge: {expected_variable_name} for expected_edge in expected_edges}
        self.produces_output = {DEFAULT_EDGE_NAME: {"sum"}}

    def run(
        self,
        name: str,
        data: Dict[str, Any],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        sum = 0
        for edge in self.expects_inputs.keys():
            value = data[edge][self.expected_variable_name]
            sum += value

        data["sum"] = sum
        return {DEFAULT_EDGE_NAME: (data, parameters)}



def test_simple_pipeline(tmp_path):

    add_two = AddValue(add=2)
    summer = Sum(expected_edges={"value_1", "value_2"}, expected_variable_name="value")

    pipeline = Pipeline()
    pipeline.add_node("first_addition", add_two)
    pipeline.add_node("second_addition", add_two)
    pipeline.add_node("third_addition", add_two)
    pipeline.add_node("sum", summer)
    pipeline.connect(["add_one_1", "add_one_2.value_1", "sum"])
    pipeline.connect(["add_one_3.value_2", "sum"])

    pipeline.draw(tmp_path / "simple_pipeline.png")

    results = pipeline.run(
        {"value": 1},
        parameters={},
    )
    pprint(results)



if __name__ == "__main__":
    test_simple_pipeline(Path(__file__).parent)
