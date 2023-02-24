from typing import Dict, Any, List, Tuple, Optional, Set

from pathlib import Path
from pprint import pprint

from new_haystack.pipeline import Pipeline
from new_haystack.actions import haystack_node

import logging

logging.basicConfig(level=logging.DEBUG)


@haystack_node
class AddValue:
    """
    Single input, single output node
    """
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
    def __init__(self, input_name: str = "value", output_name: Optional[str] = None):
        # Contract
        self.init_parameters = {"input_name": input_name}
        self.expected_inputs = [input_name]
        if not output_name:
            output_name = input_name
        self.expected_outputs = [output_name]

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


@haystack_node
class Remainder:
    """
    Single input, multi output node, returning the input value on one output edge only
    """
    def __init__(self, input_name: str, divisor: int):
        self.input_name = input_name
        self.divisor = divisor
        # Contract
        self.init_parameters = {"input_name": input_name, "divisor": divisor}
        self.expected_inputs = [input_name]
        self.expected_outputs = [str(out) for out in range(divisor)]

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        if len(data) != 1:
            raise ValueError("This node accepts a single input.")
        remainder = data[0][1] % self.divisor
        return ({str(remainder): data[0][1]}, )


@haystack_node
class Sum:
    def __init__(self, expected_inputs_name: str = "value", expected_inputs_count: int = 2):
        # Contract
        self.init_parameters = {"expected_inputs_count": expected_inputs_count, "expected_inputs_name": expected_inputs_name}
        self.expected_inputs = [expected_inputs_name] * expected_inputs_count
        self.expected_outputs = ["sum"]

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        sum = 0
        for _, value in data:
            if value:
                sum += value

        return ({"sum": sum}, )
    


@haystack_node
class NoOp:
    def __init__(self, edges: Set[str] = {"value"}):
        # Contract
        self.init_parameters = {"edges": edges}
        self.expected_inputs = list(edges)
        self.expected_outputs = list(edges)

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        output = {}
        for key, value in data:
            output[key] = value
        return (output, )



def test_pipeline(tmp_path):
    
    pipeline = Pipeline(search_actions_in=[__name__])

    add_one = AddValue(add=1, input_name="value")
    pipeline.add_node("add_one", add_one)
    pipeline.add_node("remainder", Remainder(input_name="value", divisor=3))
    pipeline.add_node("add_ten", AddValue(add=10, input_name="0"))
    pipeline.add_node("double", Double(input_name="1", output_name="value"))
    pipeline.add_node("add_three", AddValue(add=3, input_name="2"))
    pipeline.add_node("add_one_again", add_one)
    pipeline.add_node("sum", Sum(expected_inputs_count=4, expected_inputs_name="value"))
    pipeline.add_node("no-op", NoOp(edges={"value"}))

    pipeline.connect(["add_one", "remainder"])
    pipeline.connect(["remainder.0", "add_ten", "sum"])
    pipeline.connect(["remainder.1", "double", "sum"])
    pipeline.connect(["remainder.2", "add_three", "add_one_again", "sum"])
    pipeline.connect(["no-op", "sum"])
    
    pipeline.draw(tmp_path / "decision_and_merge_pipeline.png")

    results = pipeline.run({"value": 1})
    pprint(results)

    assert results == {'value': 7}


if __name__ == "__main__":
    test_pipeline(Path(__file__).parent)
