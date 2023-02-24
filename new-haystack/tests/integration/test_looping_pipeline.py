from typing import *
from pathlib import Path
from pprint import pprint

from new_haystack.pipeline import Pipeline
from new_haystack.actions import haystack_node

import logging

logging.basicConfig(level=logging.DEBUG)


@haystack_node
class Below:
    def __init__(self, threshold: int = 10, input_name: str = "value", output_above: str = "above", output_below: str = "below"):
        self.threshold = threshold
        self.output_above = output_above
        self.output_below = output_below

        # Contract
        self.init_parameters = {"threshold": threshold, "input_name": input_name, "output_above": output_above, "output_below": output_below}
        self.expected_inputs = [input_name]
        self.expected_outputs = [output_above, output_below]

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        if len(data) != 1:
            raise ValueError("Below takes one input value only")

        if data[0][1] < self.threshold:
            print("---> Below ten!")
            return {self.output_below: data[0][1]}, 
        else:
            print("---> Above ten!")
            return {self.output_above: data[0][1]}, 


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
class Merge:
    """
    Returns one single output on the output edge, which corresponds to the value of the last input edge that is not None.
    If no input edges received any value, returns None as well.
    """
    def __init__(self, expected_inputs_name: str = "value", expected_inputs_count: int = 2, output_name: str = "value"):
        self.output_name = output_name
        # Contract
        self.init_parameters = {"expected_inputs_count": expected_inputs_count, "expected_inputs_name": expected_inputs_name, "output_name": output_name}
        self.expected_inputs = [expected_inputs_name] * expected_inputs_count
        self.expected_outputs = [output_name]

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        output = None
        for _, value in data:
            if value is not None:
                output = value

        return ({self.output_name: output}, )



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



@haystack_node
class Count:
    def __init__(self, edge: str):
        self.count = 0
        # Contract
        self.init_parameters = {"edge": edge}
        self.expected_inputs = [edge]
        self.expected_outputs = [edge]

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        self.count += 1
        return ({data[0][0]: data[0][1]}, )




def test_pipeline(tmp_path):
    pipeline = Pipeline(search_actions_in=[__name__])
    counter = Count(edge="value")
    pipeline.add_node("entry_point", NoOp(edges=["value"]))
    pipeline.add_node("merge", Merge())
    pipeline.add_node("below_10", Below(threshold=10))
    pipeline.add_node("add_one", AddValue(add=1, input_name="below"))
    pipeline.add_node("counter", counter)
    pipeline.add_node("add_two", AddValue(add=2, input_name="above"))
    pipeline.connect(["entry_point", "merge", "below_10.below", "add_one", "counter",  "merge"])
    pipeline.connect(["below_10.above", "add_two"])

    pipeline.draw(tmp_path / "looping_pipeline.png")

    results = pipeline.run(
        {"value": 3},
    )
    pprint(results)
    print("counter: ", counter.count)

    assert results == {"value": 12}
    assert counter.count == 7


if __name__ == "__main__":
    test_pipeline(Path(__file__).parent)
