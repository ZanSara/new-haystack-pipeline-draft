from typing import Dict, Any, List, Tuple

from pathlib import Path
from pprint import pprint

from new_haystack.pipeline import Pipeline
from new_haystack.actions import *
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
class Sum:
    """
    Multi input, single output node
    """
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
            sum += value

        return ({"sum": sum}, )


@haystack_node
class Remainder:
    """
    Single input, multi output node, skipping all outputs except for one
    """
    def __init__(self, input_name: str = "value", divisor: int = 2):
        self.input_name = input_name
        self.divisor = divisor
        # Contract
        self.init_parameters = {"input_name": input_name, "divisor": divisor}
        self.expected_inputs = [input_name]
        self.expected_outputs = [f"remainder_is_{remainder}" for remainder in range(divisor)]

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        divisor = parameters.get(name, {}).get("divisor", self.divisor)
        for _, value in data:
            remainder = value % divisor
        return ({f"remainder_is_{remainder}": value}, )


@haystack_node
class Enumerate:
    """
    Single input, multi output node, returning on all output edges
    """
    def __init__(self, input_name: str, outputs_count: int):
        self.input_name = input_name
        self.outputs_count = outputs_count
        # Contract
        self.init_parameters = {"input_name": input_name, "outputs_count": outputs_count}
        self.expected_inputs = [input_name]
        self.expected_outputs = [str(out) for out in range(outputs_count)]

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        divisor = parameters.get(name, {}).get("divisor", self.divisor)
        output = {str(value): None for value in range(divisor)}
        return (output, )


@haystack_node
class Greet:
    """
    Single input single output no-op node.
    """
    def __init__(self, edge: str = "any", message: str = "Hi!"):
        self.message = message
        # Contract
        self.init_parameters = {"edge": edge, "message": message}
        self.expected_inputs = [edge]
        self.expected_outputs = [edge]

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        message = parameters.get(name, {}).get("message", self.message)
        print("\n#################################")
        print(message)
        print("#################################\n")
        return ({data[0][0]: data[0][1]}, )


@haystack_node
class Rename:
    """
    Single input single output rename node
    """
    def __init__(self, input_name: str, output_name: str):
        # Contract
        self.init_parameters = {"input_name": input_name, "output_name": output_name}
        self.expected_inputs = [input_name]
        self.expected_outputs = [output_name]

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        return ({self.expected_outputs[0]: data[0][1]}, )


@haystack_node
class Accumulate:
    """
    Stateful reusable node
    """
    def __init__(self, edge: str):
        self.sum = 0
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
        for _, value in data:
            self.sum += value
        return ({data[0][0]: data[0][1]}, )


@haystack_node
class Merge:
    """
    Convert the list of tuples into a dict, makings lists for repeated keys.
    """
    def __init__(self, expected_inputs: List[str] = [], output_value: str = "merge"):
        self.output_value = output_value
        # Contract
        self.init_parameters = {"expected_inputs": expected_inputs, "output_value": output_value}
        self.expected_inputs = expected_inputs
        self.expected_outputs = [output_value]

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        merged = {}
        for key, value in data:
            if key in merged.keys():
                if isinstance(merged[key], list):
                    merged[key].append(value)
                else:
                    merged[key] = [merged[key], value]
            else:
                merged[key] = value
        return ({self.output_value: merged}, )


@haystack_node
class Replicate:
    """
    Replicates the input data on all given output edges
    """
    def __init__(self, input_value: str = "value", expected_outputs: List[str] = []):
        self.expected_outputs = expected_outputs
        # Contract
        self.init_parameters = {"input_value": input_value, "expected_outputs": expected_outputs}
        self.expected_inputs = [input_value]
        self.expected_outputs = expected_outputs

    def run(
        self,
        name: str,
        data: List[Tuple[str, Any]],
        parameters: Dict[str, Any],
        stores: Dict[str, Any],
    ):
        return ({output: data[0][1] for output in self.expected_outputs}, )


def test_complex_pipeline(tmp_path):
    accumulate = Accumulate(edge="value")

    pipeline = Pipeline(search_actions_in=[__name__])
    pipeline.add_node("greet_first", Greet(edge="value", message="Hello!"))    
    pipeline.add_node("accumulate_1", accumulate)
    pipeline.add_node("add_two", AddValue(add=2))
    pipeline.add_node("parity_check", Remainder(divisor=2))
    pipeline.add_node("add_one", AddValue(add=1))
    pipeline.add_node("accumulate_2", accumulate)

    pipeline.add_node("rename_even_to_value", Rename(input_name="remainder_is_0", output_name="value"))    
    pipeline.add_node("rename_odd_to_value", Rename(input_name="remainder_is_1", output_name="value"))  
    pipeline.add_node("rename_0_to_value", Rename(input_name="0", output_name="value"))
    pipeline.add_node("rename_1_to_value", Rename(input_name="1", output_name="value"))

    pipeline.add_node("greet_again", Greet(edge="value", message="Hello again!"))    
    pipeline.add_node("sum", Sum(expected_inputs_name="value", expected_inputs_count=3))

    pipeline.add_node("greet_enumerator", Greet(edge="any", message="Hello from enumerator!"))    
    pipeline.add_node("enumerate", Enumerate(input_name="any", outputs_count=2))
    pipeline.add_node("add_three", AddValue(add=3))

    pipeline.add_node("merge", Merge(expected_inputs=["sum", "value"]))
    pipeline.add_node("greet_one_last_time", Greet(edge="merge", message="Bye bye!"))    
    pipeline.add_node("replicate", Replicate(input_value="merge", expected_outputs=["first", "second"]))
    pipeline.add_node("add_five", AddValue(add=5, input_name="first"))
    pipeline.add_node("add_four", AddValue(add=4, input_name="second"))

    pipeline.connect([
        "greet_first", 
        "accumulate_1", 
        "add_two", 
        "parity_check.remainder_is_0", 
        "rename_even_to_value", 
        "greet_again", 
        "sum", 
        "merge", 
        "greet_one_last_time", 
        "replicate.first", 
        "add_five"
    ])
    pipeline.connect([
        "replicate.second", 
        "add_four"
    ])
    pipeline.connect([
        "parity_check.remainder_is_1", 
        "rename_odd_to_value", 
        "add_one", 
        "accumulate_2", 
        "merge"
    ])
    pipeline.connect([
        "greet_enumerator", 
        "enumerate.1",
        "rename_1_to_value", 
        "sum"
    ])
    pipeline.connect([
        "enumerate.0", 
        "rename_0_to_value", 
        "add_three", 
        "sum"
    ])
    pipeline.draw(tmp_path / "complex_pipeline.png")

    results = pipeline.run({"value": 1})
    pprint(results)
    print("accumulated: ", accumulate.sum)



if __name__ == "__main__":
    test_complex_pipeline(Path(__file__).parent)
