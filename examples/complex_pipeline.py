from typing import *

from haystack.pipeline import Pipeline
from haystack.actions import *

import logging

logging.basicConfig(level=logging.DEBUG)


@haystack_action
def rename_data(name: str, data: Dict[str, Any], parameters: Dict[str, Any], outgoing_edges: List[str]):
    old_name = parameters[name]["old_name"]
    new_name = parameters[name]["new_name"]
    return {DEFAULT_EDGE_NAME: ({**data, new_name: data[old_name]}, parameters)}


@haystack_action
def rename_param(name: str, data: Dict[str, Any], parameters: Dict[str, Any], outgoing_edges: List[str]):
    target_node = parameters[name]["target_node"]
    old_name = parameters[name]["old_name"]
    new_name = parameters[name]["new_name"]
    parameters[target_node][new_name] = parameters[target_node][old_name]
    return {DEFAULT_EDGE_NAME: (data, parameters)}


@haystack_action
def edge_number(name: str, data: Dict[str, Any], parameters: Dict[str, Any], outgoing_edges: List[str]):
    return {outgoing_edge: ({**data, "edge": index+1}, parameters) for index, outgoing_edge in enumerate(outgoing_edges)}


@haystack_simple_action
def add_one(value):
    return {"value": value + 1}


@haystack_simple_action
def sum(first, second):
    return {"sum": first + second}


@haystack_simple_action
class Count:
    def __init__(self, starting_count = 0):
        self.counter = starting_count

    def run(self):
        self.counter += 1
        print(f"Count is counting! We're at {self.counter}")


counter = Count(starting_count=10)

pipeline = Pipeline()
pipeline.add_node("input_count_1", counter)
pipeline.add_node("input_count_2", counter)
#pipeline.add_node("output_count", counter)
pipeline.add_node("fork", edge_number)
pipeline.add_node("count_more", counter)
pipeline.add_node("collapse", edge_number)
pipeline.add_node("add_one_2", add_one)
pipeline.add_node("rename_1", rename_data, parameters={"old_name": "value", "new_name": "first"})
pipeline.add_node("rename_2", rename_data, parameters={"old_name": "value", "new_name": "second"})
pipeline.add_node("rename_3", rename_data)
#pipeline.add_node("rename_4", rename_data)
pipeline.add_node("sum", add_one)
pipeline.add_node("edge_number", edge_number)
pipeline.add_node("add_one_3", add_one)

pipeline.connect(["input_count_1", "fork.first", "count_more", "collapse", "rename_1", "sum", "edge_number.first", "rename_3", "add_one_3"])
pipeline.connect(["fork.second", "collapse"])
pipeline.connect(["input_count_2", "add_one_2", "rename_2", "sum", "edge_number.second"]) #, "rename_4", "output_count"])

pipeline.draw("pipeline.png")

results = pipeline.run({"value": 1}, parameters={
    "rename_3": {"old_name": "edge", "new_name": "first"},
    "rename_4": {"old_name": "edge", "new_name": "first"}
})

print(results)