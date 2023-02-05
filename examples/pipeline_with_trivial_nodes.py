from typing import *

from haystack.pipeline import Pipeline
from haystack.actions import *

import logging

logging.basicConfig(level=logging.INFO)


######################################################################################
# "Bare" Action API
#
#  Take
#       name: str, data: Dict[str, Any], parameters: Dict[str, Any], outgoing_edges: List[str]
#
#  Return:
#
#       for a single edge:
#           (altered_data, altered_parameters)  for a single edge
#
#       for many edges:
#           {"edge_1": (data_1, all_parameters), "edge_2": (data_2, all_parameters)}
#


# No parameters
@haystack_action
def plus_one_1(name: str, data: Dict[str, Any], parameters: Dict[str, Any], outgoing_edges: List[str]):
    return {outgoing_edges[0]: ({**data, "value": data["value"] + 1}, parameters)}


# A few inputs
@haystack_action
def sum_a_b(name: str, data: Dict[str, Any], parameters: Dict[str, Any], outgoing_edges: List[str]):
    return {outgoing_edges[0]: ({**data, "value": data["a"] + data["b"]}, parameters)}


# with parameters
@haystack_action
def multiply_1(name: str, data: Dict[str, Any], parameters: Dict[str, Any], outgoing_edges: List[str]):
    value = data["value"]
    by = parameters[name]["by"]
    return {outgoing_edges[0]: ({**data, "value": value * by}, parameters)}


# stateful
@haystack_action
class Count_1:

    def __init__(self, message="Hello"):
        self._init_parameters = {"message": message}
        self.counter = 0
        self.message = message

    def run(self, name: str, data: Dict[str, Any], parameters: Dict[str, Any], outgoing_edges: List[str]):
        self.counter += 1
        print(f"{self.message} / I'm {name}, instance {self}, and I was called {self.counter} times. Value is now {data['value']}")
        return {outgoing_edges[0]: (data, parameters)}

    @classmethod
    def validate(cls, init_parameters: Dict[str, Any]) -> None:
        print("~~ Count_1 is being validated! ~~")
        if "message" in init_parameters.keys() and init_parameters["message"].lower() not in ["hello", "bye"]:
            raise ActionValidationError("'message' must be either 'hello' or 'bye'!")

    @property
    def init_parameters(self):
        print("A Counter_1 instance is returning the init parameters")
        return self._init_parameters



######################################################################################
# "Simplified" Action API
#
#  Take
#       name: str, data: Dict[str, Any], parameters: Dict[str, Any], outgoing_edges: List[str]
#
#  Return:
#
#       for a single edge:
#           (altered_data, altered_parameters)  for a single edge
#
#       for many edges:
#           {"edge_1": (data_1, all_parameters), "edge_2": (data_2, all_parameters)}
#
# No parameters
@haystack_simple_action
def plus_one_2(value):
    return {"value": value + 1}


# with parameters
@haystack_simple_action
def multiply_2(value, by=4):
    return {"value": value * by}


# stateful
@haystack_simple_action
class Count_2:

    def __init__(self, message):
        self.message = message
        self.counter = 0

    def run(self, value):
        self.counter += 1
        print(f"{self.message} / I was called {self.counter} times. Value is now {value}")
        return {}

    # TODO Can we implement default validation in haystack_simple_node?
    @classmethod
    def validate(cls, init_parameters: Dict[str, Any]):
        print("~~ Count_2 is being validated! ~~")
        if "message" in init_parameters.keys() and init_parameters["message"].lower() not in ["hello", "bye"]:
            raise ActionValidationError("'message' must be either 'hello' or 'bye'!")



###########################################################################################################


pipe = Pipeline()

counter = Count_1()
pipe.add_node(name="counter", action=counter)
pipe.add_node(name="counter1", action=counter)
pipe.add_node(name="multiplier", action=multiply_1)
pipe.add_node(name="counter2", action=counter)
pipe.add_node(name="plus_one", action=plus_one_1)
pipe.add_node(name="counter3", action=counter)
pipe.add_node(name="counter4", action=counter)
pipe.connect(["counter", "counter1", "multiplier", "counter2", "plus_one", "counter3", "counter4"])

pipe.draw("pipeline1.png")

results = pipe.run(data={"value": 2}, parameters={"multiplier": {"by": 3}})

print(results)

print("#################################################")


pipe = Pipeline()


@haystack_action
def edge_number(name: str, data: Dict[str, Any], parameters: Dict[str, Any], outgoing_edges: List[str]):
    return {outgoing_edge: ({**data, "value": index}, parameters) for index, outgoing_edge in enumerate(outgoing_edges)}


@haystack_action
def rename(name: str, data: Dict[str, Any], parameters: Dict[str, Any], outgoing_edges: List[str]):
    old_name = parameters[name]["old_name"]
    new_name = parameters[name]["new_name"]
    return {outgoing_edges[0]: ({**data, new_name: data[old_name]}, parameters)}


counter = Count_1()

pipe.add_node(name="counter", action=counter)
pipe.add_node(name="enumerate", action=edge_number)
pipe.add_node(name="plus_one_1", action=plus_one_1)
pipe.add_node(name="plus_one_2", action=plus_one_1)
pipe.add_node(name="rename1", action=rename, parameters={"old_name": "value", "new_name": "a"})
pipe.add_node(name="rename2", action=rename)
pipe.add_node(name="sum", action=sum_a_b)
pipe.add_node(name="counter2", action=counter)
pipe.connect(["counter", "enumerate.first", "plus_one_1", "rename1", "sum", "counter2"])
pipe.connect(["enumerate.second", "plus_one_2", "rename2", "sum"])

pipe.draw("pipeline2.png")

results = pipe.run(data={"value": 0}, parameters={
    "rename2": {"old_name": "value", "new_name": "b"}
})

print(results)

print("#################################################")

pipe = Pipeline()

counter2 = Count_2()
counter3 = Count_1(message="Ciao!")
pipe.add_node(name="counter1", action=counter2)
pipe.add_node(name="multiplier", action=multiply_2, parameters={"by": 1})
pipe.add_node(name="counter2", action=counter2)
pipe.add_node(name="plus_one", action=plus_one_2)
pipe.add_node(name="counter3", action=counter3)
pipe.connect(["counter1", "multiplier", "counter2", "plus_one", "counter3"])

pipe.draw("pipeline3.png")

results = pipe.run(data={"value": 2})
print("------")
results = pipe.run(data={"value": 2}, parameters={"multiplier": {"by": 2}})

print(results)

pipe.save("pipeline.yaml")

print("#################################################")

pipe2 = Pipeline("pipeline.yaml")

results = pipe2.run(data={"value": 2})
print("------")
results = pipe2.run(data={"value": 2}, parameters={"multiplier": {"by": 2}})

print(results)
