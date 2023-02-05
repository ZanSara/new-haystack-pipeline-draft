from typing import *
from haystack.actions import haystack_action, haystack_simple_action


######################################################################################
# "Bare" Node API
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
def multiplier_1(name: str, data: Dict[str, Any], parameters: Dict[str, Any], outgoing_edges: List[str]):
    value = data["value"]
    by = parameters[name]["by"]
    return {outgoing_edges[0]: ({**data, "value": value * by}, parameters)}


# stateful
@haystack_action
class Counter_1:

    def __init__(self, message="Hello"):
        self._init_parameters = {"message": message}
        self.counter = 0
        self.message = message

    def __call__(self, name: str, data: Dict[str, Any], parameters: Dict[str, Any], outgoing_edges: List[str]):
        self.counter += 1
        print(f"{self.message} / I'm {name}, instance {self}, and I was called {self.counter} times. Value is now {data['value']}")
        return {outgoing_edges[0]: (data, parameters)}

    @classmethod
    def validate(cls):
        print("~~ Counter_1 is being validated! ~~")
        return True

    @property
    def init_parameters(self):
        print("A Counter_1 instance is returning the init parameters")
        return self._init_parameters



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

    def __init__(self, message="hello"):
        self.message = message
        self.counter = 0

    def __call__(self, value):
        self.counter += 1
        print(f"{self.message} / I was called {self.counter} times. Value is now {value}")
        return {}

    # TODO implement default validation in haystack_simple_node
    @classmethod
    def validate(cls):
        print("~~ Count_2 is being validated! ~~")
        return True
