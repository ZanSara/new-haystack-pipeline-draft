from typing import *

from haystack.pipeline import Pipeline
from haystack.actions import *

import logging

logging.basicConfig(level=logging.INFO)


pipe = Pipeline()

counter = Counter_1()
pipe.add_node(name="counter", action=counter)
pipe.add_node(name="counter1", action=counter)
pipe.add_node(name="multiplier", action=multiplier_1)
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


counter = Counter_1()

pipe.add_node(name="counter", action=counter)
pipe.add_node(name="enumerate", action=edge_number)
pipe.add_node(name="plus_one_1", action=plus_one_1)
pipe.add_node(name="plus_one_2", action=plus_one_1)
pipe.add_node(name="rename1", action=rename, parameters={"old_name": "value", "new_name": "a"})
pipe.add_node(name="rename2", action=rename)
pipe.add_node(name="sum", action=sum_a_b)
pipe.add_node(name="counter2", action=counter)
pipe.connect(["counter", "enumerate", "plus_one_1", "rename1", "sum", "counter2"])
pipe.connect(["enumerate", "plus_one_2", "rename2", "sum"])

pipe.draw("pipeline2.png")

results = pipe.run(data={"value": 0}, parameters={
    "rename2": {"old_name": "value", "new_name": "b"}
})

print(results)

print("#################################################")

pipe = Pipeline()

counter2 = Count_2(message="Bye")
counter3 = Counter_1()
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
