from typing import *
from pathlib import Path
from pprint import pprint

from new_haystack.pipeline import Pipeline
from new_haystack.actions import *

import logging

logging.basicConfig(level=logging.INFO)


@haystack_action
def rename_data(
    name: str,
    data: Dict[str, Any],
    parameters: Dict[str, Any],
    outgoing_edges: List[str],
    stores: Dict[str, Any],
):
    old_name = parameters[name]["old_name"]
    new_name = parameters[name]["new_name"]
    value = data.pop(old_name)
    return {DEFAULT_EDGE_NAME: ({**data, new_name: value}, parameters)}


@haystack_simple_action
def add_one(value):
    return {"value": value + 1}


@haystack_simple_action
def sum(first, second):
    return {"sum": first + second}


def test_simple_pipeline(tmp_path):
    pipeline = Pipeline()
    pipeline.add_node("add_one_1", add_one)
    pipeline.add_node("add_one_2", add_one)
    pipeline.add_node("add_one_3", add_one)
    pipeline.add_node("rename_1", rename_data)
    pipeline.add_node("rename_2", rename_data)
    pipeline.add_node("sum", sum)
    pipeline.connect(["add_one_1", "add_one_2", "rename_1", "sum"])
    pipeline.connect(["add_one_3", "rename_2", "sum"])

    pipeline.draw(tmp_path / "simple_pipeline.png")

    results = pipeline.run(
        {"value": 1},
        parameters={
            "rename_1": {"old_name": "value", "new_name": "first"},
            "rename_2": {"old_name": "value", "new_name": "second"},
        },
        with_debug_info=False,
    )
    pprint(results)

    assert results == {"first": 3, "second": 2, "sum": 5}



if __name__ == "__main__":
    test_simple_pipeline(Path(__file__).parent)
