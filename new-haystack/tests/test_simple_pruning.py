from typing import *
from pathlib import Path
from pprint import pprint

from new_haystack.pipeline import Pipeline
from new_haystack.actions import *

import logging

logging.basicConfig(level=logging.INFO)



@haystack_node
def even_number_classifier(
    name: str,
    data: Dict[str, Any],
    parameters: Dict[str, Any],
    outgoing_edges: Set[str],
    stores: Dict[str, Any],
):
    print(outgoing_edges)
    if outgoing_edges != {"even", "odd"}:
        raise ActionError("Connect both edges 'even' and 'odd'.")

    if data["value"] % 2 == 0:
        return {"even": (data, parameters)}
    else:
        return {"odd": (data, parameters)}


@haystack_simple_action
def add_message(message):
    return {"message": message}


@haystack_simple_action
def add_one(value):
    return {"value": value + 1}


@haystack_simple_action
def add_ten(value):
    return {"value": value + 10}


def test_simple_pruning(tmp_path):
    pipeline = Pipeline()
    pipeline.add_node("classifier", even_number_classifier)
    pipeline.add_node("even_number", add_message, parameters={"message": "The number was even!"})
    pipeline.add_node("odd_number", add_message, parameters={"message": "The number was odd!"})
    pipeline.add_node("add_one_even", add_one)
    pipeline.add_node("add_one_odd", add_one)
    pipeline.add_node("add_ten", add_ten)
    pipeline.connect(["classifier.even", "even_number", "add_one_even", "add_ten"])
    pipeline.connect(["classifier.odd", "odd_number", "add_one_odd", "add_ten"])

    pipeline.draw(tmp_path / "pruning_pipeline.png")

    results = pipeline.run(
        {"value": 0},
        with_debug_info=False,
    )
    pprint(results)
    assert results == {"value": 11, "message": "The number was even!"}

    results = pipeline.run(
        {"value": 1},
        with_debug_info=False,
    )    
    pprint(results)
    assert results == {"value": 12, "message": "The number was odd!"}


if __name__ == "__main__":
    test_simple_pruning(Path(__file__).parent)
