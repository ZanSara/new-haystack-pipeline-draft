from typing import *
from pathlib import Path
from pprint import pprint

from new_haystack.pipeline import Pipeline
from new_haystack.actions import *

import logging

logging.basicConfig(level=logging.DEBUG)



@haystack_node
def below_ten(
    name: str,
    data: Dict[str, Any],
    parameters: Dict[str, Any],
    outgoing_edges: List[str],
    stores: Dict[str, Any],
):
    if set(outgoing_edges) != {"above_ten", "below_ten"}:
        raise ActionError("This action needs 'above_ten' and 'below_ten' edges.")

    if data["value"] < 3:
        print("---> Below ten!")
        return {"below_ten": (data, parameters)}
    else:
        print("---> Above ten!")
        return {"above_ten": (data, parameters)}


@haystack_simple_action
def add_one(value):
    return {"value": value + 1}


def test_simple_looping(tmp_path):
    pipeline = Pipeline()
    pipeline.add_node("entry_point", add_one)
    pipeline.add_node("checker", below_ten)
    pipeline.add_node("add_one", add_one)
    pipeline.add_node("add_one_final", add_one)
    pipeline.connect(["entry_point", "checker.below_ten", "add_one", "checker"])
    pipeline.connect(["checker.above_ten", "add_one_final"])

    pipeline.draw(tmp_path / "looping_pipeline.png")

    results = pipeline.run(
        {"value": 0},
        with_debug_info=False,
    )
    pprint(results)



if __name__ == "__main__":
    test_simple_looping(Path(__file__).parent)
