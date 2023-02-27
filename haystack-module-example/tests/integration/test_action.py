import pytest

from new_haystack.pipeline import Pipeline
from haystack_module_example.my_node import my_node


def test_my_node():
    pipe = Pipeline()
    pipe.add_node("my_node", node=my_node)

    results = pipe.run(data={"test": "value"})
    assert results == {"test": "value"}

