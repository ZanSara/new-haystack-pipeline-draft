import pytest

from new_haystack.pipeline import Pipeline
from haystack_module_example.my_action import my_action


def test_my_action():
    pipe = Pipeline()
    pipe.add_node("my_action", action=my_action)

    results = pipe.run(data={"test": "value"})
    assert results == {"test": "value"}

