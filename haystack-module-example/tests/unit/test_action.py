from haystack_module_example.my_action import my_action


def test_my_action_2():
    assert my_action(name="my_action", data={"test": "value"}, parameters={}, outgoing_edges=["edge"], stores={}) == {"edge": ({"test": "value"}, {})}
