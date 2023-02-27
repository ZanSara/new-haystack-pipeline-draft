from haystack_module_example.my_node import my_node


def test_my_node_2():
    assert my_node(name="my_node", data={"test": "value"}, parameters={}, outgoing_edges=["edge"], stores={}) == {"edge": ({"test": "value"}, {})}

