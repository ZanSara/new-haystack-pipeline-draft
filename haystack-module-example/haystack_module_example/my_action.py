from typing import Dict, Set, Any

import logging

from new_haystack.actions import haystack_action


logger = logging.getLogger(__name__)


@haystack_action
def my_action(
    name: str,
    data: Dict[str, Any],
    parameters: Dict[str, Any],
    outgoing_edges: Set[str],
    stores: Dict[str, Any],
):

    #  ...  do something ...

    return {edge: (data, parameters) for edge in outgoing_edges}
