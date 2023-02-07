from typing import Dict, List, Any

import logging

from haystack.actions import haystack_action


logger = logging.getLogger(__name__)


@haystack_action
def my_action(
    name: str,
    data: Dict[str, Any],
    parameters: Dict[str, Any],
    outgoing_edges: List[str],
    stores: Dict[str, Any],
):

    #  ...  do something ...

    return {edge: (data, parameters) for edge in outgoing_edges}
