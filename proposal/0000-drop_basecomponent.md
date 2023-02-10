- Title: Drop `BaseComponent` and reimplement `Pipeline`.
- Decision driver: @ZanSara
- Start Date: (today's date, in format YYYY-MM-DD)
- Proposal PR: (fill in after opening the PR)
- Github Issue or Discussion: (only if available, link the original request for this change)

# Summary

Haystack Pipelines are very powerful objects, but they still have a number of unnecessary limitations, by design and by implementation.

This proposal aims to address most of the implementation issues, some fundamental assumptions like the need for DAGs and the `BaseComponent` class, and proposes a solution for the question of `DocumentStore`'s status with respect to the `Pipeline`.


# Motivation

Pipelines are the fundamental component of Haystack and one of its most powerful concepts. At its core, a Pipeline is a DAG (Directed Acyclic Graph) of classes called Nodes, or Components, each of whom executes a specific transformation on the data flowing along the pipeline. In this way, users can combine powerful libraries, NLP models, and simple Python snippets to connect a herd of tools into a one single, coherent object that can fulfill an infinite variety of tasks.

However as it currently stands the Pipeline object is also imposing a number of limitations on its use, most of which are likely to be unnecessary. Some of these include:
- DAGs are safe, but loops could enable many more usecases, like `Agents`.
- `Pipeline` can select among branches, but cannot run such branches in parallel, except for some specific and inconsistent corner cases.
- `Pipeline`s are forced to have one single input and one single output node, and the input node has to be called either `Query` or `Indexing`, which disallows any other type of pipeline.
- The fixed set of allowed inputs (`query`, `file_paths`, `labels`, `documents`, `meta`, `params` and `debug`) disallows several usecases, like summarization pipelines, translation pipelines, even some sort of generative pipelines.
- `Pipeline`s are often required to have a `DocumentStore` _somewhere_ (see below), even in situation where it wouldn't be needed.
- The redundant concept of `run` and `run_batch()`: nodes should take care of this distinction internally if it's important, otherwise run in batches by default.
- The distinction between a `Pipeline` and its YAML representation is confusing: YAMLs can contain several pipelines, but `Pipeline.save_to_yaml()` can only save a single pipeline at time.

In addition, there are a number of known bugs that makes the current Pipeline implementation hard to work with. Some of these include:
- Branching and merging logic is known to be buggy.
- Nodes can't be added twice to the same pipeline in different locations, limiting their reusability.
- Pipeline YAML validation needs to happen with a schema because Pipelines can only be loaded along with all their nodes, which is a very heavy operation. Shallow or lazy loading of nodes doesn't exist.
- Being forced to use a schema for YAML validation makes impossible to validate the graph in advance.

On top of these issues, there is the tangential issue of `DocumentStore`s and their uncertain relationship with `Pipeline`s. This problem has to be taken into account during a redesign of `Pipeline` and, if necessary, `DocumentStore`s should also be partially impacted. Some of these issues include:
- `DocumentStore`s are nodes in theory, but in practice they can be added to `Pipeline`s only to receive documents to be stored. However `DocumentStore`'s most prominent usecase is as a _source_ of documents, and currently they are not suited for this task without going through an intermediary, most often a `Retriever` class.
  - The relationship between `DocumentStore` and `Retriever` should be left as a topic for a separate proposal but kept in mind, because Retrievers currently act as the main interface for `DocumentStore`s into `Pipeline`s.

This proposal tries to adress all the above point by taking a radical stance with:
- A full reimplementation of the `Pipeline` class that does not limit itself to DAGs, can run branches in parallel, can prune branches, and can process loops.
- Dropping the concept of `BaseComponent` and introducing the concept of stateless `Action`s in its place.
- Define a clear and flexible contract between `Pipeline` and the `Action`s, along with a simplified contract to ease the learning curve.
- Define a clear place for `DocumentStore`s with respect to `Pipeline`s that doesn't forcefully involve `Retriever`s.
- Redesign the YAML representation of `Pipeline`s.

# Basic example

A simple example of how the new Pipeline could look like is shown here. This example does not address `DocumentStore`s, YAML serialization or stateful nodes. However, the API for these features is not as different from the one outlined below, which focuses on the shift between `BaseComponent` and Actions.

For the detailed explanation behind the design choices and all open questions, see the Detailed Design section and the draft implementation here: https://github.com/ZanSara/haystack-2.0-draft 

```python
from typing import *
from pathlib import Path
from pprint import pprint

from new_haystack.pipeline import Pipeline
from new_haystack.actions import haystack_action, haystack_simple_action

# A Haystack Action. See below for details about this contract.
@haystack_action
def rename_data(
    name: str,
    data: Dict[str, Any],
    parameters: Dict[str, Any],
    outgoing_edges: List[str],
    stores: Dict[str, Any],
):
    relevant_parameters = parameters.get(name, {})
    old_name = relevant_parameters["old_name"]
    new_name = relevant_parameters["new_name"]
    value = data.pop(old_name)
    return {edge: ({**data, new_name: value}, parameters) for edge in outgoing_edges}

# A simplified Haystack Action. See below for details about this contract.
@haystack_simple_action
def add_one(value):
    return {"value": value + 1}

# Another simplified Haystack Action. See below for details about this contract.
@haystack_simple_action
def sum(first, second):
    return {"sum": first + second}


pipeline = Pipeline()

# Nodes are added with a name and an action.
# Note the lack of references to any other node.
pipeline.add_node("add_one_1", add_one)
pipeline.add_node("add_one_2", add_one)
pipeline.add_node("add_one_3", add_one)

# Nodes can also accept a set of default parameters.
# They will override the functions's own default, if any,
# but they can be overwritten by new parameters if they are
# given to Pipeline.run() (see below)
pipeline.add_node("rename_1", rename_data, parameters="rename_1": {"old_name": "value", "new_name": "first"})
pipeline.add_node("rename_2", rename_data)
pipeline.add_node("sum", sum)

# Nodes are the connected in a chain with a separate call to Pipeline.connect()
pipeline.connect(["add_one_1", "add_one_2", "rename_1", "sum"])
pipeline.connect(["add_one_3", "rename_2", "sum"])

pipeline.draw("simple_pipeline.png")

# Pipeline run accepts data and parameters only. Such dictionaries can contain 
# anything, depending on what the first node of the pipeline requires.
# Pipeline does not validate the input in any way.
results = pipeline.run(
    data={"value": 1},
    parameters={
        "rename_2": {"old_name": "value", "new_name": "second"},
    }
)
assert results == {"first": 3, "second": 2, "sum": 5}
```

The result of `Pipeline.draw()`:

![image](simple_pipeline.png)


# Detailed design

This section focuses on the concept rather than the implementation strategy. For a discussion on the implementation, see the draft implemetation here: https://github.com/ZanSara/haystack-2.0-draft 

## The Pipeline API

These are the core features that drove the design of the revised Pipeline API:
- An execution graph that is more flexible than a DAG.
- A clear place for `DocumentStore`s
- Shallow/lazy loading of heavy nodes to enable easy validation

Therefore, the revised Pipeline object has the following API:

- Core functions:
    - `run(data, parameters)`: the core of the class. Relies on `networkx` for most of the heavy-lifting. Check out the implementation (https://github.com/ZanSara/haystack-2.0-draft/blob/main/new-haystack/new_haystack/pipeline/pipeline.py) for details: the code is heavily commented on the main loop and on the handling of non-trivial execution paths like branch pruning, parallel branch execution, loops handling, multiple input/output, weighting of inputs, and so on.
    - `draw(path)`: as in the old Pipeline object. Based on `pygraphviz` (which requires `graphviz`), but we might need to look for pure Python alternatives based on Matplotlib to reduce our dependencies.
- Graph building:
    - `add_node(name, action, parameters)`: adds a disconnected node to the graph.
        - Note here the distinction between a **node** (the graph entity) and an **action** (the function that processes the data).
        - `action` accepts callables, however: 
            - Simple functions have to be decorated with either `@haystack_action` or `@haystack_simple_action` (see Action Contract below)
            - Classes need also to be decorated with either of the above decorators and must have a `run()` method, to which all the same limitation as stateless actions apply depending on the decorator used. Optionally they can provide also a `validate()` method and define their own `__init__()` (see Action Contract below).
            See below for a more detailed discussion of the Actions' design.
    - `get_node(name)`: returns the node's information stored in the graph
    - `connect(nodes, weights)`: chains a series of nodes together, adding weights to the edges if given.
- Docstore management:
    - `connect_store(name, store)`: adds a DocumentStore to the stores that are passed down to the actions through the `stores` variable.
    - `list_stores()`: returns all connected stores
    - `get_store(name)`: returns a specific document store by name
    - `disconnect_store(name, store)`: removes a store from the registry.
- Serialization and validation:
    - `warm_up()`: iterate over the nodes of the graph and initializes all of the actions, replacing their names with callables or object instances, depending on the type of action. A "warm" pipeline is ready to run, but can't be serialized right away. A call to `save()` on a "warm" pipeline causes `cool_down()` to run first.
    - `cool_down()`: iterate over the nodes of the graph and replaces all actions, stateless or stateful, with their name in the known actions registry. A "cool" pipeline can't be run right away, but it's ready for serialization. Calling `run()` on a "cold" pipeline causes `warm_up()` to run first.
    - `__init__(path=None)`: if a path is given, loads the pipeline from the YAML found at that path. Note that at this stage Pipeline will collect actions from all imported modules (see the implementation - the search can be scoped down to selected modules) and **all actions are validated (see `validate()`) but not initialized**: therefore, `__init__` creates "cold" pipelines.
    - `save(path)`: serializes and saves the pipeline as a YAML at the given path. If the pipeline is not "cold", `cool_down()` is called first.
    - `validate()`: iterate over the nodes of the graph to check whether `action` is present, as a string or as a callable, into the known actions registry. `validate()` works on both "warm" pipelines, "cold" pipeines, and even pipelines in mixed states, without affecting it (so it doesn't need to call `warm_up()` or `cool_down()` to work).

Example pipeline topologies supported by this new implementation (images taken from the test suite):
<details>
<summary>Branching pipeline with parallel branch execution</summary>

![image](many2many_pipeline.png)

In this pipeline all branches will run in parallel. Due to the difference in length across branches, merge nodes might need to wait for all upstream nodes to have a chance at running before attempting to run themselves.

</details>

<details>
<summary>Branching pipeline with branch pruning</summary>

![image](pruning_pipeline.png)

In this pipeline, only one edge will run depending on the decision taken by the classifier.

</details>

<details>
<summary>Simple pipeline with a loop</summary>

![image](looping_pipeline.png)

Note: the "entry point" node highlights an interesting limitation of the automatic detection of pipeline heads.

Currently the algorithm defines a pipeline head as any node with no input edges. However in pipelines with loops there might be no nodes matching this condition, hence the need for the "entry point" no-op node to point the algorithm in the right direction.

</details>

## The Action contract

Actions can be of two types: stateless and stateful. They follow a similar contract.

### Stateless Action

```python
@haystack_action
def my_stateless_action(
    name: str,
    data: Dict[str, Any],
    parameters: Dict[str, Dict[str, Any]],
    outgoing_edges: List[str],
    stores: Dict[str, Any],
):
    my_own_parameters = parameters.get(name, {})

    ... some code ...

    return {edge: (data, parameters) for edge in outgoing_edges}
```

This is the contract for a stateless Haystack Action. It takes:

- `name: str`: the name of the node that is running it. Allows the action to find its own parameters in the parameters dictionary (see below).
- `data: Dict[str, Any]`: the input data flowing down the pipeline.
- `parameters: Dict[str, Dict[str, Any]]`: a dict of dicts with all the parameters for all nodes flowing down the pipeline. Note that all nodes have access to all parameters for all other nodes: this might come handy to nodes like `Agent`s, that might want to influence the behavior of nodes downstream.
- `outgoing_edges`: the name of the edges connected downstream of this action. Mostly useful for decision nodes, for error messages and basic validation
- `stores`: a dictionary of all the (Document)Stores connected to this pipeline.

This function is supposed to return a tuple of dictionaries `(data, parameters)` along all edges that should run. Decision nodes should output on selected edges only, in order to prevent the deselected branches from running.

The decorator is needed for this action to be recognized as a Haystack action.

### Stateful Action

```python
@haystack_action
class Counter:

    def __init__(self, start_from: int = 0):
        self.counter = start_from
        self.init_parameters = {"start_from": start_from}

    def run(
        self,
        name: str,
        data: Dict[str, Any],
        parameters: Dict[str, Any],
        outgoing_edges: Set[str],
        stores: Dict[str, Any],
    ):
        self.counter += 1
        return {edge: (data, parameters) for edge in outgoing_edges}

    @staticmethod
    def validate(init_parameters: Dict[str, Any]) -> None:
        if init_parameters.get("start_from", 0) < 0:
            raise ActionError("We count only positive numbers here.")
```

This is the contract for a stateful Haystack Action. 

Note how the `run()` method follows an identical contract than a stateless action: however, the decorator should be placed on top of the class.

The `__init__` method is optional and can take an arbitrary number of parameters. However there are two requirements:
- The init parameters need to be serializable for the `Pipeline.save()` method to work.
- All relevant init parameters need to be manually added to the `self.init_parameters` attribute. Failure to do so means that missing parameters won't be serialized.

The `validate()` method is also optional. Must be `@staticmethod` to allow validation to be performed on cold pipelines.

### Simplified Actions

```python
@haystack_simple_action
def action(value, add):
    return {"value": value + add}
```
     
In simplified Haystack actions, all parameters name are automatically extracted from the signature, looked for in the `data` and `parameters` dictionaries, and passed over to this function. Their output will be merged to the rest of the data flowing down the pipeline, overwriting any value already present under the same key.

Simplified actions have a series of limitations due to their extremely simplified contract:
- They can only output the same values on all outgoing edges
- They don't know their names
- They can't access document stores
- They can't change any other node's parameters

Most of the above limitations are arbitrary and set in order to minimize the impact on the contract on the layout of
the function. Some intermediate simplifications can be designed along this one, for example:
- A simplified action taking also the `stores` variable to access the stores registry, like:
```python
@haystack_simple_action_with_stores
def action(value, add, stores):
    return {"value": value + add}
```
- A simplified action that can return primitives instead of dictionaries and specify the output key in the decorator, like:
```python
@haystack_very_simple_action(output_name="value")
def action(value, add):
    return value + add
```
- Simplified decision nodes, like:
```python
@haystack_decision
def action(value, threshold):
    if value > threshold:
        return "above"  # name of the selected edge: all the data flows down unchanged
    return "below"
```
and so on.

** Bonus: Why run() and not __call__()? **

Internally, both `@haystack_action` and `@haystack_simple_action` map `run()` to `__call__()` to simplify the job of `Pipeline.run()`. However, the simplified contract wraps the `run()` method heavily, destroying its original signature. To keep the original `run()` method usable outside of `Pipeline`s, the decorator assigns the wrapped version to `__call__()` to leave `run()` untouched. See the (arguably very scary) implementation of `@haystack_simple_node` if you want a headache or just love second-order functions wrapping multiple dunder class methods all at once.

TODO: `@haystack_simple_action` works amazingly, but needs a better implementation.


### Actions discovery logic

Currently, at init time `Pipeline` scans the entire `sys.modules` looking for any function or class which is decorated with the `@haystack_action` decorator (or `@haystack_simple_action` and other similar simplifiers). 

Such search can be scoped down or directed elsewhere by setting the `search_actions_in` init parameter in `Pipeline`: however, all modules must be imported for the search to be successful. 

Search also might fail in narrow corner cases: for example, inner functions are not discovered (often the case in tests). For these scenarios, `Pipeline` also accepts a `extra_actions` init parameter that allows users to explicitly provide a dictionary of actions to integrate with the other discovered actions.

Name collisions are handled by prefixing the action name with the name of the module it was imported from.

See the draft implementation for details. 

### YAML representation

TO BE DEFINED - the implementation in the draft is just stubbed and by no means representative of the desired outcome.

### Distinction between a Pipeline and a (Bundle? Blueprint? Manifest? Project? Schematic? Let's find a good name)

Bundles are tiny wrappers on top of a set of pipelines, little more than dictionaries containing several pipelines.

However, they can contain nodes or stores that are shared across different pipelines and pass them over when they're initialized.

They also have `warm_up` and `cool_down` methods, which simply mirror the pipeline's.


# Open questions

- YAML representation
- Name for the group of Pipeline
- Which "simplified contracts" we want to implement

# Drawbacks

There are a number of drawbacks about the proposed approach:

- Migration is going to be far from straightforward for us. Although many nodes can probably work with minor adaptations into the new system, it would be beneficial for most of them to be reduced to their `run()` method, especially indexing nodes. This means nodes need, at least, to be migrated one by one to the new system and code copied over.
- Migration is going to be far from straightforward for the users: see Adoption strategy.
- This system allows for pipelines with more complex topologies, which brings the risk of more corner cases. `Pipeline.run()` must be made very solid in order to avoid this scenario.
- Stateless actions need less upfront validation, but might more easily break while running due to unexpected inputs. While well designed nodes should internally check and deal with such situations, we might face larger amount of bugs due to our failure at spotting lack of checks for unexpected inputs at review time.
- The entire system work on the assumption that actions are well behaving and respect both the contract and a number of "unspoken rules", like not touching other node's parameters unless necessary, pop their own input instead of letting it flow down the pipeline, etc. Malicious or otherwise "rude" actions can wreak havoc in `Pipeline`s very easily by messing with other action's parameters and inputs.

# Adoption strategy

Old and new Pipeline, Nodes and actions are going to be, in most cases, fully incompatible.

We must provide a migration script that can convert their existing pipeline YAMLs into the new ones.

This proposal is best thought as part of the design of Haystack 2.0, where we can afford drastic API changes such as this.

Adoption for dC: still an open question.
