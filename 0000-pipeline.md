- Title: Pipeline Refactoring
- Decision driver: ZanSara
- Start Date: (today's date, in format YYYY-MM-DD)
- Proposal PR: (fill in after opening the PR)
- Github Issue or Discussion: (only if available, link the original request for this change)

# Summary




# Motivation

Pipelines are the fundamental component of Haystack and one of its most powerful concepts. At its core, a Pipeline is intented as a DAG (Directed Acyclic Graph) of classes called Nodes, or Components, each of whom executes a specific transformation on the data flowing along the pipeline. In this way, users can combine powerful libraries, NLP models, and simple Python snippets to connect a herd of tools into a one single, coherent object that can fulfill an infinite variety of tasks.

However, as it currently stands, the Pipeline object is also imposing a number of limitations on its use, most of which are likely to be unnecessary. Some of these include:
- DAGs are safe, but loops could enable many more usecases, like `Agents`.
- `Pipeline` can select among branches, but cannot run such branches in parallel, except for some specific and inconsistent corner cases.
- `Pipeline`s are forced to have one single input and one single output node.
- The requirement to have one input node called either `Query` or `Indexing`, which disallows any other type of pipeline.
- Fixed set of allowed inputs: `query`, `file_paths`, `labels`, `documents`, `meta`, `params` and `debug`
- The redundant concept of `run` and `run_batch()`: nodes should take care of this distinction internally if it's important, otherwise run in batches by default.
- The distinction between a `Pipeline` and its YAML representation is confusing: YAMLs can contain several pipelines, but `Pipeline.save_to_yaml()` can only save a single pipeline at time.

In addition, there are a number of known bugs that makes the current Pipeline implementation hard to work with. Some of these include:
- Branching and merging logic is known to be buggy.
- Nodes can't be added twice to the same pipeline in different locations, limiting their reusability.
- Pipeline YAML validation needs to happen with a schema because Pipelines can only be loaded along with all their nodes, which is a very heavy operation. Shallow or lazy loading of nodes doesn't exist.
- Being forced to use a schema for YAML validation makes impossible to validate the graph in advance.

On top of these issues, a tangential situation is the one concerning `DocumentStore`s. Their uncertain relationship with `Pipeline`s hads to be taken into account during the redesing, and if necessary, they could also be impacted. Some of these issues include:
- `DocumentStore`s are nodes in theory, but in practice they can be added to `Pipeline`s only to receive documents to be stored. However `DocumentStore`'s most prominent usecase is as a source of documents, and currently they are not suited for this task without going through an intermediary, most often a `Retriever` class.
  - The relationship between `DocumentStore` and `Retriever` should be left as a topic for a separate proposal.

This proposal tries to adress all the above point by taking a radical stance and proposing:
- A full reimplementation of the `Pipeline` class.
- Dropping the concept of `BaseComponent` and introducing the concept of `Action` in its place.
- Define a clear contract between `Pipeline` and the `Action`s.
- Define a clear place for `DocumentStore`s with respect to `Pipeline`s.

# Basic example

When applicable, write a snippet of code showing how the new feature would
be used.

# Detailed design

This is the bulk of the proposal. Explain the design in enough detail for somebody
familiar with Haystack to understand, and for somebody familiar with the
implementation to implement. Get into specifics and corner-cases,
and include examples of how the feature is used. Also, if there's any new terminology involved,
define it here.

# Drawbacks

Look at the feature from the other side: what are the reasons why we should _not_ work on it? Consider the following:

- What's the implementation cost, both in terms of code size and complexity?
- Can the solution you're proposing be implemented as a separate package, outside of Haystack?
- Does it teach people more about Haystack?
- How does this feature integrate with other existing and planned features?
- What's the cost of migrating existing Haystack pipelines (is it a breaking change?)?

There are tradeoffs to choosing any path. Attempt to identify them here.

# Alternatives

What other designs have you considered? What's the impact of not adding this feature?

# Adoption strategy

If we implement this proposal, how will the existing Haystack users adopt it? Is
this a breaking change? Can we write a migration script?

# How we teach this

Would implementing this feature mean the documentation must be re-organized
or updated? Does it change how Haystack is taught to new developers at any level?

How should this feature be taught to the existing Haystack users (for example with a page in the docs,
a tutorial, ...).

# Unresolved questions

Optional, but suggested for first drafts. What parts of the design are still
TBD?
