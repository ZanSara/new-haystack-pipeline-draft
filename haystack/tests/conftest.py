import pytest
import haystack


@pytest.fixture(autouse=True)
def search_modules_for_pipeline(request, monkeypatch):
    """
    The pipeline's default search paths don't work well with PyTest.
    Here we replace the __main__ default search path with the test's name,
    so all actions defined in the test's scope are loaded properly

    NOTE: actions don't need to be _in_ the test body. They just need to
    be visible from it. For example, they might be defined in the same file,
    or might be imported in the test suite.
    """
    testname = request.node.name
    monkeypatch.setattr(
        haystack.pipeline.pipeline, "DEFAULT_SEARCH_MODULES", [testname]
    )
