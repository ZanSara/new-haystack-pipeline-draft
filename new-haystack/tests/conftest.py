import sys

import pytest

import new_haystack
# from new_haystack.pipeline.pipeline import DEFAULT_SEARCH_MODULES


# @pytest.fixture(autouse=True)
# def search_modules_for_pipeline(request, monkeypatch):
#     """
#     The pipeline's default search paths don't work well with PyTest.
#     Here we replace the __main__ default search path with the test's name,
#     so all actions defined in the test's scope are loaded properly

#     NOTE: actions don't need to be _in_ the test body. They just need to
#     be visible from it. For example, they might be defined in the same file,
#     or might be imported in the test suite.
#     """
#     # testname = request.node.name
#     #if testname in sys.modules.keys():

#     test_modules = [module_name for module_name in sys.modules.keys() if module_name.startswith("test")]
#     monkeypatch.setattr(
#         new_haystack.pipeline.pipeline, "DEFAULT_SEARCH_MODULES", DEFAULT_SEARCH_MODULES + test_modules
#     )
