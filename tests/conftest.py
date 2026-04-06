import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--snapshot", action="store_true", default=False)
