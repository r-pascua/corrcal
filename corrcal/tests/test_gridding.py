import pytest


def test_placeholder():
    try:
        import pyfof
        PYFOF = True
    except ImportError:
        PYFOF = False
    assert PYFOF
