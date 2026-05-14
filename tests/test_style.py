# tests/test_style.py
from src.utils.style import REGIME_COLOURS, METHOD_COLOURS

def test_regime_colours():
    assert set(REGIME_COLOURS.keys()) == {"A", "B", "C", "D", "E"}

def test_method_colours_exist():
    expected = {"FTE", "HH", "ProjEuler", "KL", "ChoiKwok", "Exact"}
    assert expected.issubset(METHOD_COLOURS.keys())