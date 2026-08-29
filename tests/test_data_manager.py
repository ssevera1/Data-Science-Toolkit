"""Tests for column type auto-detection and column creation."""

import io

import pandas as pd
import pytest
import streamlit as st

from core import constants
from core.data_manager import _auto_detect_type, add_column, load_csv


@pytest.fixture(autouse=True)
def clean_state():
    """Give each test an empty var-types map and no DataFrame."""
    st.session_state["stats_var_types"] = {}
    st.session_state["stats_col_counter"] = 0
    st.session_state.pop("df", None)
    yield


def detect(values):
    """Run auto-detection over a one-column frame and return the assigned type."""
    df = pd.DataFrame({"col": values})
    _auto_detect_type("col", df)
    return st.session_state["stats_var_types"]["col"]


def test_variable_types_is_a_list_of_the_named_constants():
    # pages/data_input.py calls VARIABLE_TYPES.index(...) and passes it to
    # st.selectbox as the options list, so it must stay an ordered list.
    assert constants.VARIABLE_TYPES == [
        constants.METRIC,
        constants.NOMINAL,
        constants.ORDINAL,
    ]


@pytest.mark.parametrize(
    "values, expected",
    [
        # All-empty column falls back to Metric.
        ([None, None, None], constants.METRIC),
        # Binary numeric -> Nominal.
        ([0, 1, 0, 1, 1, 0], constants.NOMINAL),
        # Few distinct numeric levels over many rows -> Ordinal.
        ([1, 2, 3] * 10, constants.ORDINAL),
        # Continuous numeric -> Metric.
        ([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5], constants.METRIC),
        # Low-cardinality strings -> Nominal.
        (["a", "b", "c", "a", "b"], constants.NOMINAL),
        # High-cardinality strings -> Nominal (no "Text" type exists).
        ([f"s{i}" for i in range(50)], constants.NOMINAL),
    ],
)
def test_auto_detect_type_assigns_a_valid_variable_type(values, expected):
    result = detect(values)
    assert result == expected
    # Regression: a bad constants lookup used to raise TypeError here, and any
    # assigned type must be selectable in the Data Input dropdown.
    assert result in constants.VARIABLE_TYPES


def test_load_csv_detects_every_column():
    csv = "num,grp,label\n1.5,0,alpha\n2.5,1,beta\n9.75,0,gamma\n"
    ok, err = load_csv(io.StringIO(csv))

    assert (ok, err) == (True, None)
    types = st.session_state["stats_var_types"]
    assert set(types) == {"num", "grp", "label"}
    assert all(t in constants.VARIABLE_TYPES for t in types.values())
    assert types["num"] == constants.METRIC
    assert types["grp"] == constants.NOMINAL
    assert types["label"] == constants.NOMINAL


def test_add_column_marks_the_new_column_as_metric():
    add_column()

    new_col = "Var1"
    assert new_col in st.session_state["df"].columns
    assert st.session_state["stats_var_types"][new_col] == constants.METRIC
