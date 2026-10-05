"""Tests for the empty/all-NaN DataFrame guard in core/validators.py."""

import pandas as pd
import pytest
import streamlit as st

from core.validators import (
    validate_column_exists,
    validate_groups,
    validate_survival_inputs,
    validate_two_metrics,
)

EMPTY_MESSAGE = "DataFrame is empty or contains only NaN values."


@pytest.fixture(autouse=True)
def clean_state():
    st.session_state.pop("df", None)
    yield
    st.session_state.pop("df", None)


def set_df(df):
    st.session_state["df"] = df


# validate_groups, validate_two_metrics, and validate_survival_inputs are the
# validators pages/*.py actually call. Each must short-circuit on an empty or
# all-NaN DataFrame instead of falling through to a KeyError or a generic
# "insufficient data" message once a real column name is missing.


def test_validate_groups_rejects_empty_df():
    set_df(pd.DataFrame())
    assert validate_groups("metric", "group") == (False, EMPTY_MESSAGE)


def test_validate_groups_rejects_all_nan_df():
    set_df(pd.DataFrame({"metric": [None, None], "group": [None, None]}))
    assert validate_groups("metric", "group") == (False, EMPTY_MESSAGE)


def test_validate_groups_passes_through_for_populated_df():
    set_df(pd.DataFrame({"metric": [1, 2, 3, 4], "group": ["a", "a", "b", "b"]}))
    ok, msg = validate_groups("metric", "group")
    assert (ok, msg) == (True, None)


def test_validate_two_metrics_rejects_empty_df():
    set_df(pd.DataFrame())
    assert validate_two_metrics("col1", "col2") == (False, EMPTY_MESSAGE)


def test_validate_two_metrics_rejects_all_nan_df():
    set_df(pd.DataFrame({"col1": [None, None], "col2": [None, None]}))
    assert validate_two_metrics("col1", "col2") == (False, EMPTY_MESSAGE)


def test_validate_survival_inputs_rejects_empty_df():
    set_df(pd.DataFrame())
    assert validate_survival_inputs("time", "event") == (False, EMPTY_MESSAGE)


def test_validate_survival_inputs_rejects_all_nan_df():
    set_df(pd.DataFrame({"time": [None, None], "event": [None, None]}))
    assert validate_survival_inputs("time", "event") == (False, EMPTY_MESSAGE)


def test_validate_column_exists_rejects_all_nan_df():
    set_df(pd.DataFrame({"col": [None, None]}))
    assert validate_column_exists("col") == (False, EMPTY_MESSAGE)
