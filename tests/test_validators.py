"""Tests for the empty/all-NaN DataFrame guard in core/validators.py."""

import pandas as pd
import pytest
import streamlit as st

from core.validators import (
    validate_categorical,
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


def test_validate_categorical_accepts_string_values():
    set_df(pd.DataFrame({"col": ["red", "blue", "green", "red", "blue", "green"]}))
    assert validate_categorical("col") == (True, None)


def test_validate_categorical_rejects_too_few_values():
    set_df(pd.DataFrame({"col": ["red", None]}))
    ok, msg = validate_categorical("col")
    assert ok is False
    assert "at least 2 valid values" in msg


def test_validate_categorical_accepts_single_category_with_enough_observations():
    set_df(pd.DataFrame({"col": ["red", "red", "red"]}))
    assert validate_categorical("col") == (True, None)


def test_validate_categorical_rejects_undersized_category():
    set_df(pd.DataFrame({"col": ["red", "red", "blue"]}))
    ok, msg = validate_categorical("col", min_category_size=2)
    assert ok is False
    assert "1 category with fewer than 2 observations" in msg
    assert "blue" in msg


def test_validate_categorical_accepts_all_categories_meeting_min_size():
    set_df(pd.DataFrame({"col": ["red", "red", "blue", "blue"]}))
    assert validate_categorical("col", min_category_size=2) == (True, None)


def test_validate_categorical_rejects_below_min_sample_size():
    set_df(pd.DataFrame({"col": ["red", "red", "blue", "blue"]}))
    ok, msg = validate_categorical("col", min_sample_size=10)
    assert ok is False
    assert "at least 10 observations, found 4" in msg


def test_validate_categorical_accepts_at_min_sample_size():
    set_df(pd.DataFrame({"col": ["red", "red", "blue", "blue"]}))
    assert validate_categorical("col", min_sample_size=4) == (True, None)


def test_validate_categorical_min_sample_size_message_takes_precedence_over_floor():
    set_df(pd.DataFrame({"col": ["red"]}))
    ok, msg = validate_categorical("col", min_sample_size=30)
    assert ok is False
    assert "at least 30 observations, found 1" in msg
