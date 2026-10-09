"""Tests for pages/data_input.py rendering, especially on a fresh session."""

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest


def _render_page():
    from core.state import init_state
    from pages.data_input import render

    init_state()
    render()


@pytest.fixture(autouse=True)
def clean_state():
    yield


def test_fresh_session_renders_data_editor():
    """A brand-new session has no 'df' in session_state, so get_df() lazily
    creates an all-NaN scaffold DataFrame. The page must still render the
    data editor so the user can enter data manually, instead of blocking
    behind a validation error that can never be cleared from this page.
    """
    at = AppTest.from_function(_render_page)
    at.run()

    assert at.exception == []
    assert at.get_by_key("data_editor") is not None


def test_fresh_session_shows_non_blocking_hint():
    at = AppTest.from_function(_render_page)
    at.run()

    assert len(at.info) >= 1
    assert len(at.warning) == 0


def test_all_rows_deleted_still_renders_data_editor():
    """st.data_editor's num_rows="dynamic" lets a user delete every row in
    the UI, leaving a 0-row DataFrame with columns intact. The page must
    keep rendering the editor so rows can be added back; only a DataFrame
    with zero columns should block rendering (st.columns(0) raises).
    """
    at = AppTest.from_function(_render_page)
    at.session_state["df"] = pd.DataFrame(columns=["Var1", "Var2"])
    at.run()

    assert at.exception == []
    assert at.get_by_key("data_editor") is not None
