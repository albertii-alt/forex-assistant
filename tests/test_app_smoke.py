from datetime import date

import pytest
from streamlit.testing.v1 import AppTest


APP_PATH = "../src/app/main.py"


def load_app() -> AppTest:
    app = AppTest.from_file(APP_PATH, default_timeout=60)
    app.run()
    assert not app.exception
    return app


def checkbox(app: AppTest, label: str):
    matches = [item for item in app.checkbox if item.label == label]
    assert len(matches) == 1
    return matches[0]


def test_app_default_settings_has_no_exception() -> None:
    load_app()


@pytest.mark.parametrize("timeframe", ["15m", "1h", "4h", "1D"])
def test_fvg_overlay_runs_for_each_timeframe(timeframe: str) -> None:
    app = load_app()
    app.selectbox[0].select(timeframe).run()
    assert not app.exception

    checkbox(app, "Show fair value gaps").set_value(True)
    app.run()
    assert not app.exception


@pytest.mark.parametrize("timeframe", ["15m", "1h", "4h", "1D"])
def test_swings_and_fvg_overlays_run_together(timeframe: str) -> None:
    app = load_app()
    app.selectbox[0].select(timeframe).run()
    assert not app.exception

    checkbox(app, "Show swing highs/lows").set_value(True)
    checkbox(app, "Show fair value gaps").set_value(True)
    app.run()
    assert not app.exception


@pytest.mark.parametrize("timeframe", ["15m", "1h", "4h", "1D"])
def test_structure_overlay_runs_for_each_timeframe(timeframe: str) -> None:
    app = load_app()
    app.selectbox[0].select(timeframe).run()
    assert not app.exception

    checkbox(app, "Show structure breaks").set_value(True)
    app.run()
    assert not app.exception


@pytest.mark.parametrize("timeframe", ["15m", "1h", "4h", "1D"])
def test_all_three_overlays_run_together(timeframe: str) -> None:
    app = load_app()
    app.selectbox[0].select(timeframe).run()
    assert not app.exception

    checkbox(app, "Show swing highs/lows").set_value(True)
    checkbox(app, "Show fair value gaps").set_value(True)
    checkbox(app, "Show structure breaks").set_value(True)
    app.run()
    assert not app.exception


def test_single_date_and_no_data_ranges() -> None:
    app = load_app()
    date_picker = app.date_input[0]

    try:
        date_picker.set_value(date(2026, 10, 30))
        app.run()
        assert not app.exception

        date_picker.set_value((date(2024, 1, 6), date(2024, 1, 6)))
        app.run()
        assert not app.exception
    except Exception as exc:
        pytest.fail(f"AppTest could not drive the date-range widget: {exc}")
