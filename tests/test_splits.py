import pandas as pd
import pytest

from src.data.splits import development_set, holdout_set


def make_df(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({"value": range(len(index))}, index=index)


def test_boundary_no_overlap_and_nothing_lost() -> None:
    index = pd.DatetimeIndex(
        [
            "2024-10-31 23:30",
            "2024-10-31 23:45",
            "2024-11-01 00:00",
            "2024-11-01 00:15",
        ],
        tz="UTC",
    )
    df = make_df(index)
    development = development_set(df)
    holdout = holdout_set(df)

    assert development.index.tolist() == [index[0], index[1]]
    assert holdout.index.tolist() == [index[2], index[3]]
    assert development.index.intersection(holdout.index).empty
    pd.testing.assert_frame_equal(
        pd.concat([development, holdout]).sort_index(),
        df,
    )


def test_split_does_not_mutate_input() -> None:
    df = make_df(
        pd.date_range("2024-10-31 23:45", periods=2, freq="15min", tz="UTC")
    )
    original = df.copy(deep=True)

    development = development_set(df)
    holdout = holdout_set(df)
    development.iloc[0, 0] = -1
    holdout.iloc[0, 0] = -1

    pd.testing.assert_frame_equal(df, original)


@pytest.mark.parametrize(
    "index",
    [
        pd.DatetimeIndex(["2024-10-31 23:45"]),
        pd.DatetimeIndex(["2024-10-31 23:45"], tz="America/New_York"),
    ],
)
def test_naive_or_non_utc_index_raises(index: pd.DatetimeIndex) -> None:
    df = make_df(index)

    with pytest.raises(ValueError, match="UTC"):
        development_set(df)
    with pytest.raises(ValueError, match="UTC"):
        holdout_set(df)
