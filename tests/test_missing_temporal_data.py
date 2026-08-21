from pathlib import Path

import __main__ as main


def test_check_if_new_data_returns_true_when_csv_is_missing(tmp_path):
    missing_path = tmp_path / "nested" / "temporal_data_seasonal_df.csv"

    assert main.check_if_new_data(str(missing_path)) is True
