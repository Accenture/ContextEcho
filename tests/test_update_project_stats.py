from scripts.update_project_stats import roll_download_total, update_stats


def test_download_rollup_adds_new_month_bucket_even_when_lower():
    current = {
        "dataset_total_downloads": 66_000,
        "dataset_historical_downloads": 39_000,
        "dataset_hf_monthly_downloads": {"2026-06": 27_000},
    }

    total, buckets = roll_download_total(current, 5_002, "2026-07")

    assert buckets == {"2026-06": 27_000, "2026-07": 5_002}
    assert total == 71_002


def test_download_rollup_updates_same_month_to_max_without_subtracting():
    current = {
        "dataset_historical_downloads": 39_000,
        "dataset_hf_monthly_downloads": {"2026-07": 5_100},
    }

    total, buckets = roll_download_total(current, 5_002, "2026-07")

    assert buckets == {"2026-07": 5_100}
    assert total == 44_100


def test_update_stats_migrates_previous_snapshot_to_prior_month_bucket():
    current = {
        "dataset_total_downloads": 47_350,
        "dataset_total_downloads_updated": "2026-07-06",
        "dataset_historical_downloads": 39_000,
        "dataset_hf_downloads_last_month": 5_029,
        "dataset_hf_downloads_last_month_previous": 8_350,
    }

    updated = update_stats(current, {"downloads": 5_029, "likes": 5}, "2026-07-06")

    assert updated["dataset_total_downloads"] == 52_379
    assert updated["dataset_hf_monthly_downloads"] == {"2026-06": 8_350, "2026-07": 5_029}
    assert updated["dataset_hf_downloads_last_month"] == 5_029
    assert updated["dataset_hf_downloads_last_month_period"] == "2026-07"
