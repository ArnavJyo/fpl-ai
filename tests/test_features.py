"""Check rolling-window timing, player isolation, and missing-week behavior."""

from __future__ import annotations

import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from src.features.build_rolling import build_rolling_features


class RollingFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.history = pd.DataFrame({
            "element_id": [1, 1, 1, 1, 1, 1],
            "gameweek": [1, 2, 3, 4, 5, 6],
            "total_points": [2, 13, 9, 9, 6, 0],
            "minutes": [90, 90, 90, 90, 90, 0],
        })

    def build(self, history: pd.DataFrame) -> pd.DataFrame:
        return build_rolling_features(history, windows=[3, 5], stats=["total_points", "minutes"])

    def test_averages_use_previous_weeks_and_match_target_week(self) -> None:
        features = self.build(self.history)
        self.assertEqual(features.loc[3, "total_points"], 9)
        self.assertEqual(features.loc[3, "total_points_mean_last_3"], 8)
        self.assertEqual(features.loc[5, "total_points_mean_last_3"], 8)
        self.assertEqual(features.loc[5, "total_points_mean_last_5"], 7.8)
        self.assertEqual(features.loc[5, "minutes_mean_last_3"], 90)

    def test_changing_current_and_future_outcomes_cannot_change_earlier_inputs(self) -> None:
        changed = self.history.copy()
        changed.loc[changed.gameweek >= 4, ["total_points", "minutes"]] = 9999
        before = self.build(self.history).drop(columns="total_points")
        after = self.build(changed).drop(columns="total_points")
        assert_frame_equal(before.iloc[:4], after.iloc[:4])
        assert_frame_equal(self.build(self.history.iloc[:4]), self.build(self.history).iloc[:4])

    def test_unsorted_players_are_isolated_and_input_is_unchanged(self) -> None:
        other = self.history.assign(element_id=2, total_points=100, minutes=0)
        mixed = pd.concat([self.history, other], ignore_index=True).sample(frac=1, random_state=7)
        original = mixed.copy(deep=True)
        features = self.build(mixed)
        assert_frame_equal(mixed, original)
        assert_frame_equal(features.iloc[:6].reset_index(drop=True), self.build(self.history))
        self.assertEqual(features.iloc[9].total_points_mean_last_3, 100)

    def test_new_players_have_unknown_history_and_real_zeros_are_kept(self) -> None:
        late = pd.DataFrame({
            "element_id": [2, 2, 2], "gameweek": [4, 5, 6],
            "total_points": [0, 6, 2], "minutes": [0, 90, 90],
        })
        features = self.build(late)
        self.assertTrue(pd.isna(features.loc[0, "total_points_mean_last_3"]))
        self.assertEqual(features.history_count_last_3.tolist(), [0, 1, 2])
        self.assertEqual(features.loc[1, "total_points_mean_last_3"], 0)
        self.assertEqual(features.loc[2, "total_points_mean_last_3"], 3)

    def test_missing_weeks_do_not_extend_windows_or_become_zero(self) -> None:
        history = self.history[self.history.gameweek.isin([1, 2, 5, 6])]
        features = self.build(history)
        self.assertEqual(features.loc[2, "total_points_mean_last_3"], 13)
        self.assertEqual(features.loc[2, "history_count_last_3"], 1)
        self.assertEqual(features.loc[3, "total_points_mean_last_3"], 6)
        self.assertEqual(features.loc[3, "history_count_last_3"], 1)
        self.assertEqual(features.loc[3, "total_points_mean_last_5"], 7)

    def test_duplicate_player_weeks_are_rejected(self) -> None:
        duplicated = pd.concat([self.history, self.history.iloc[:1]])
        with self.assertRaisesRegex(ValueError, "one row per player per gameweek"):
            self.build(duplicated)

    def test_invalid_inputs_fail_with_a_clear_error(self) -> None:
        for windows in ([], [0], [3, 3], [True], [1.5]):
            with self.subTest(windows=windows), self.assertRaises(ValueError):
                build_rolling_features(self.history, windows, ["total_points"])
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            build_rolling_features(self.history, [3], ["expected_goals"])
        for value in (float("nan"), float("inf")):
            bad = self.history.astype({"minutes": "float64"})
            bad.loc[0, "minutes"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.build(bad)


if __name__ == "__main__":
    unittest.main()
