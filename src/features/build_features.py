"""Assemble identical historical and forecast feature schemas."""

from __future__ import annotations

import pandas as pd

from src.features.build_rolling import build_rolling_features
from src.features.context import KEY, build_fixture_features

POSITIONS = ["GKP", "DEF", "MID", "FWD"]
STATUSES = ["a", "d", "i", "s", "u"]
SNAPSHOT_COLUMNS = [
    "price_tenths_snapshot", "form_snapshot", "ownership_percent_snapshot", "availability_pct",
    "is_penalty_taker", "is_corner_taker", "is_freekick_taker",
    *[f"position_{position}" for position in POSITIONS],
    *[f"status_{status}" for status in STATUSES],
]
MARKET_COLUMNS = [
    "price_tenths_prior", "ownership_rank_prior", "ownership_rank_change_prior",
    "transfers_in_rate_prior", "transfers_out_rate_prior", "transfer_net_rate_prior",
]


def player_snapshot(players: pd.DataFrame, gameweek: int) -> pd.DataFrame:
    """Encode today's attributes only for the week this snapshot forecasts."""
    if players.element_id.duplicated().any():
        raise ValueError("Player IDs must be unique.")
    if not players.position.isin(POSITIONS).all() or not players.status.isin(STATUSES).all():
        raise ValueError("Unknown player position or status.")
    result = players[["element_id"]].copy()
    result["gameweek"] = gameweek
    for source, destination in (
        ("price_tenths", "price_tenths_snapshot"), ("form", "form_snapshot"),
        ("selected_by_percent", "ownership_percent_snapshot"), ("availability_pct", "availability_pct"),
    ):
        result[destination] = players[source].astype(float)
    for column in ["is_penalty_taker", "is_corner_taker", "is_freekick_taker"]:
        result[column] = players[column].fillna(False).astype(float)
    for position in POSITIONS:
        result[f"position_{position}"] = (players.position == position).astype(float)
    for status in STATUSES:
        result[f"status_{status}"] = (players.status == status).astype(float)
    return result


def build_market_features(history: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Use the latest strictly earlier market observation, with its age."""
    market = history.loc[history.context_observed == 1, [*KEY, "price_tenths", "selected", "transfers_in", "transfers_out"]].copy()
    market = market.sort_values(KEY)
    market["ownership_rank_prior"] = market.groupby("gameweek").selected.rank(method="average", pct=True)
    market["ownership_rank_change_prior"] = market.groupby("element_id").ownership_rank_prior.diff()
    denominator = market.selected.clip(lower=1)
    market["transfers_in_rate_prior"] = market.transfers_in / denominator
    market["transfers_out_rate_prior"] = market.transfers_out / denominator
    market["transfer_net_rate_prior"] = (market.transfers_in - market.transfers_out) / denominator
    market = market.rename(columns={"price_tenths": "price_tenths_prior", "gameweek": "market_observed_gameweek"})
    query = rows[KEY].copy().assign(_row=range(len(rows)))
    query["gameweek"] = query.gameweek.astype("int64")
    market["market_observed_gameweek"] = market.market_observed_gameweek.astype("int64")
    merged = pd.merge_asof(
        query.sort_values("gameweek"),
        market[["element_id", "market_observed_gameweek", *MARKET_COLUMNS]].sort_values("market_observed_gameweek"),
        left_on="gameweek", right_on="market_observed_gameweek", by="element_id", allow_exact_matches=False,
    ).sort_values("_row")
    merged["market_age_gameweeks"] = merged.gameweek - merged.market_observed_gameweek
    result = merged[[*MARKET_COLUMNS, "market_age_gameweeks"]].reset_index(drop=True)
    result.index = rows.index
    return result


def assemble_features(
    history: pd.DataFrame,
    players: pd.DataFrame,
    fixtures: pd.DataFrame,
    team_results: pd.DataFrame,
    snapshots: pd.DataFrame,
    windows: list[int],
    stats: list[str],
    target: str,
    prediction_gameweek: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Reuse the same past-only builders for labelled and unlabelled rows."""
    if prediction_gameweek is not None and not history.empty and prediction_gameweek <= history.gameweek.max():
        raise ValueError("Prediction gameweek must be later than all historical outcomes.")
    if players.element_id.duplicated().any():
        raise ValueError("Player IDs must be unique.")
    history = history.copy()
    reliability = ["appearance_rate", "sixty_minutes_rate", "start_rate"]
    history["appearance_rate"] = (history.minutes > 0).astype(float)
    history["sixty_minutes_rate"] = (history.minutes >= 60).astype(float)
    history["start_rate"] = (history.starts > 0).astype(float)
    rolling_stats = list(dict.fromkeys([*stats, *reliability]))
    forecast = pd.DataFrame(columns=history.columns)
    if prediction_gameweek is not None:
        forecast = pd.DataFrame({
            "element_id": players.element_id.to_numpy(), "gameweek": prediction_gameweek,
            "own_team_id": players.team_id.to_numpy(),
        })
        for stat in dict.fromkeys([target, *rolling_stats]):
            forecast[stat] = 0
    combined = pd.concat([history, forecast], ignore_index=True) if not forecast.empty else history.copy()
    combined["element_id"] = combined.element_id.astype("int64")
    combined["gameweek"] = combined.gameweek.astype("int64")
    rolling = build_rolling_features(combined, windows, rolling_stats, target)
    rows = rolling[KEY].merge(combined[[*KEY, "own_team_id"]], on=KEY, validate="one_to_one")
    context = build_fixture_features(rows, fixtures, team_results, windows)
    market = build_market_features(history, rows)
    features = pd.concat([rolling, context, market], axis=1)
    if snapshots.duplicated(KEY).any():
        raise ValueError("Snapshots must have unique player/gameweek keys.")
    features = features.merge(snapshots[[*KEY, *SNAPSHOT_COLUMNS]], on=KEY, how="left", validate="one_to_one")
    feature_columns = [column for column in features.columns if column not in [*KEY, target]]
    labelled = features.merge(history[[*KEY]], on=KEY, how="inner", validate="one_to_one")
    predicted = features.merge(forecast[KEY], on=KEY, how="inner", validate="one_to_one") if not forecast.empty else features.iloc[:0]
    predicted = predicted.drop(columns=target)
    return labelled.reset_index(drop=True), predicted.reset_index(drop=True), feature_columns
