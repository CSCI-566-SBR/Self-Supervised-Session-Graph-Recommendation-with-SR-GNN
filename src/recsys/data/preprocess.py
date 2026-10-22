"""Create clean, model-ready SR-GNN artifacts from raw click logs."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .artifacts import save_pickle
from .processed import load_processed_dataset, sha256_file

LOGGER = logging.getLogger("recsys.data.preprocess")
DEFAULT_SPLIT_DAYS = {"diginetica": (7, 7), "retailrocket": (14, 14)}


def configure_logging(output_dir: Path) -> None:
    """Log every preprocessing run both to stderr and preprocessing.log."""
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    for existing_handler in LOGGER.handlers:
        existing_handler.close()
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(output_dir / "preprocessing.log", mode="a"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def _find_column(frame: pd.DataFrame, candidates: list[str]) -> str:
    normalized = {column.lower().replace("_", ""): column for column in frame.columns}
    for candidate in candidates:
        key = candidate.lower().replace("_", "")
        if key in normalized:
            return normalized[key]
    raise ValueError(f"expected one of {candidates}; found {list(frame.columns)}")


def load_diginetica(path: Path) -> pd.DataFrame:
    """Normalize the CIKM Diginetica view log into session/item/timestamp rows."""
    raw = pd.read_csv(path, sep=None, engine="python")
    session_column = _find_column(raw, ["sessionId", "session_id"])
    item_column = _find_column(raw, ["itemId", "item_id"])
    date_column = _find_column(raw, ["eventdate", "event_date", "date"])
    timestamps = pd.to_datetime(raw[date_column], errors="raise")
    try:
        offset_column = _find_column(raw, ["timeframe", "time_frame"])
        timestamps += pd.to_timedelta(pd.to_numeric(raw[offset_column]), unit="ms")
    except ValueError:
        pass
    frame = pd.DataFrame(
        {
            "session": raw[session_column].astype(str),
            "item": raw[item_column].astype(str),
            "timestamp": timestamps,
            "row_order": range(len(raw)),
        }
    )
    frame.attrs["source_counts"] = {
        "interactions": int(len(raw)),
        "sessions": int(raw[session_column].nunique()),
        "items": int(raw[item_column].nunique()),
    }
    return frame


def load_retailrocket(path: Path, inactivity_minutes: int = 30) -> pd.DataFrame:
    """Keep RetailRocket views and derive sessions using an inactivity boundary."""
    raw = pd.read_csv(path)
    source_interactions = len(raw)
    event_column = _find_column(raw, ["event"])
    raw = raw[raw[event_column].astype(str).str.lower().eq("view")].copy()
    user_column = _find_column(raw, ["visitorid", "visitor_id", "user"])
    item_column = _find_column(raw, ["itemid", "item_id"])
    timestamp_column = _find_column(raw, ["timestamp", "time"])
    raw["timestamp_parsed"] = pd.to_datetime(
        pd.to_numeric(raw[timestamp_column], errors="raise"), unit="ms", errors="raise"
    )
    raw["row_order"] = range(len(raw))
    raw = raw.sort_values([user_column, "timestamp_parsed", "row_order"], kind="stable")
    new_session = raw.groupby(user_column, sort=False)["timestamp_parsed"].diff().gt(
        pd.Timedelta(minutes=inactivity_minutes)
    )
    session_number = new_session.groupby(raw[user_column], sort=False).cumsum().astype(int)
    frame = pd.DataFrame(
        {
            "session": raw[user_column].astype(str) + ":" + session_number.astype(str),
            "item": raw[item_column].astype(str),
            "timestamp": raw["timestamp_parsed"],
            "row_order": raw["row_order"],
        }
    )
    frame.attrs["source_counts"] = {
        "all_events": int(source_interactions),
        "view_events": int(len(raw)),
        "non_view_events_dropped": int(source_interactions - len(raw)),
        "users_with_views": int(raw[user_column].nunique()),
        "items_with_views": int(raw[item_column].nunique()),
    }
    return frame


def fixed_point_filter(
    frame: pd.DataFrame, min_item_frequency: int = 5, min_session_length: int = 2
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Recursively remove short sessions and infrequent items until stable."""
    if min_item_frequency < 1 or min_session_length < 2:
        raise ValueError("minimum item frequency must be >= 1 and session length >= 2")
    current = frame.copy()
    initial_sessions = current["session"].nunique()
    initial_items = set(current["item"].unique())
    initial_interactions = len(current)
    iterations = 0
    while True:
        iterations += 1
        before_index = current.index
        current = current[
            current.groupby("session")["item"].transform("size") >= min_session_length
        ]
        frequencies = current["item"].value_counts()
        current = current[current["item"].isin(frequencies[frequencies >= min_item_frequency].index)]
        if current.index.equals(before_index):
            break
    return current.copy(), {
        "iterations": iterations,
        "sessions_dropped": int(initial_sessions - current["session"].nunique()),
        "items_dropped": len(initial_items - set(current["item"].unique())),
        "interactions_dropped": int(initial_interactions - len(current)),
    }


def chronological_split(
    frame: pd.DataFrame, validation_days: int, test_days: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Split complete sessions by their final event timestamp."""
    if validation_days < 1 or test_days < 1:
        raise ValueError("validation_days and test_days must both be at least 1")
    session_ends = frame.groupby("session")["timestamp"].max()
    test_start = session_ends.max() - pd.Timedelta(days=test_days)
    validation_start = test_start - pd.Timedelta(days=validation_days)
    train_ids = set(session_ends[session_ends < validation_start].index)
    validation_ids = set(
        session_ends[(session_ends >= validation_start) & (session_ends < test_start)].index
    )
    test_ids = set(session_ends[session_ends >= test_start].index)
    splits = tuple(
        frame[frame["session"].isin(ids)].copy()
        for ids in (train_ids, validation_ids, test_ids)
    )
    if any(split.empty for split in splits):
        raise ValueError(
            "chronological split produced an empty train, validation, or test set; "
            "adjust the validation/test windows"
        )
    return (*splits, validation_start, test_start)


def apply_training_vocabulary(
    frame: pd.DataFrame, training_items: set[str], min_session_length: int = 2
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop unknown events, then discard sessions that become too short."""
    initial_interactions = len(frame)
    initial_sessions = frame["session"].nunique()
    known = frame["item"].isin(training_items)
    restricted = frame[known].copy()
    restricted = restricted[
        restricted.groupby("session")["item"].transform("size") >= min_session_length
    ].copy()
    return restricted, {
        "unknown_interactions_dropped": int((~known).sum()),
        "sessions_dropped_after_unknown_filter": int(
            initial_sessions - restricted["session"].nunique()
        ),
        "total_interactions_dropped": int(initial_interactions - len(restricted)),
    }


def ordered_sessions(frame: pd.DataFrame) -> list[list[str]]:
    """Return deterministic event-ordered sessions in chronological session order."""
    ordered = frame.sort_values(["timestamp", "session", "row_order"], kind="stable")
    sessions = [
        (session_id, group["timestamp"].max(), group["item"].tolist())
        for session_id, group in ordered.groupby("session", sort=False)
    ]
    sessions.sort(key=lambda entry: (entry[1], str(entry[0])))
    return [items for _, _, items in sessions]


def expand_prefixes(sessions: list[list[int]]) -> tuple[list[list[int]], list[int]]:
    """Expand each session into all non-empty prefix/next-item examples."""
    prefixes: list[list[int]] = []
    targets: list[int] = []
    for session in sessions:
        for position in range(1, len(session)):
            prefixes.append(session[:position])
            targets.append(session[position])
    return prefixes, targets


def _mapped_sessions(frame: pd.DataFrame, mapping: dict[str, int]) -> list[list[int]]:
    return [[mapping[item] for item in session] for session in ordered_sessions(frame)]


def preprocess_clean(
    dataset: str,
    input_path: Path,
    output_dir: Path,
    *,
    min_item_frequency: int = 5,
    min_session_length: int = 2,
    validation_days: int | None = None,
    test_days: int | None = None,
) -> dict[str, Any]:
    """Run raw parsing through clean artifact creation and contract validation."""
    if dataset not in DEFAULT_SPLIT_DAYS:
        raise ValueError(f"unsupported dataset: {dataset}")
    default_validation_days, default_test_days = DEFAULT_SPLIT_DAYS[dataset]
    validation_days = default_validation_days if validation_days is None else validation_days
    test_days = default_test_days if test_days is None else test_days
    configure_logging(output_dir)
    LOGGER.info("START clean dataset=%s input=%s", dataset, input_path)
    frame = load_diginetica(input_path) if dataset == "diginetica" else load_retailrocket(input_path)
    source_counts = dict(frame.attrs.get("source_counts", {}))
    raw_counts = {
        "sessions": int(frame["session"].nunique()),
        "interactions": int(len(frame)),
        "items": int(frame["item"].nunique()),
    }
    LOGGER.info(
        "source_counts %s", " ".join(f"{key}={value}" for key, value in source_counts.items())
    )
    LOGGER.info(
        "normalized_counts %s", " ".join(f"{key}={value}" for key, value in raw_counts.items())
    )
    filtered, filtering = fixed_point_filter(
        frame,
        min_item_frequency=min_item_frequency,
        min_session_length=min_session_length,
    )
    LOGGER.info("fixed_point_filter %s", " ".join(f"{key}={value}" for key, value in filtering.items()))
    train_frame, validation_frame, test_frame, validation_start, test_start = chronological_split(
        filtered, validation_days, test_days
    )
    training_items = set(train_frame["item"].unique())
    validation_frame, validation_oov = apply_training_vocabulary(
        validation_frame, training_items, min_session_length
    )
    test_frame, test_oov = apply_training_vocabulary(
        test_frame, training_items, min_session_length
    )
    if validation_frame.empty or test_frame.empty:
        raise ValueError("validation or test became empty after applying training vocabulary")
    LOGGER.info(
        "validation_unknown_filter %s",
        " ".join(f"{key}={value}" for key, value in validation_oov.items()),
    )
    LOGGER.info(
        "test_unknown_filter %s", " ".join(f"{key}={value}" for key, value in test_oov.items())
    )
    item_mapping = {
        item: index for index, item in enumerate(sorted(training_items, key=str), start=1)
    }
    session_frames = {"train": train_frame, "validation": validation_frame, "test": test_frame}
    session_data = {
        name: _mapped_sessions(split_frame, item_mapping)
        for name, split_frame in session_frames.items()
    }
    expanded = {name: expand_prefixes(sessions) for name, sessions in session_data.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, artifact in expanded.items():
        save_pickle(artifact, output_dir / f"{name}.txt")
    save_pickle(item_mapping, output_dir / "item_mapping.pkl")
    counts = {
        **{f"{name}_sessions": len(sessions) for name, sessions in session_data.items()},
        **{f"{name}_examples": len(artifact[1]) for name, artifact in expanded.items()},
    }
    summary: dict[str, Any] = {
        "dataset": dataset,
        "protocol_version": f"{dataset}_clean_v1",
        "num_items": len(item_mapping),
        "split_strategy": {
            "type": "chronological_session_end",
            "validation_definition": f"{validation_days} days immediately before test",
            "test_definition": f"final {test_days} days",
            "validation_start": validation_start.isoformat(),
            "test_start": test_start.isoformat(),
        },
        "filtering_protocol": (
            "recursive session-length then item-frequency filtering before splitting; "
            "unknown validation/test events removed after training-only mapping"
        ),
        "minimum_item_frequency": min_item_frequency,
        "minimum_session_length": min_session_length,
        "maximum_prefix_length": None,
        "item_mapping_scope": "training_split_only",
        "unknown_item_policy": (
            "drop unknown validation/test events, then drop sessions shorter than two"
        ),
        "prefix_target_generation": "all_nonempty_prefixes",
        "raw_source": {
            "file": input_path.name,
            "sha256": sha256_file(input_path),
        },
        "source_counts": source_counts,
        "raw_counts": raw_counts,
        "filtering_counts": filtering,
        "validation_unknown_filter_counts": validation_oov,
        "test_unknown_filter_counts": test_oov,
        "counts": counts,
    }
    with (output_dir / "preprocessing_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    loaded = load_processed_dataset(output_dir)
    LOGGER.info(
        "artifact_counts train=%d validation=%d test=%d num_items=%d",
        len(loaded.train), len(loaded.validation), len(loaded.test), loaded.num_items,
    )
    LOGGER.info("END clean dataset=%s", dataset)
    return summary


# Backward-compatible name for callers of the first pipeline version.
preprocess = preprocess_clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DEFAULT_SPLIT_DAYS))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-item-frequency", type=int, default=5)
    parser.add_argument("--min-session-length", type=int, default=2)
    parser.add_argument("--validation-days", type=int)
    parser.add_argument("--test-days", type=int)
    args = parser.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"raw file not found: {args.input}")
    preprocess_clean(
        args.dataset,
        args.input,
        args.output_dir,
        min_item_frequency=args.min_item_frequency,
        min_session_length=args.min_session_length,
        validation_days=args.validation_days,
        test_days=args.test_days,
    )


if __name__ == "__main__":
    main()
