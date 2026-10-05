"""Preprocess Diginetica and RetailRocket into SR-GNN-style artifacts."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from .artifacts import save_pickle

LOGGER = logging.getLogger("recsys.data.preprocess")


def configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(output_dir / "preprocessing.log", mode="a")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    LOGGER.addHandler(stream_handler)


def _find_column(frame: pd.DataFrame, candidates: list[str]) -> str:
    normalized = {column.lower().replace("_", ""): column for column in frame.columns}
    for candidate in candidates:
        key = candidate.lower().replace("_", "")
        if key in normalized:
            return normalized[key]
    raise ValueError(f"expected one of {candidates}; found {list(frame.columns)}")


def load_diginetica(path: Path) -> pd.DataFrame:
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
    return pd.DataFrame(
        {
            "session": raw[session_column].astype(str),
            "item": raw[item_column].astype(str),
            "timestamp": timestamps,
        }
    ).sort_values(["session", "timestamp"], kind="stable")


def load_retailrocket(path: Path, inactivity_minutes: int = 30) -> pd.DataFrame:
    raw = pd.read_csv(path)
    event_column = _find_column(raw, ["event"])
    raw = raw[raw[event_column].astype(str).str.lower().eq("view")].copy()
    user_column = _find_column(raw, ["visitorid", "visitor_id", "user"])
    item_column = _find_column(raw, ["itemid", "item_id"])
    timestamp_column = _find_column(raw, ["timestamp", "time"])
    raw["timestamp_parsed"] = pd.to_datetime(
        pd.to_numeric(raw[timestamp_column], errors="raise"), unit="ms", errors="raise"
    )
    raw = raw.sort_values([user_column, "timestamp_parsed"], kind="stable")
    new_session = raw.groupby(user_column)["timestamp_parsed"].diff().gt(
        pd.Timedelta(minutes=inactivity_minutes)
    )
    number = new_session.groupby(raw[user_column]).cumsum().fillna(0).astype(int)
    return pd.DataFrame(
        {
            "session": raw[user_column].astype(str) + ":" + number.astype(str),
            "item": raw[item_column].astype(str),
            "timestamp": raw["timestamp_parsed"],
        }
    )


def fixed_point_filter(frame: pd.DataFrame, min_item_frequency: int = 5):
    current = frame.copy()
    initial_sessions = current["session"].nunique()
    initial_items = set(current["item"].unique())
    iterations = 0
    while True:
        iterations += 1
        before = len(current)
        current = current[current.groupby("session")["item"].transform("size") >= 2]
        frequencies = current["item"].value_counts()
        current = current[current["item"].isin(frequencies[frequencies >= min_item_frequency].index)]
        if len(current) == before:
            break
    current = current[current.groupby("session")["item"].transform("size") >= 2].copy()
    return current, {
        "filter_iterations": iterations,
        "sessions_dropped": int(initial_sessions - current["session"].nunique()),
        "unique_items_dropped": len(initial_items - set(current["item"].unique())),
        "interactions_dropped": len(frame) - len(current),
    }

def apply_training_vocabulary(frame: pd.DataFrame,training_items: set[str],minimum_session_length: int = 2,):
    initial_interactions = len(frame)
    initial_sessions = frame["session"].nunique()

    known_item_mask = frame["item"].isin(training_items)
    out_of_vocabulary_interactions = int((~known_item_mask).sum())

    restricted = frame[known_item_mask].copy()

    restricted = restricted[
        restricted.groupby("session")["item"].transform("size")
        >= minimum_session_length
    ].copy()

    statistics = {
        "oov_interactions_dropped": out_of_vocabulary_interactions,
        "short_sessions_dropped": int(
            initial_sessions - restricted["session"].nunique()
        ),
        "total_interactions_dropped": int(
            initial_interactions - len(restricted)
        ),
    }

    return restricted, statistics
    

# def chronological_split(frame: pd.DataFrame, test_days: int):
#     session_ends = frame.groupby("session")["timestamp"].max()
#     cutoff = session_ends.max() - pd.Timedelta(days=test_days)
#     test_ids = set(session_ends[session_ends >= cutoff].index)
#     train = frame[~frame["session"].isin(test_ids)].copy()
#     test = frame[frame["session"].isin(test_ids)].copy()
#     if train.empty or test.empty:
#         raise ValueError("chronological split produced an empty train or test set")
#     return train, test, cutoff

def chronological_split(frame: pd.DataFrame, validation_days: int, test_days: int,):
    if validation_days < 1:
        raise ValueError("validation_days must be at least 1")

    if test_days < 1:
        raise ValueError("test_days must be at least 1")

    session_ends = frame.groupby("session")["timestamp"].max()
    latest_timestamp = session_ends.max()
    test_start = latest_timestamp - pd.Timedelta(days=test_days)
    validation_start = test_start - pd.Timedelta(days=validation_days)
    train_ids = set(session_ends[session_ends < validation_start].index)
    validation_ids = set(session_ends[(session_ends >= validation_start)& (session_ends < test_start)].index)
    test_ids = set(session_ends[session_ends >= test_start].index)
    train = frame[frame["session"].isin(train_ids)].copy()
    validation = frame[frame["session"].isin(validation_ids)].copy()
    test = frame[frame["session"].isin(test_ids)].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError("chronological split produced an empty train, validation, or test set")

    return (train,validation,test,validation_start,test_start,)


def ordered_sessions(frame: pd.DataFrame) -> list[list[str]]:
    ordered = frame.sort_values(["session", "timestamp"], kind="stable")
    return ordered.groupby("session", sort=False)["item"].apply(list).tolist()


def expand_prefixes(sessions: list[list[int]]):
    sequences: list[list[int]] = []
    targets: list[int] = []
    for session in sessions:
        for position in range(1, len(session)):
            sequences.append(session[:position])
            targets.append(session[position])
    return sequences, targets


def preprocess(
    dataset: str, input_path: Path, output_dir: Path, min_item_frequency: int = 5, validation_days: int | None = None, test_days: int | None = None,
) -> dict:
    configure_logging(output_dir)
    LOGGER.info("START dataset=%s input=%s", dataset, input_path)
    if dataset == "diginetica":
        frame, default_validation_days, default_test_days = load_diginetica(input_path), 7, 7
    elif dataset == "retailrocket":
        frame, default_validation_days, default_test_days = load_retailrocket(input_path), 14, 14
    else:
        raise ValueError(f"unsupported dataset: {dataset}")

    if validation_days is None:
        validation_days = default_validation_days

    if test_days is None:
        test_days = default_test_days
    
    LOGGER.info(
        "raw sessions=%d interactions=%d items=%d",
        frame["session"].nunique(), len(frame), frame["item"].nunique(),
    )

    filtered, filtering = fixed_point_filter(frame,min_item_frequency=min_item_frequency,)
    LOGGER.info(
        "global_filtering %s",
        " ".join(
            f"{key}={value}"
            for key, value in filtering.items()
        ),
    )

    (
        train_frame,
        validation_frame,
        test_frame,
        validation_start,
        test_start,
    ) = chronological_split(
        filtered,
        validation_days=validation_days,
        test_days=test_days,
    )

    training_items = set(train_frame["item"].unique())

    validation_frame, validation_vocabulary_filtering = (
        apply_training_vocabulary(
            validation_frame,
            training_items,
        )
    )

    test_frame, test_vocabulary_filtering = (
        apply_training_vocabulary(
            test_frame,
            training_items,
        )
    )

    if validation_frame.empty:
        raise ValueError(
            "validation set became empty after restricting it to training items"
        )

    if test_frame.empty:
        raise ValueError(
            "test set became empty after restricting it to training items"
        )

    LOGGER.info(
        "validation_vocabulary_filtering %s",
        " ".join(
            f"{key}={value}"
            for key, value in validation_vocabulary_filtering.items()
        ),
    )

    LOGGER.info(
        "test_vocabulary_filtering %s",
        " ".join(
            f"{key}={value}"
            for key, value in test_vocabulary_filtering.items()
        ),
    )
    
    vocabulary = {
        item: index + 1 for index, item in enumerate(sorted(train_frame["item"].unique(), key=str))
    }
    train_sessions = [
        [vocabulary[item] for item in session] for session in ordered_sessions(train_frame)
    ]
    validation_sessions = [
            [vocabulary[item] for item in session] for session in ordered_sessions(validation_frame)
        ]
    test_sessions = [
        [vocabulary[item] for item in session] for session in ordered_sessions(test_frame)
    ]
    train_sequences, train_targets = expand_prefixes(train_sessions)
    validation_sequences, validation_targets = expand_prefixes(validation_sessions)
    test_sequences, test_targets = expand_prefixes(test_sessions)
    save_pickle((train_sequences, train_targets), output_dir / "train.txt")
    save_pickle((validation_sequences, validation_targets), output_dir / "validation.txt")
    save_pickle((test_sequences, test_targets), output_dir / "test.txt")
    save_pickle(train_sessions, output_dir / "all_train_seq.txt")
    save_pickle(vocabulary, output_dir / "item_mapping.pkl")
    summary = {
        "dataset": dataset,
        "input": str(input_path.resolve()),
        "raw_sessions": int(frame["session"].nunique()),
        "raw_interactions": len(frame),
        "raw_items": int(frame["item"].nunique()),
        # "split_cutoff": cutoff.isoformat(),
        "validation_start": validation_start.isoformat(),
        "test_start": test_start.isoformat(),
        "validation_days": validation_days,
        "test_days": test_days,
        "train_sessions": len(train_sessions),
        "validation_sessions": len(validation_sessions),
        "test_sessions": len(test_sessions),
        "train_examples": len(train_targets),
        "validation_examples": len(validation_targets),
        "test_examples": len(test_targets),
        "vocab_size": len(vocabulary),
        # **filtering,
        "min_item_frequency": min_item_frequency,
        "filtering_vocabulary_protocol": ("global_frequency_filter_then_training_vocabulary"),
        "filter_iterations": filtering["filter_iterations"],
        "sessions_dropped_by_global_filtering": filtering["sessions_dropped"],
        "unique_items_dropped_by_global_filtering": filtering["unique_items_dropped"],
        "interactions_dropped_by_global_filtering": filtering["interactions_dropped"],
        "validation_oov_interactions_dropped": (validation_vocabulary_filtering["oov_interactions_dropped"]),
        "validation_short_sessions_dropped": (validation_vocabulary_filtering["short_sessions_dropped"]),
        "validation_total_interactions_dropped": (validation_vocabulary_filtering["total_interactions_dropped"]),
        "test_oov_interactions_dropped": (test_vocabulary_filtering["oov_interactions_dropped"]),
        "test_short_sessions_dropped": (test_vocabulary_filtering["short_sessions_dropped"]),
        "test_total_interactions_dropped": (test_vocabulary_filtering["total_interactions_dropped"]),
    }
    
    (output_dir / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # LOGGER.info(
    #     "split cutoff=%s train_sessions=%d test_sessions=%d train_examples=%d "
    #     "test_examples=%d vocab_size=%d",
    #     cutoff.isoformat(), len(train_sessions), len(test_sessions), len(train_targets),
    #     len(test_targets), len(vocabulary),
    # )
    LOGGER.info(
        "split validation_start=%s test_start=%s "
        "train_sessions=%d validation_sessions=%d test_sessions=%d "
        "train_examples=%d validation_examples=%d test_examples=%d "
        "vocab_size=%d",
        validation_start.isoformat(),
        test_start.isoformat(),
        len(train_sessions),
        len(validation_sessions),
        len(test_sessions),
        len(train_targets),
        len(validation_targets),
        len(test_targets),
        len(vocabulary),
    )
    LOGGER.info("END dataset=%s", dataset)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["diginetica", "retailrocket"])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-item-frequency", type=int, default=5)
    parser.add_argument("--validation-days",type=int,default=None,help="number of days immediately before the test period",)
    parser.add_argument("--test-days",type=int,default=None,help="number of final days assigned to testing",)
    args = parser.parse_args()
    if not args.input.is_file():
        raise SystemExit(f"raw file not found: {args.input}")
    preprocess(dataset=args.dataset,input_path=args.input,output_dir=args.output_dir,min_item_frequency=args.min_item_frequency,validation_days=args.validation_days,test_days=args.test_days,)

if __name__ == "__main__":
    main()
