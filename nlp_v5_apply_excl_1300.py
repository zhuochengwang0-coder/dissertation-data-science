"""
Apply the frozen V5 six-factor NLP classifier to the 98,173-row dataset that
excludes the 1,000-row V5 development set and 300-row fixed evaluation set.

PyCharm setup
-------------
1. Put these five files in the same project folder:
   - nlp_v5_apply_excl_1300.py
   - nlp_v5_develop_1000_test_300.py
   - nlp_validation_sample_20260722_ai_labeled.csv
   - nlp_v3_independent_test_20260723_manual_labeled.csv
   - steam_reviews_v5_nlp_input_excl_v5_1300_20260724.csv
2. In PyCharm Terminal, run:
      pip install pandas==2.2.3 numpy==2.3.5 scipy==1.17.0 scikit-learn==1.8.0
3. Open this file and click Run.

Python 3.11 or newer is required. The pinned versions above are the exact
versions used for the verified full-data run.

The 300-row fixed evaluation labels are deliberately NOT used for fitting,
thresholding, or prediction. Its recommendation IDs are read only to prove
that those 300 rows are absent from the 98,173-row analysis input. Each TF-IDF
vectorizer and LinearSVC is fitted only on the frozen 1,000-row development
set. The 98,173 rows are transform/predict data only.

Main output
-----------
nlp_v5_excl_1300_results/steam_reviews_analysis_v5_excl_1300_20260724.csv

The main output preserves the original 25 columns and appends exactly:
combat, challenge, progression, exploration, narrative, immersion.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

try:
    import nlp_v5_develop_1000_test_300 as frozen_v5
except ImportError as exc:
    raise ImportError(
        "\nCannot import nlp_v5_develop_1000_test_300.py.\n"
        "Put the frozen V5 source file in the same folder as this script."
    ) from exc


# =========================================================
# 1. Fixed files, schema, and frozen-input expectations
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DEVELOPMENT_CANDIDATES = [
    "nlp_validation_sample_20260722_ai_labeled.csv",
    "nlp_validation_sample_20260722_ai_labeled(1).csv",
]

EVALUATION_ID_CANDIDATES = [
    "nlp_v3_independent_test_20260723_manual_labeled.csv",
    "nlp_v3_independent_test_20260723_manual_labeled(1).csv",
    "nlp_v3_independent_test_20260723_manual_labeled(2).csv",
]

FULL_INPUT_CANDIDATES = [
    "steam_reviews_v5_nlp_input_excl_v5_1300_20260724.csv",
    "steam_reviews_v5_nlp_input_excl_v5_1300_20260724(1).csv",
]

DEFAULT_OUTPUT_DIR = BASE_DIR / "nlp_v5_excl_1300_results"
DEFAULT_OUTPUT_NAME = "steam_reviews_analysis_v5_excl_1300_20260724.csv"
DEFAULT_SUMMARY_NAME = "v5_excl_1300_prediction_summary.csv"
DEFAULT_MANIFEST_NAME = "v5_excl_1300_run_manifest.json"
DEFAULT_AUDIT_NAME = "v5_excl_1300_prediction_audit.csv"

EXPECTED_FULL_ROWS = 98_173
DEFAULT_BATCH_SIZE = 5_000

EXPECTED_INPUT_COLUMNS = [
    "game",
    "appid",
    "recommendation_id",
    "language",
    "review",
    "voted_up",
    "playtime_at_review_hours",
    "playtime_forever_hours",
    "playtime_last_two_weeks_hours",
    "deck_playtime_at_review_hours",
    "timestamp_created",
    "timestamp_updated",
    "last_played",
    "votes_up",
    "votes_funny",
    "weighted_vote_score",
    "comment_count",
    "num_games_owned",
    "num_reviews_by_author",
    "steam_purchase",
    "received_for_free",
    "written_during_early_access",
    "primarily_steam_deck",
    "scraped_at_utc",
    "review_word_count",
]

EXPECTED_FULL_GAME_COUNTS = {
    "Cyberpunk 2077": 19_631,
    "Elden Ring": 19_667,
    "Hogwarts Legacy": 19_587,
    "Monster Hunter Wilds": 19_674,
    "The Witcher 3: Wild Hunt": 19_614,
}

EXPECTED_FACTORS = [
    "combat",
    "challenge",
    "progression",
    "exploration",
    "narrative",
    "immersion",
]

EXPECTED_MODEL_CONFIGS = {
    "combat": {
        "features": "char",
        "c": 2.0,
        "threshold": 0.26168627393741006,
    },
    "challenge": {
        "features": "char",
        "c": 2.0,
        "threshold": 0.12492285681498969,
    },
    "progression": {
        "features": "word",
        "c": 0.25,
        "threshold": 0.3029445220112233,
    },
    "exploration": {
        "features": "both",
        "c": 0.25,
        "threshold": 0.6961137993719319,
    },
    "narrative": {
        "features": "char",
        "c": 1.0,
        "threshold": 0.2249272283651293,
    },
    "immersion": {
        "features": "char",
        "c": 2.0,
        "threshold": 0.33202377681200534,
    },
}

# SHA-256 of canonical JSON containing FACTORS, MODEL_CONFIGS, ALL_RULES,
# and EXCLUDE_RULES from the validated V5 source. This detects accidental
# edits to the frozen classifier while ignoring harmless source formatting.
EXPECTED_FROZEN_FINGERPRINT = (
    "5819523b401e604d983fd20ccb688b582"
    "555e17599174bf7608e71f0241e4335"
)

# Exact byte-level hashes of the validated V5 source and the frozen 1,000-row
# development labels. Unlike the canonical configuration fingerprint above,
# these checks also detect edits to preprocessing/functions or training rows.
EXPECTED_FROZEN_SOURCE_SHA256 = (
    "2c467e563e4cd6ba4b9a784574387ff0"
    "b4c42bf458364388d3334e49f9ae233b"
)
EXPECTED_DEVELOPMENT_SHA256 = (
    "5b213cc1b5b2a467ae38dec68f8fc3cc"
    "7bb8df8893d91a6d8ec83df80f44d05f"
)
EXPECTED_EVALUATION_SHA256 = (
    "e732666ae6275014a016018941a5412c"
    "f2160e8323722fb5664c868054ee524c"
)

# The exact audited 98,173-row export has this byte-level hash. This check is
# intentionally strict because it proves that the already-audited file is used.
EXPECTED_FULL_INPUT_SHA256 = (
    "35b529c38bc57953c68d7c70fd4b050f"
    "2538170775ae01389080a28a1c70c926"
)

# These values were independently derived by filtering the earlier verified
# 99,473-row V5 output to the exact 98,173 IDs. They are output-integrity
# checks only and do not affect any prediction.
EXPECTED_OVERALL_MENTION_COUNTS = {
    "combat": 12_279,
    "challenge": 7_770,
    "progression": 5_979,
    "exploration": 6_382,
    "narrative": 17_087,
    "immersion": 4_942,
}


# =========================================================
# 2. General helpers and strict frozen-version checks
# =========================================================

def sha256_file(file_path: Path, block_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without loading a file into memory."""
    digest = hashlib.sha256()
    with file_path.open("rb") as file_handle:
        while True:
            block = file_handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_frozen_fingerprint() -> str:
    """Hash the exact rules, exclusions, factors, and model configuration."""
    payload = {
        "factors": frozen_v5.FACTORS,
        "configs": frozen_v5.MODEL_CONFIGS,
        "all_rules": frozen_v5.ALL_RULES,
        "exclude_rules": frozen_v5.EXCLUDE_RULES,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_frozen_v5() -> tuple[str, str]:
    """Stop if the imported V5 rules or settings differ from the tested V5."""
    if list(frozen_v5.FACTORS) != EXPECTED_FACTORS:
        raise RuntimeError(
            "The factor names/order in the imported V5 source have changed."
        )

    if frozen_v5.MODEL_CONFIGS != EXPECTED_MODEL_CONFIGS:
        raise RuntimeError(
            "MODEL_CONFIGS in the imported V5 source do not match the "
            "validated frozen configuration."
        )

    if frozen_v5.RANDOM_SEED != 20260723:
        raise RuntimeError(
            "RANDOM_SEED in the imported V5 source is not 20260723."
        )

    fingerprint = canonical_frozen_fingerprint()
    if fingerprint != EXPECTED_FROZEN_FINGERPRINT:
        raise RuntimeError(
            "\nThe imported V5 rules/exclusions/configuration have changed."
            f"\nExpected fingerprint: {EXPECTED_FROZEN_FINGERPRINT}"
            f"\nActual fingerprint:   {fingerprint}"
            "\nRestore the validated nlp_v5_develop_1000_test_300.py "
            "before applying it to the full dataset."
        )

    source_path = Path(frozen_v5.__file__).resolve()
    source_sha256 = sha256_file(source_path)
    if source_sha256 != EXPECTED_FROZEN_SOURCE_SHA256:
        raise RuntimeError(
            "\nThe imported V5 source file is not the exact validated file."
            f"\nExpected SHA-256: {EXPECTED_FROZEN_SOURCE_SHA256}"
            f"\nActual SHA-256:   {source_sha256}"
            "\nRestore the original nlp_v5_develop_1000_test_300.py "
            "without editing or resaving it."
        )

    print(
        "PASS: frozen V5 source, configuration, rules, and functions "
        "are unchanged."
    )
    return fingerprint, source_sha256


def resolve_file(
    explicit_path: Path | None,
    candidates: list[str],
    description: str,
) -> Path:
    """Resolve an explicit path or a known filename beside the script."""
    if explicit_path is not None:
        resolved = explicit_path.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(
                f"\nCannot find {description}:\n{resolved}"
            )
        return resolved

    search_folders = [BASE_DIR, BASE_DIR / "upload"]
    for folder in search_folders:
        for file_name in candidates:
            candidate = folder / file_name
            if candidate.is_file():
                return candidate.resolve()

    expected = "\n".join(
        f"  - {folder / name}"
        for folder in search_folders
        for name in candidates
    )
    raise FileNotFoundError(
        f"\nCannot find {description}. Checked:\n{expected}"
    )


# =========================================================
# 3. Read and validate the two inputs
# =========================================================

def read_full_csv_strictly(file_path: Path) -> pd.DataFrame:
    """
    Read the full CSV without treating literal reviews 'NA'/'N/A' as missing.

    UTF-8 decoding is strict: damaged bytes stop the run instead of being
    silently replaced.
    """
    read_options = {
        "encoding": "utf-8-sig",
        "keep_default_na": False,
        "na_values": [""],
        "low_memory": False,
        "engine": "c",
        "on_bad_lines": "error",
        "dtype": {"recommendation_id": "string"},
    }

    try:
        return pd.read_csv(
            file_path,
            encoding_errors="strict",
            **read_options,
        )
    except TypeError:
        # Compatibility path for an older pandas version.
        return pd.read_csv(file_path, **read_options)


def validate_full_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Verify the exact filtered V5 input without cleaning or filtering."""
    result = df.copy()
    result.columns = (
        result.columns.astype(str).str.strip().str.lstrip("\ufeff")
    )

    if list(result.columns) != EXPECTED_INPUT_COLUMNS:
        missing = [
            column
            for column in EXPECTED_INPUT_COLUMNS
            if column not in result.columns
        ]
        extra = [
            column
            for column in result.columns
            if column not in EXPECTED_INPUT_COLUMNS
        ]
        raise ValueError(
            "\nThe full-input columns/order do not match the validated "
            "25-column schema."
            f"\nMissing columns: {missing}"
            f"\nUnexpected columns: {extra}"
            f"\nActual order: {list(result.columns)}"
        )

    if len(result) != EXPECTED_FULL_ROWS:
        raise ValueError(
            f"Full input: expected {EXPECTED_FULL_ROWS:,} rows, "
            f"found {len(result):,}."
        )

    ids = result["recommendation_id"]
    invalid_id_mask = ids.isna() | ids.astype("string").str.strip().eq("")
    if invalid_id_mask.any():
        raise ValueError(
            "Full input contains blank recommendation_id values."
        )
    if ids.duplicated().any():
        duplicate_count = int(ids.duplicated(keep=False).sum())
        raise ValueError(
            "Full input contains duplicate recommendation_id values: "
            f"{duplicate_count:,} affected rows."
        )

    review_text = result["review"]
    blank_review_mask = (
        review_text.isna()
        | review_text.fillna("").astype(str).str.strip().eq("")
    )
    if blank_review_mask.any():
        raise ValueError(
            "Full input contains true blank reviews: "
            f"{int(blank_review_mask.sum()):,}."
        )

    non_english_mask = (
        result["language"].isna()
        | result["language"].fillna("").astype(str)
        .str.strip().str.lower().ne("english")
    )
    if non_english_mask.any():
        raise ValueError(
            "Full input contains non-English/blank language values: "
            f"{int(non_english_mask.sum()):,}."
        )

    actual_game_counts = (
        result["game"].value_counts().sort_index().to_dict()
    )
    expected_game_counts = dict(sorted(EXPECTED_FULL_GAME_COUNTS.items()))
    if actual_game_counts != expected_game_counts:
        raise ValueError(
            "\nThe five-game distribution differs from the validated "
            "V5 input."
            f"\nExpected: {expected_game_counts}"
            f"\nActual:   {actual_game_counts}"
        )

    print("\nPASS: full-input structure is valid.")
    print(f"  Rows: {len(result):,}")
    print(f"  Columns: {len(result.columns)}")
    print(f"  Unique recommendation IDs: {ids.nunique():,}")
    print("  True blank reviews: 0")
    print("  Non-English rows: 0")
    for game, count in actual_game_counts.items():
        print(f"  {game}: {count:,}")

    return result


def load_and_validate_development(file_path: Path) -> pd.DataFrame:
    """Use the original V5 validator for the frozen 1,000-row labels."""
    raw_development = frozen_v5.read_csv_safely(file_path)
    return frozen_v5.validate_dataset(
        raw_development,
        dataset_name="development_1000",
        expected_rows=1000,
        expected_game_counts=frozen_v5.EXPECTED_DEVELOPMENT_COUNTS,
    )


def load_and_validate_evaluation_ids(file_path: Path) -> pd.DataFrame:
    """
    Validate the frozen 300-row file before using only its IDs for exclusion QA.

    No evaluation label is passed to a vectorizer, classifier, rule, threshold,
    or full-data prediction function.
    """
    raw_evaluation = frozen_v5.read_csv_safely(file_path)
    return frozen_v5.validate_dataset(
        raw_evaluation,
        dataset_name="fixed_evaluation_300_ID_check_only",
        expected_rows=300,
        expected_game_counts=frozen_v5.EXPECTED_VALIDATION_COUNTS,
    )


# =========================================================
# 4. Frozen TF-IDF + LinearSVC, with batched transform only
# =========================================================

def make_vectorizers(feature_kind: str) -> list[TfidfVectorizer]:
    """Construct the exact frozen V5 vectorizer(s) for one factor."""
    vectorizers: list[TfidfVectorizer] = []

    if feature_kind in {"word", "both"}:
        vectorizers.append(
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.98,
                sublinear_tf=True,
                max_features=40000,
            )
        )

    if feature_kind in {"char", "both"}:
        vectorizers.append(
            TfidfVectorizer(
                analyzer="char_wb",
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(3, 5),
                min_df=2,
                sublinear_tf=True,
                max_features=80000,
            )
        )

    if not vectorizers:
        raise ValueError(f"Unknown frozen feature type: {feature_kind}")
    return vectorizers


def combine_sparse_parts(parts):
    """Return one sparse matrix, matching the original V5 hstack behavior."""
    if len(parts) == 1:
        return parts[0]
    return hstack(parts, format="csr")


def fit_development_and_score_full_in_batches(
    development_df: pd.DataFrame,
    full_df: pd.DataFrame,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """
    Fit only on the 1,000 labels, then transform/predict the full data in chunks.

    Chunking changes only peak memory use. No vectorizer or classifier is ever
    fitted on the 98,173 full-data rows.
    """
    development_text = (
        development_df["review"].fillna("").astype(str)
    )
    full_text = full_df["review"].fillna("").astype(str)
    all_scores: dict[str, np.ndarray] = {}

    for factor_number, factor in enumerate(
        frozen_v5.FACTORS, start=1
    ):
        config = frozen_v5.MODEL_CONFIGS[factor]
        labels = development_df[
            f"{factor}_manual"
        ].to_numpy(dtype=int)

        print(
            f"\n[{factor_number}/{len(frozen_v5.FACTORS)}] "
            f"Training frozen {factor} model on 1,000 rows..."
        )

        vectorizers = make_vectorizers(config["features"])
        training_parts = [
            vectorizer.fit_transform(development_text)
            for vectorizer in vectorizers
        ]
        x_train = combine_sparse_parts(training_parts)

        model = LinearSVC(
            C=config["c"],
            class_weight="balanced",
            random_state=frozen_v5.RANDOM_SEED,
        )
        model.fit(x_train, labels)

        factor_scores = np.empty(len(full_df), dtype=np.float64)
        total_batches = (len(full_df) + batch_size - 1) // batch_size

        for batch_number, start in enumerate(
            range(0, len(full_df), batch_size), start=1
        ):
            end = min(start + batch_size, len(full_df))
            batch_text = full_text.iloc[start:end]
            evaluation_parts = [
                vectorizer.transform(batch_text)
                for vectorizer in vectorizers
            ]
            x_batch = combine_sparse_parts(evaluation_parts)
            factor_scores[start:end] = model.decision_function(x_batch)

            print(
                f"\r  Predicting batch {batch_number}/{total_batches} "
                f"({end:,}/{len(full_df):,})",
                end="",
                flush=True,
            )

            del evaluation_parts, x_batch

        print()
        if not np.isfinite(factor_scores).all():
            raise RuntimeError(
                f"{factor}: non-finite LinearSVC scores were generated."
            )

        all_scores[factor] = factor_scores
        del vectorizers, training_parts, x_train, model, factor_scores
        gc.collect()

    return all_scores


# =========================================================
# 5. Frozen sentence rules + final hybrid prediction
# =========================================================

def apply_rules_and_hybrid_in_batches(
    full_df: pd.DataFrame,
    scores_by_factor: dict[str, np.ndarray],
    batch_size: int,
    audit_temp_path: Path | None,
) -> dict[str, np.ndarray]:
    """Apply frozen rules and combine rule OR ML predictions in row order."""
    final_predictions = {
        factor: np.empty(len(full_df), dtype=np.int8)
        for factor in frozen_v5.FACTORS
    }

    if audit_temp_path is not None and audit_temp_path.exists():
        audit_temp_path.unlink()

    total_batches = (len(full_df) + batch_size - 1) // batch_size
    print("\nApplying frozen sentence rules and hybrid decisions...")

    for batch_number, start in enumerate(
        range(0, len(full_df), batch_size), start=1
    ):
        end = min(start + batch_size, len(full_df))

        # Only these original columns are needed for rule prediction/auditing.
        base_batch = full_df.iloc[start:end][
            ["recommendation_id", "game", "review"]
        ].copy()
        rule_batch = frozen_v5.add_rule_predictions(base_batch)
        batch_scores = {
            factor: scores_by_factor[factor][start:end]
            for factor in frozen_v5.FACTORS
        }
        hybrid_batch = frozen_v5.apply_hybrid_predictions(
            rule_batch, batch_scores
        )

        for factor in frozen_v5.FACTORS:
            final_predictions[factor][start:end] = hybrid_batch[
                f"{factor}_mention"
            ].to_numpy(dtype=np.int8)

        if audit_temp_path is not None:
            audit_columns = [
                "recommendation_id",
                "game",
                "review",
            ]
            for factor in frozen_v5.FACTORS:
                audit_columns.extend(
                    [
                        f"{factor}_rule_mention",
                        f"{factor}_matched_terms",
                        f"{factor}_excluded_terms",
                        f"{factor}_ml_score",
                        f"{factor}_ml_threshold",
                        f"{factor}_ml_mention",
                        f"{factor}_mention",
                        f"{factor}_decision_source",
                    ]
                )

            hybrid_batch[audit_columns].to_csv(
                audit_temp_path,
                mode="w" if batch_number == 1 else "a",
                header=batch_number == 1,
                index=False,
                encoding="utf-8-sig" if batch_number == 1 else "utf-8",
                errors="strict",
                na_rep="",
                quoting=csv.QUOTE_MINIMAL,
                quotechar='"',
                doublequote=True,
                lineterminator="\n",
            )

        print(
            f"\r  Rule/hybrid batch {batch_number}/{total_batches} "
            f"({end:,}/{len(full_df):,})",
            end="",
            flush=True,
        )
        del base_batch, rule_batch, batch_scores, hybrid_batch
        gc.collect()

    print()
    return final_predictions


# =========================================================
# 6. Output, QA summary, and run manifest
# =========================================================

def build_prediction_summary(output_df: pd.DataFrame) -> pd.DataFrame:
    """Create overall and per-game mention counts without changing V5."""
    rows = []
    scopes = [("all_games", output_df)]
    scopes.extend(
        (
            game,
            output_df.loc[output_df["game"] == game],
        )
        for game in frozen_v5.EXPECTED_GAMES
    )

    for scope_name, scope_df in scopes:
        for factor in frozen_v5.FACTORS:
            mention_count = int(scope_df[factor].sum())
            rows.append(
                {
                    "scope": scope_name,
                    "factor": factor,
                    "total_reviews": len(scope_df),
                    "mention_count": mention_count,
                    "non_mention_count": len(scope_df) - mention_count,
                    "mention_percentage": round(
                        100.0 * mention_count / len(scope_df), 4
                    ),
                }
            )
    return pd.DataFrame(rows)


def validate_expected_mention_counts(summary_df: pd.DataFrame) -> None:
    """Confirm that the exact audited input reproduces the expected V5 totals."""
    overall = summary_df.loc[
        summary_df["scope"] == "all_games",
        ["factor", "mention_count"],
    ]
    actual_counts = dict(
        zip(
            overall["factor"],
            overall["mention_count"].astype(int),
        )
    )
    if actual_counts != EXPECTED_OVERALL_MENTION_COUNTS:
        raise RuntimeError(
            "\nThe V5 mention totals differ from the independently "
            "verified totals."
            f"\nExpected: {EXPECTED_OVERALL_MENTION_COUNTS}"
            f"\nActual:   {actual_counts}"
            "\nStop: do not use this output for regression."
        )
    print("PASS: all six overall mention counts match the verified V5 totals.")


def validate_output(
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
) -> None:
    """Verify row/ID order, original fields, and six binary outputs."""
    expected_output_columns = EXPECTED_INPUT_COLUMNS + frozen_v5.FACTORS
    if list(output_df.columns) != expected_output_columns:
        raise RuntimeError("Output columns/order are incorrect.")

    if len(output_df) != len(input_df):
        raise RuntimeError("Output row count differs from input row count.")

    if not output_df["recommendation_id"].equals(
        input_df["recommendation_id"]
    ):
        raise RuntimeError(
            "Output recommendation_id values/order differ from the input."
        )

    # The original columns come from a direct copy; this assertion guards
    # against accidental transformation during later maintenance.
    if not output_df[EXPECTED_INPUT_COLUMNS].equals(
        input_df[EXPECTED_INPUT_COLUMNS]
    ):
        raise RuntimeError("At least one original input field changed.")

    for factor in frozen_v5.FACTORS:
        if output_df[factor].isna().any():
            raise RuntimeError(f"{factor}: output contains missing values.")
        unique_values = set(output_df[factor].unique().tolist())
        if not unique_values.issubset({0, 1}):
            raise RuntimeError(
                f"{factor}: output contains non-binary values "
                f"{sorted(unique_values)}."
            )
        if unique_values != {0, 1}:
            raise RuntimeError(
                f"{factor}: every prediction is the same value. "
                "Stop and investigate implementation/data integrity."
            )

    print("\nPASS: final output QA completed.")
    print(f"  Rows preserved: {len(output_df):,}")
    print("  ID values/order preserved: Yes")
    print("  Original 25 fields unchanged in memory: Yes")
    print("  Six factor columns contain only 0/1: Yes")
    print("  Missing factor predictions: 0")


def write_csv_atomically(
    df: pd.DataFrame,
    final_path: Path,
    overwrite: bool,
) -> str:
    """Write a CSV through a temporary file, then publish it atomically."""
    if final_path.exists() and not overwrite:
        raise FileExistsError(
            f"\nOutput already exists:\n{final_path}\n"
            "Rename/delete it, or run with --overwrite."
        )

    temp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()

    try:
        df.to_csv(
            temp_path,
            index=False,
            encoding="utf-8-sig",
            errors="strict",
            na_rep="",
            quoting=csv.QUOTE_MINIMAL,
            quotechar='"',
            doublequote=True,
            lineterminator="\n",
        )
        temp_path.replace(final_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise

    return sha256_file(final_path)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the frozen six-factor NLP V5 to the validated "
            "98,173-row Steam review CSV after excluding V5 development "
            "and evaluation rows."
        )
    )
    parser.add_argument(
        "--development-file",
        type=Path,
        default=None,
        help="Optional explicit path to the frozen 1,000-row labelled CSV.",
    )
    parser.add_argument(
        "--evaluation-id-file",
        type=Path,
        default=None,
        help=(
            "Optional explicit path to the frozen 300-row evaluation CSV. "
            "Only its IDs are used to verify exclusion."
        ),
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Optional explicit path to the audited 98,173-row V5 input CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output folder (default: nlp_v5_full_results beside script).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Transform/rule batch size (default: 5000).",
    )
    parser.add_argument(
        "--save-audit",
        action="store_true",
        help=(
            "Also save rule matches, SVC margins, thresholds, and decision "
            "sources. The audit CSV is substantially larger."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing files in the output folder.",
    )
    return parser.parse_args()


# =========================================================
# 7. Main
# =========================================================

def main() -> None:
    args = parse_arguments()
    if args.batch_size < 100:
        raise ValueError("--batch-size must be at least 100.")

    started_at = time.time()
    print("=" * 78)
    print("FROZEN V5 FILTERED-DATA APPLICATION")
    print("=" * 78)
    print(
        "Training source: 1,000 labelled development reviews only\n"
        "Filtered data: transform/predict only\n"
        "300-row fixed evaluation labels: not used\n"
        "Development/evaluation IDs: exclusion verification only"
    )

    frozen_fingerprint, frozen_source_sha256 = verify_frozen_v5()

    development_path = resolve_file(
        args.development_file,
        DEVELOPMENT_CANDIDATES,
        "1,000-row labelled development CSV",
    )
    evaluation_id_path = resolve_file(
        args.evaluation_id_file,
        EVALUATION_ID_CANDIDATES,
        "300-row fixed evaluation CSV for ID exclusion QA",
    )
    full_input_path = resolve_file(
        args.input_file,
        FULL_INPUT_CANDIDATES,
        "98,173-row filtered V5 input CSV",
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    main_output_path = output_dir / DEFAULT_OUTPUT_NAME
    summary_path = output_dir / DEFAULT_SUMMARY_NAME
    manifest_path = output_dir / DEFAULT_MANIFEST_NAME
    audit_path = output_dir / DEFAULT_AUDIT_NAME
    audit_temp_path = (
        audit_path.with_suffix(audit_path.suffix + ".tmp")
        if args.save_audit
        else None
    )

    for existing_path in [
        main_output_path,
        summary_path,
        manifest_path,
        audit_path if args.save_audit else None,
    ]:
        if (
            existing_path is not None
            and existing_path.exists()
            and not args.overwrite
        ):
            raise FileExistsError(
                f"\nOutput already exists:\n{existing_path}\n"
                "Rename/delete the existing output, or run with --overwrite."
            )

    print(f"\nDevelopment file:\n{development_path}")
    development_sha256 = sha256_file(development_path)
    if development_sha256 != EXPECTED_DEVELOPMENT_SHA256:
        raise RuntimeError(
            "\nThe 1,000-row development file is not the exact frozen "
            "labelled dataset."
            f"\nExpected SHA-256: {EXPECTED_DEVELOPMENT_SHA256}"
            f"\nActual SHA-256:   {development_sha256}"
            "\nRestore nlp_validation_sample_20260722_ai_labeled.csv "
            "without editing or resaving it."
        )
    print("PASS: 1,000-row development-file SHA-256 is unchanged.")
    development_df = load_and_validate_development(development_path)

    print(f"\nEvaluation ID file (labels not used):\n{evaluation_id_path}")
    evaluation_sha256 = sha256_file(evaluation_id_path)
    if evaluation_sha256 != EXPECTED_EVALUATION_SHA256:
        raise RuntimeError(
            "\nThe 300-row evaluation file is not the exact frozen file."
            f"\nExpected SHA-256: {EXPECTED_EVALUATION_SHA256}"
            f"\nActual SHA-256:   {evaluation_sha256}"
        )
    print("PASS: 300-row evaluation-file SHA-256 is unchanged.")
    evaluation_df = load_and_validate_evaluation_ids(evaluation_id_path)

    print(f"\nFull-input file:\n{full_input_path}")
    input_sha256 = sha256_file(full_input_path)
    if input_sha256 != EXPECTED_FULL_INPUT_SHA256:
        raise RuntimeError(
            "\nThe input is not the exact audited 98,173-row export."
            f"\nExpected SHA-256: {EXPECTED_FULL_INPUT_SHA256}"
            f"\nActual SHA-256:   {input_sha256}"
            "\nUse steam_reviews_v5_nlp_input_excl_v5_1300_20260724.csv "
            "without editing or resaving it."
        )
    print("PASS: full-input SHA-256 matches the audited 98,173-row export.")

    full_df = validate_full_dataset(
        read_full_csv_strictly(full_input_path)
    )

    development_ids = set(
        development_df["recommendation_id"].astype(str).str.strip()
    )
    evaluation_ids = set(
        evaluation_df["recommendation_id"].astype(str).str.strip()
    )
    full_ids = set(
        full_df["recommendation_id"].astype(str).str.strip()
    )
    development_overlap_count = len(development_ids.intersection(full_ids))
    evaluation_overlap_count = len(evaluation_ids.intersection(full_ids))
    development_evaluation_overlap_count = len(
        development_ids.intersection(evaluation_ids)
    )
    if development_evaluation_overlap_count != 0:
        raise RuntimeError(
            "The frozen development and evaluation files overlap by "
            f"{development_evaluation_overlap_count:,} IDs."
        )
    if development_overlap_count != 0 or evaluation_overlap_count != 0:
        raise RuntimeError(
            "\nThe 98,173-row input still contains frozen labelled rows."
            f"\nDevelopment overlap: {development_overlap_count:,}"
            f"\nEvaluation overlap:  {evaluation_overlap_count:,}"
        )
    print("\nPASS: labelled-row exclusion is exact.")
    print("  Development IDs found in input: 0/1,000")
    print("  Evaluation IDs found in input:  0/300")
    print("  Development/evaluation overlap: 0")

    scores_by_factor = fit_development_and_score_full_in_batches(
        development_df,
        full_df,
        batch_size=args.batch_size,
    )
    final_predictions = apply_rules_and_hybrid_in_batches(
        full_df,
        scores_by_factor,
        batch_size=args.batch_size,
        audit_temp_path=audit_temp_path,
    )

    output_df = full_df.copy()
    for factor in frozen_v5.FACTORS:
        output_df[factor] = final_predictions[factor]

    validate_output(full_df, output_df)
    summary_df = build_prediction_summary(output_df)
    validate_expected_mention_counts(summary_df)

    print("\nOverall V5 mention rates:")
    print(
        summary_df.loc[
            summary_df["scope"] == "all_games",
            ["factor", "mention_count", "mention_percentage"],
        ].to_string(index=False)
    )

    main_output_sha256 = write_csv_atomically(
        output_df,
        main_output_path,
        overwrite=args.overwrite,
    )
    summary_sha256 = write_csv_atomically(
        summary_df,
        summary_path,
        overwrite=args.overwrite,
    )

    audit_sha256 = None
    if audit_temp_path is not None:
        if not audit_temp_path.exists():
            raise RuntimeError("The requested audit temporary file is missing.")
        if audit_path.exists() and not args.overwrite:
            raise FileExistsError(f"Audit output exists:\n{audit_path}")
        audit_temp_path.replace(audit_path)
        audit_sha256 = sha256_file(audit_path)

    elapsed_seconds = round(time.time() - started_at, 2)
    manifest = {
        "nlp_version": "V5 frozen",
        "method": "sentence rules OR thresholded TF-IDF LinearSVC",
        "training_rows": len(development_df),
        "full_prediction_rows": len(full_df),
        "development_rows_found_in_full_input": (
            development_overlap_count
        ),
        "evaluation_rows_used_for_training_or_prediction": 0,
        "evaluation_rows_used_for_exclusion_id_check": len(evaluation_df),
        "evaluation_rows_found_in_full_input": evaluation_overlap_count,
        "development_evaluation_id_overlap": (
            development_evaluation_overlap_count
        ),
        "factors": frozen_v5.FACTORS,
        "model_configs": frozen_v5.MODEL_CONFIGS,
        "random_seed": frozen_v5.RANDOM_SEED,
        "batch_size": args.batch_size,
        "frozen_configuration_fingerprint_sha256": (
            frozen_fingerprint
        ),
        "frozen_v5_source_sha256": frozen_source_sha256,
        "development_file": development_path.name,
        "development_file_sha256": development_sha256,
        "evaluation_id_file": evaluation_id_path.name,
        "evaluation_id_file_sha256": evaluation_sha256,
        "full_input_file": full_input_path.name,
        "full_input_file_sha256": input_sha256,
        "main_output_file": main_output_path.name,
        "main_output_file_sha256": main_output_sha256,
        "summary_file": summary_path.name,
        "summary_file_sha256": summary_sha256,
        "audit_file": audit_path.name if args.save_audit else None,
        "audit_file_sha256": audit_sha256,
        "runtime_versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "elapsed_seconds": elapsed_seconds,
        "important_note": (
            "The 300-row fixed evaluation labels were not used for "
            "training, thresholding, or prediction. Its IDs and the "
            "1,000 development IDs were used only to verify that all "
            "1,300 labelled rows are absent from this output."
        ),
    }

    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"Manifest output exists:\n{manifest_path}")
    temp_manifest_path = manifest_path.with_suffix(".json.tmp")
    try:
        with temp_manifest_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as manifest_file:
            json.dump(
                manifest,
                manifest_file,
                ensure_ascii=False,
                indent=2,
            )
            manifest_file.write("\n")
        temp_manifest_path.replace(manifest_path)
    except Exception:
        if temp_manifest_path.exists():
            temp_manifest_path.unlink()
        raise

    print("\n" + "=" * 78)
    print("FILTERED V5 APPLICATION COMPLETE")
    print("=" * 78)
    print(f"Main 31-column dataset:\n{main_output_path}")
    print(f"\nPrediction QA summary:\n{summary_path}")
    print(f"\nReproducibility manifest:\n{manifest_path}")
    if args.save_audit:
        print(f"\nOptional prediction audit:\n{audit_path}")
    print(f"\nElapsed time: {elapsed_seconds:.2f} seconds")
    print(
        "\nNext step: use the main 31-column CSV to construct the final "
        "logistic-regression variables. Do not tune V5 from these "
        "full-data mention rates or later regression results."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\n" + "=" * 78, file=sys.stderr)
        print("V5 FILTERED APPLICATION STOPPED", file=sys.stderr)
        print("=" * 78, file=sys.stderr)
        print(str(error), file=sys.stderr)
        raise
