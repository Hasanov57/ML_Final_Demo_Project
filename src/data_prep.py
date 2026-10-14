"""
data_prep.py — loading, cleaning, feature building and deterministic splitting.

The project uses only target-independent cleaning rules before the split. Any statistic
that can leak information (price-tier threshold, scaling mean/std) is learned from the
training set only.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
DATA_PATH = "data/bina_az_sale.csv"

# Explicit target-derived leakage columns from the project statement.
LEAKAGE_COLUMNS: list[str] = ["unit_price", "total_price"]

# Fixed levels keep the design matrix reproducible without learning category vocabulary
# from validation/test data.
CATEGORY_LEVELS = [
    "Yeni tikili",
    "Köhnə tikili",
    "Həyət evi/Bağ evi",
    "Torpaq",
    "Obyekt",
    "Ofis",
    "Qaraj",
]
CITY_LEVELS = ["bakı", "xırdalan", "xaçmaz", "astara", "naxçıvan mr", "sumqayıt", "ağdam", "gədəbəy"]


def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    """Read the raw UTF-8 CSV and return it unmodified."""
    return pd.read_csv(path, encoding="utf-8")


def _extract_number(series: pd.Series) -> pd.Series:
    """Extract the first decimal number from text such as '120 m²' or '3.5 sot'."""
    text = series.astype("string")
    number = text.str.extract(r"([0-9]+(?:[\.,][0-9]+)?)", expand=False)
    return pd.to_numeric(number.str.replace(",", ".", regex=False), errors="coerce")


def _parse_floor(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Parse '8 / 17' into current_floor=8 and total_floors=17."""
    parts = series.astype("string").str.extract(r"^\s*([0-9]+)\s*/\s*([0-9]+)\s*$")
    current = pd.to_numeric(parts[0], errors="coerce")
    total = pd.to_numeric(parts[1], errors="coerce")
    return current, total


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the raw scrape into one row per listing with modelling-friendly columns.

    Main decisions:
      * keep the latest scrape for each listing URL (removes repeated snapshots),
      * keep AZN listings only,
      * remove target-derived leakage (unit_price, total_price), identifiers and text IDs,
      * parse area/floor/land fields,
      * use fixed plausibility bounds for obvious scrape/data-entry outliers,
      * encode missing/not-applicable structural values with 0 plus missing indicators.

    No target-dependent statistics are calculated here.
    """
    out = df.copy()

    # Keep only AZN listings (the supplied dump is overwhelmingly/all AZN, but this is explicit).
    if "currency_x" in out.columns:
        out = out[out["currency_x"].astype(str).str.upper().eq("AZN")].copy()

    # One listing can occur many times because the site was scraped repeatedly.
    # Keep the most recent available snapshot for that listing.
    listing_key = "estate_rel_url_y" if "estate_rel_url_y" in out.columns else "estate_rel_url"
    time_col = "datetime_scrape_y" if "datetime_scrape_y" in out.columns else "datetime_scrape_x"
    if listing_key in out.columns:
        if time_col in out.columns:
            out["_scrape_time"] = pd.to_datetime(out[time_col], errors="coerce")
            out = out.sort_values("_scrape_time")
        out = out.drop_duplicates(subset=[listing_key], keep="last")

    # Required numeric fields.
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["area_m2"] = _extract_number(out["Sahə"])
    out["rooms"] = pd.to_numeric(out["Otaq sayı"], errors="coerce")
    out["land_area_sot"] = _extract_number(out["Torpaq sahəsi"])
    out["current_floor"], out["total_floors"] = _parse_floor(out["Mərtəbə"])
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lng"] = pd.to_numeric(out["lng"], errors="coerce")

    # Fixed plausibility rules: they do not use quantiles/medians from validation/test data.
    # They remove obvious typos such as price=11 AZN, price=600M AZN, area=200,000 m²,
    # and impossible coordinates for Azerbaijan.
    mask = (
        out["price"].between(10_000, 10_000_000)
        & out["area_m2"].between(1, 10_000)
        & (out["rooms"].isna() | out["rooms"].between(1, 20))
        & (out["land_area_sot"].isna() | out["land_area_sot"].between(0.1, 1_000))
        & (out["current_floor"].isna() | out["current_floor"].between(0, 60))
        & (out["total_floors"].isna() | out["total_floors"].between(1, 60))
        & out["lat"].between(38.0, 42.5)
        & out["lng"].between(44.0, 51.5)
    )
    out = out.loc[mask].copy()

    # Missing indicators preserve the distinction between 0 and "not supplied/not applicable".
    for col in ["rooms", "land_area_sot", "current_floor", "total_floors"]:
        out[f"{col}_missing"] = out[col].isna().astype(float)
        out[col] = out[col].fillna(0.0)

    # Low-cardinality categorical/binary fields.
    out["category"] = out["Kateqoriya"].fillna("Unknown").astype(str)
    out["city_clean"] = out["city"].fillna("Unknown").astype(str).str.lower().str.strip()
    out["repair_bin"] = out["Təmir"].fillna("yoxdur").eq("var").astype(float)
    out["deed_bin"] = out["Çıxarış"].fillna("yoxdur").eq("var").astype(float)
    out["mortgage_bin"] = out["İpoteka"].fillna("yoxdur").eq("var").astype(float)

    # Remove explicitly forbidden target leakage if present. The modelling dataframe contains
    # only the cleaned columns below, so identifiers/raw text cannot accidentally enter X.
    out = out.drop(columns=[c for c in LEAKAGE_COLUMNS if c in out.columns], errors="ignore")

    keep = [
        "price",
        "area_m2",
        "rooms",
        "land_area_sot",
        "current_floor",
        "total_floors",
        "lat",
        "lng",
        "rooms_missing",
        "land_area_sot_missing",
        "current_floor_missing",
        "total_floors_missing",
        "repair_bin",
        "deed_bin",
        "mortgage_bin",
        "category",
        "city_clean",
    ]
    return out[keep].reset_index(drop=True)


def make_features(df: pd.DataFrame):
    """Build a fully numeric design matrix and return (X, price, feature_names)."""
    numeric_cols = [
        "area_m2",
        "rooms",
        "land_area_sot",
        "current_floor",
        "total_floors",
        "lat",
        "lng",
        "rooms_missing",
        "land_area_sot_missing",
        "current_floor_missing",
        "total_floors_missing",
        "repair_bin",
        "deed_bin",
        "mortgage_bin",
    ]

    parts = [df[numeric_cols].astype(float).reset_index(drop=True)]

    cat = pd.Categorical(df["category"], categories=CATEGORY_LEVELS)
    cat_dummies = pd.get_dummies(cat, prefix="category", dtype=float)
    parts.append(cat_dummies)

    city = pd.Categorical(df["city_clean"], categories=CITY_LEVELS)
    city_dummies = pd.get_dummies(city, prefix="city", dtype=float)
    parts.append(city_dummies)

    X_df = pd.concat(parts, axis=1)
    X = X_df.to_numpy(dtype=float)
    y = df["price"].to_numpy(dtype=float)
    return X, y, X_df.columns.tolist()


def make_tier_label(y_price, threshold=None):
    """
    Convert price into standard=0 / premium=1.

    If threshold is None it is the median of the passed array. In the pipeline this function
    is first called with y_train only, then the returned threshold is reused for val/test.
    """
    y_price = np.asarray(y_price, dtype=float)
    if threshold is None:
        threshold = float(np.median(y_price))
    y_tier = (y_price > threshold).astype(int)
    return y_tier, float(threshold)


def _allocate_stratified(indices, labels, n_take, rng):
    """Take approximately n_take indices while preserving class proportions."""
    indices = np.asarray(indices)
    labels = np.asarray(labels)
    chosen = []
    classes, counts = np.unique(labels, return_counts=True)
    raw = counts / counts.sum() * n_take
    take_counts = np.floor(raw).astype(int)
    # Distribute rounding remainder to classes with largest fractional part.
    for j in np.argsort(-(raw - take_counts))[: n_take - take_counts.sum()]:
        take_counts[j] += 1
    for cls, k in zip(classes, take_counts):
        cls_idx = indices[labels == cls].copy()
        rng.shuffle(cls_idx)
        chosen.extend(cls_idx[:k])
    chosen = np.asarray(chosen, dtype=int)
    rng.shuffle(chosen)
    return chosen


def split_indices(y_price, val_size=0.15, test_size=0.15, seed=SEED):
    """
    Create one leakage-safe split used by both tasks.

    Training rows are selected first. The price-tier threshold is then learned from training
    prices only. The remaining rows are divided into validation/test with stratification by
    that fixed tier label. The training split is naturally ~50/50 because its own median is
    used as threshold.
    """
    y_price = np.asarray(y_price, dtype=float)
    n = len(y_price)
    if not (0 < val_size < 1 and 0 < test_size < 1 and val_size + test_size < 1):
        raise ValueError("val_size and test_size must be positive and sum to < 1")

    rng = np.random.default_rng(seed)
    all_idx = np.arange(n)
    rng.shuffle(all_idx)
    n_train = int(round(n * (1.0 - val_size - test_size)))
    tr_idx = all_idx[:n_train]
    remain_idx = all_idx[n_train:]

    threshold = float(np.median(y_price[tr_idx]))
    remain_labels = (y_price[remain_idx] > threshold).astype(int)
    n_val = int(round(n * val_size))
    val_idx = _allocate_stratified(remain_idx, remain_labels, n_val, rng)
    val_set = set(val_idx.tolist())
    te_idx = np.asarray([i for i in remain_idx if i not in val_set], dtype=int)
    rng.shuffle(te_idx)
    return tr_idx, val_idx, te_idx, threshold


def train_val_test_split(X, y, val_size=0.15, test_size=0.15, seed=SEED):
    """Deterministic leakage-safe split. Returns X/y for train, validation and test."""
    X = np.asarray(X)
    y = np.asarray(y)
    tr, va, te, _ = split_indices(y, val_size=val_size, test_size=test_size, seed=seed)
    return X[tr], y[tr], X[va], y[va], X[te], y[te]


def standardize(X_tr, *others):
    """
    Standardize with TRAIN mean/std only and apply the same transform to all other arrays.
    Constant columns receive std=1 to avoid division by zero.
    """
    X_tr = np.asarray(X_tr, dtype=float)
    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    scaled = [(X_tr - mean) / std]
    scaled.extend((np.asarray(arr, dtype=float) - mean) / std for arr in others)
    return tuple(scaled)
