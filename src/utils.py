"""Shared constants and helpers for the Phishing URL ID3 project."""

from __future__ import annotations

from pathlib import Path
from typing import Final

# Default CSV location (relative to project root)
DEFAULT_CSV_PATH: Final[Path] = Path("data") / "PhiUSIIL_Phishing_URL_Dataset.csv"

TARGET_COL: Final[str] = "label"
ID_COL_DROP: Final[str] = "FILENAME"

# High-cardinality raw text: dropped by default for stable ID3 (see README)
HIGH_CARD_TEXT_COLS: Final[tuple[str, ...]] = ("URL", "Domain", "Title")

# All feature columns from the dataset schema (excluding FILENAME and label)
FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "URL",
    "URLLength",
    "Domain",
    "DomainLength",
    "IsDomainIP",
    "TLD",
    "URLSimilarityIndex",
    "CharContinuationRate",
    "TLDLegitimateProb",
    "URLCharProb",
    "TLDLength",
    "NoOfSubDomain",
    "HasObfuscation",
    "NoOfObfuscatedChar",
    "ObfuscationRatio",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "IsHTTPS",
    "LineOfCode",
    "LargestLineLength",
    "HasTitle",
    "Title",
    "DomainTitleMatchScore",
    "URLTitleMatchScore",
    "HasFavicon",
    "Robots",
    "IsResponsive",
    "NoOfURLRedirect",
    "NoOfSelfRedirect",
    "HasDescription",
    "NoOfPopup",
    "NoOfiFrame",
    "HasExternalFormSubmit",
    "HasSocialNet",
    "HasSubmitButton",
    "HasHiddenFields",
    "HasPasswordField",
    "Bank",
    "Pay",
    "Crypto",
    "HasCopyrightInfo",
    "NoOfImage",
    "NoOfCSS",
    "NoOfJS",
    "NoOfSelfRef",
    "NoOfEmptyRef",
    "NoOfExternalRef",
)

# Categorical / text-like (after dropping high-cardinality URL, Domain, Title)
CATEGORICAL_FEATURES: Final[tuple[str, ...]] = ("TLD",)

# Subset for manual prediction form (curated important fields)
MANUAL_PREDICTION_FEATURES: Final[tuple[str, ...]] = (
    "URLLength",
    "TLDLegitimateProb",
    "IsHTTPS",
    "NoOfSubDomain",
    "HasObfuscation",
    "URLSimilarityIndex",
    "TLD",
)

# Internal binary label encoding (PhiUSIIL convention)
# 1 = legitimate, 0 = phishing
LABEL_LEGITIMATE: Final[int] = 1
LABEL_PHISHING: Final[int] = 0

LABEL_DISPLAY: Final[dict[int, str]] = {
    LABEL_LEGITIMATE: "legitimate",
    LABEL_PHISHING: "phishing",
}


def label_to_display(y: int) -> str:
    return LABEL_DISPLAY.get(int(y), str(y))


def project_root() -> Path:
    """Directory containing app.py when run from project root."""
    return Path(__file__).resolve().parent.parent
