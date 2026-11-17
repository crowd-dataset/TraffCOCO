"""
sign_text.py

Keep what a traffic sign says, in any language, through to the dataset.

The scene model reads sign text and often translates it, for example
"text reads '次の信号 左折' (Next signal, left turn)". Ontology reasoning
then maps the sign onto one of a fixed set of classes by embedding
similarity, and for a guide sign with words on it the nearest class can
be a pictogram sign with a different meaning (here ``left_only_sign``,
a mandatory turn). The words, and with them the meaning, were lost.

This module

* extracts ``text``, ``language`` and ``meaning`` for a sign, from the
  structured ``visual_attributes`` fields when the model filled them, or
  else from the quoted text in its description;
* decides when the chosen ontology class cannot explain the sign: the
  sign shows words, but neither its text nor its English meaning
  contains any of the class's typical texts or symbols. Such signs are
  relabelled as ``guide_information_sign``, keeping the replaced class
  for review. A stop sign reading "STOP", "止まれ" or (translated) "stop"
  keeps its class; number-only signs (speed limits) are never touched.
"""

from __future__ import annotations

import re
from typing import Any

GUIDE_SIGN_CLASS = "guide_information_sign"

_NO_TEXT = {"", "none", "unknown", "unreadable", "n/a", "na", "null", "..."}

# Quoted text after a cue word, then an optional "(translation)".
_QUOTED_TEXT = re.compile(
    r"(?:text|reads|reading|says|saying|written|characters|words)\b[^'\"「“‘]{0,40}"
    r"(?:'([^']+)'|\"([^\"]+)\"|「([^」]+)」|“([^”]+)”|‘([^’]+)’)"
    r"(?:\s*\(([^)]+)\))?",
    re.IGNORECASE,
)

_MEANING_CUE = re.compile(
    r"(?:meaning|means|translat\w*(?:\s+as)?|i\.e\.)\s*[:,]?\s*['\"“]?([^'\".”;]+)",
    re.IGNORECASE,
)


def has_words(text: Any) -> bool:
    """True when ``text`` contains letters in any script (not only
    digits, arrows or punctuation)."""

    return isinstance(text, str) and any(ch.isalpha() for ch in text)


def _clean(value: Any) -> str | None:

    if not isinstance(value, str):
        return None

    value = value.strip().strip("'\"“”「」").strip()

    if value.lower() in _NO_TEXT:
        return None

    return value


def extract_sign_text(scene_object: dict[str, Any]) -> dict[str, str] | None:
    """Return {"text", "language", "meaning", "source"} for a sign with
    readable text, or None."""

    visual = scene_object.get("visual_attributes")
    visual = visual if isinstance(visual, dict) else {}

    text = _clean(visual.get("text"))
    language = _clean(visual.get("language"))
    meaning = _clean(visual.get("meaning") or visual.get("translation"))
    source = "visual_attributes" if text else None

    description = scene_object.get("description")
    description = description if isinstance(description, str) else ""

    if text is None and description:
        match = _QUOTED_TEXT.search(description)

        if match:
            quoted = next((g for g in match.groups()[:5] if g), None)
            text = _clean(quoted)
            source = "description" if text else None

            if meaning is None and match.group(6):
                meaning = _clean(match.group(6))

    if text is None:
        return None

    if meaning is None and description:
        cue = _MEANING_CUE.search(description)
        if cue:
            meaning = _clean(cue.group(1))

    result = {"text": text, "source": source or "description"}

    if language:
        result["language"] = language

    if meaning:
        result["meaning"] = meaning

    return result


# Signs whose text legitimately varies; never relabelled.
_VARIABLE_TEXT_CLASSES = {"variable_message_sign"}


def _phrase_found(phrase: str, text: str) -> bool:
    """Case-insensitive match of an ontology phrase in sign text or
    meaning. Entries of one or two characters ("P") must be a whole
    token; longer ones must appear on word boundaries."""

    phrase = re.sub(r"[-_]+", " ", phrase).strip().lower()
    text = re.sub(r"[-_]+", " ", text).lower()

    if not phrase or not has_words(phrase):
        return False

    if len(phrase) <= 2:
        return phrase in re.findall(r"\w+", text)

    if not phrase.isascii():
        # Scripts without spaces (Japanese, Chinese, ...): substring.
        return phrase in text

    return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text) is not None


def class_matches_text(
    class_entry: dict[str, Any] | None,
    sign_text: dict[str, str],
) -> bool:
    """True when the sign's text or English meaning contains one of the
    class's typical texts or symbols (e.g. "STOP", "止まれ", "parking")."""

    info = (class_entry or {}).get("text_information") or {}
    phrases = list(info.get("common_text") or []) + list(
        info.get("common_symbols") or []
    )

    said = " ".join(
        str(sign_text.get(key) or "") for key in ("text", "meaning")
    )

    return any(
        _phrase_found(str(phrase), said)
        for phrase in phrases
        if isinstance(phrase, str)
    )


def needs_guide_sign_class(
    sign_text: dict[str, str] | None,
    predicted_class: str | None,
    class_entry: dict[str, Any] | None,
) -> bool:
    """Whether a sign should be relabelled as a guide/information sign:
    it shows words, and the predicted sign class does not match them."""

    if not sign_text or not has_words(sign_text.get("text")):
        return False

    if predicted_class in (GUIDE_SIGN_CLASS, *_VARIABLE_TEXT_CLASSES):
        return False

    super_category = (class_entry or {}).get("super_category")

    if super_category not in (None, "traffic_sign"):
        return False

    return not class_matches_text(class_entry, sign_text)
