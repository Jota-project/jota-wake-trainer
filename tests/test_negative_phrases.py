# tests/test_negative_phrases.py
from trainer.negative_phrases import build_negative_phrases, _near_miss_variants


def test_build_negative_phrases_returns_requested_count():
    phrases = build_negative_phrases("ok jota", n=20)
    assert len(phrases) == 20


def test_build_negative_phrases_never_includes_the_wake_word_itself():
    phrases = build_negative_phrases("ok jota", n=40)
    assert "ok jota" not in [p.lower() for p in phrases]


def test_build_negative_phrases_has_no_duplicates():
    phrases = build_negative_phrases("ok jota", n=40)
    assert len(phrases) == len(set(phrases))


def test_near_miss_variants_substitutes_first_word():
    variants = _near_miss_variants("ok jota")
    assert any(v.endswith("jota") and not v.startswith("ok") for v in variants)


def test_near_miss_variants_handles_single_word():
    # No debe petar con una wake word de una sola palabra
    variants = _near_miss_variants("jarvis")
    assert isinstance(variants, list)
