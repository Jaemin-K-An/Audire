"""E0 — source registry, licence policy, manifests/checksums and stimulus catalogues."""

from __future__ import annotations

import json

import pytest

from audire.data.manifest import Manifest, sha256_file
from audire.data.sources import (
    AcknowledgementRequired,
    Source,
    SourceRegistry,
    SourceUseViolation,
    load_registry,
    registry,
)
from audire.data.stimuli import (
    BUILTIN_CODAS,
    DESIGN_SPACE_SIZE,
    build_balanced_catalog,
    catalog_from_syllables,
    phoneme_balanced_subset,
    random_subset,
)
from audire.hangul import NUCLEUS_JAMO, ONSET_JAMO

# =========================================================== source registry


def test_registry_parses_and_contains_the_declared_sources() -> None:
    reg = registry()
    assert reg.schema_version == 1
    assert set(reg.sources) == {
        "korean_monosyllabic_speech",
        "zeroth_korean_test",
        "hapaa_audiology",
    }
    assert set(reg.literature) == {
        "joo2026_error_rates",
        "ma2026_similarity",
        "who2021_hearing_grades",
        "kim2015_ksmwla_reliability",
    }


def test_every_source_declares_licence_verification_date_and_use_policy() -> None:
    for sid, src in registry().sources.items():
        assert src.license, sid
        assert src.verified_at, sid
        assert src.permitted_uses, f"{sid} declares no permitted uses"
        assert src.prohibited_uses, f"{sid} declares no prohibited uses"


def test_every_literature_reference_has_doi_and_access_date() -> None:
    for lid, ref in registry().literature.items():
        assert ref.doi, lid
        assert ref.accessed, lid
        assert ref.url, lid


def test_primary_corpus_is_nd_so_redistribution_is_not_allowed() -> None:
    src = registry().get("korean_monosyllabic_speech")
    assert src.license == "CC-BY-NC-ND-4.0"
    assert src.redistribution_allowed is False
    assert src.requires_human_acknowledgement is True


def test_zeroth_is_cc_by_so_derivatives_are_allowed() -> None:
    src = registry().get("zeroth_korean_test")
    assert src.redistribution_allowed is True
    assert src.requires_human_acknowledgement is False


def test_zeroth_fetch_scope_is_the_test_split_only() -> None:
    src = registry().get("zeroth_korean_test")
    patterns = list(src.expected["allow_patterns"])
    assert src.expected["parquet_file"] in patterns
    assert "README.md" in patterns
    assert not any("train" in pattern for pattern in patterns)


def test_acknowledgement_gate_blocks_until_env_var_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    src = registry().get("korean_monosyllabic_speech")
    monkeypatch.delenv("AUDIRE_PRIMARY_DATA_USE_NOTIFIED", raising=False)
    assert src.acknowledgement_satisfied() is False
    with pytest.raises(AcknowledgementRequired, match="inform the creator"):
        src.require_acknowledgement()

    monkeypatch.setenv("AUDIRE_PRIMARY_DATA_USE_NOTIFIED", "1")
    assert src.acknowledgement_satisfied() is True
    src.require_acknowledgement()  # must not raise


def test_acknowledgement_message_never_offers_to_send_the_notification() -> None:
    src = registry().get("korean_monosyllabic_speech")
    try:
        src.require_acknowledgement()
    except AcknowledgementRequired as exc:
        assert "will not send that message" in str(exc)
    else:  # pragma: no cover - only reachable if the env var leaked in
        pytest.skip("acknowledgement already set in this environment")


def test_prohibited_use_tripwire_rejects_redistribution_of_the_nd_corpus() -> None:
    src = registry().get("korean_monosyllabic_speech")
    with pytest.raises(SourceUseViolation, match="prohibits"):
        src.assert_permits("redistributing modified audio as a derived corpus")
    with pytest.raises(SourceUseViolation, match="prohibits"):
        src.assert_permits("commercial use in a paid product")
    src.assert_permits("local calibration stimulus playback")


def test_auxiliary_dataset_may_not_be_used_as_korean_phoneme_ground_truth() -> None:
    src = registry().get("hapaa_audiology")
    with pytest.raises(SourceUseViolation, match="prohibits"):
        src.assert_permits("treating its speech labels as Korean phoneme-confusion ground truth")


def test_registry_rejects_unknown_ids() -> None:
    reg = registry()
    with pytest.raises(KeyError, match="unregistered source"):
        reg.get("not_a_source")
    with pytest.raises(KeyError, match="unregistered literature"):
        reg.cite("not_a_paper")


def test_registry_is_loadable_from_an_explicit_path() -> None:
    from audire.config.paths import sources_file

    assert load_registry(sources_file()).schema_version == 1


# =========================================================== manifests


def test_manifest_records_checksums_and_verifies(tmp_path) -> None:
    root = tmp_path / "corpus"
    (root / "sub").mkdir(parents=True)
    (root / "a.txt").write_text("alpha", encoding="utf-8")
    (root / "sub" / "b.txt").write_text("beta", encoding="utf-8")

    m = Manifest.build(source_id="demo", license="CC-BY-4.0", local_path=root)
    assert m.n_files == 2
    assert m.total_bytes == len("alpha") + len("beta")
    assert m.verify() == []
    assert {f.path for f in m.files} == {"a.txt", "sub/b.txt"}
    assert all(len(f.sha256) == 64 for f in m.files)


def test_manifest_detects_modification_missing_and_extra_files(tmp_path) -> None:
    root = tmp_path / "c"
    root.mkdir()
    (root / "a.txt").write_text("alpha", encoding="utf-8")
    (root / "b.txt").write_text("beta", encoding="utf-8")
    m = Manifest.build(source_id="demo", license="CC-BY-4.0", local_path=root)

    (root / "a.txt").write_text("ALPHA", encoding="utf-8")  # same length, different bytes
    (root / "b.txt").unlink()
    (root / "c.txt").write_text("gamma", encoding="utf-8")

    problems = m.verify()
    assert any("checksum mismatch: a.txt" in p for p in problems)
    assert any("missing file: b.txt" in p for p in problems)
    assert any("unrecorded file present: c.txt" in p for p in problems)


def test_shallow_verify_catches_size_change_but_skips_checksums(tmp_path) -> None:
    root = tmp_path / "c"
    root.mkdir()
    (root / "a.txt").write_text("alpha", encoding="utf-8")
    m = Manifest.build(source_id="demo", license="CC-BY-4.0", local_path=root)
    (root / "a.txt").write_text("ALPHA", encoding="utf-8")
    assert m.verify(deep=False) == []
    assert m.verify(deep=True) != []
    (root / "a.txt").write_text("alphabet", encoding="utf-8")
    assert any("size mismatch" in p for p in m.verify(deep=False))


def test_content_digest_is_stable_and_location_independent(tmp_path) -> None:
    def make(where: str) -> Manifest:
        root = tmp_path / where
        root.mkdir()
        (root / "x.bin").write_bytes(b"\x00\x01\x02")
        return Manifest.build(source_id="demo", license="CC-BY-4.0", local_path=root)

    a, b = make("one"), make("two")
    assert a.local_path != b.local_path
    assert a.content_digest == b.content_digest


def test_manifest_json_roundtrip_and_missing_manifest_message(tmp_path) -> None:
    root = tmp_path / "c"
    root.mkdir()
    (root / "a.txt").write_text("alpha", encoding="utf-8")
    m = Manifest.build(source_id="demo", license="CC-BY-4.0", local_path=root)
    out = tmp_path / "demo.json"
    m.save(out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["content_digest"] == m.content_digest
    assert Manifest.from_dict(payload).content_digest == m.content_digest

    with pytest.raises(FileNotFoundError, match="Run `make data`"):
        Manifest.load("nope", tmp_path / "nope.json")


def test_missing_local_path_reports_one_clear_problem(tmp_path) -> None:
    root = tmp_path / "c"
    root.mkdir()
    (root / "a.txt").write_text("alpha", encoding="utf-8")
    m = Manifest.build(source_id="demo", license="CC-BY-4.0", local_path=root)
    (root / "a.txt").unlink()
    root.rmdir()
    assert m.verify() == [f"local path is missing: {root}"]


def test_sha256_file_matches_hashlib(tmp_path) -> None:
    import hashlib

    p = tmp_path / "f"
    p.write_bytes(b"audire" * 1000)
    assert sha256_file(p) == hashlib.sha256(b"audire" * 1000).hexdigest()


def test_fetch_does_not_reuse_a_manifest_after_registry_expectations_change(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from audire.data import fetch as fetch_module

    monkeypatch.setenv("AUDIRE_RAW_DIR", str(tmp_path / "raw"))
    monkeypatch.setenv("AUDIRE_MANIFESTS_DIR", str(tmp_path / "manifests"))
    source = Source(
        id="scope-test",
        role="test",
        title="scope test",
        kind="unit_test",
        license="CC-BY-4.0",
        verified_at="2026-08-16",
        revision="rev-1",
        expected={"allow_patterns": ["test.parquet"]},
    )
    reg = SourceRegistry(schema_version=1, sources={source.id: source}, literature={})
    dest = fetch_module.local_path_for(source)
    dest.mkdir(parents=True)
    (dest / "test.parquet").write_bytes(b"fixed")
    Manifest.build(
        source_id=source.id,
        license=source.license,
        local_path=dest,
        revision=source.revision,
        expected={},
    ).save()
    called = False

    def fetch_again(_source: Source, _dest) -> dict[str, str]:
        nonlocal called
        called = True
        return {"backend": "unit_test"}

    monkeypatch.setitem(fetch_module._BACKENDS, "unit_test", fetch_again)
    result = fetch_module.fetch_source(source.id, reg=reg)

    assert called is True
    assert result.expected == source.expected


# =========================================================== stimulus catalogue


def test_design_space_matches_the_documented_3192_combinations() -> None:
    assert DESIGN_SPACE_SIZE == 19 * 21 * 8 == 3192


def test_builtin_catalog_is_deterministic() -> None:
    a = build_balanced_catalog(100)
    b = build_balanced_catalog(100)
    assert [s.syllable for s in a] == [s.syllable for s in b]
    assert [s.stimulus_id for s in a] == [s.stimulus_id for s in b]


def test_builtin_full_catalog_visits_every_combination_exactly_once() -> None:
    full = build_balanced_catalog()
    triples = {(s.decomposition.onset, s.decomposition.nucleus, s.decomposition.coda) for s in full}
    assert len(full) == DESIGN_SPACE_SIZE
    assert len(triples) == DESIGN_SPACE_SIZE


def test_builtin_head_is_the_balanced_subset() -> None:
    """head(k) of the built-in catalogue must be exactly the first k of the full design."""
    full = build_balanced_catalog()
    for k in (10, 25, 50, 100):
        assert [s.syllable for s in build_balanced_catalog(k)] == [s.syllable for s in full.head(k)]


@pytest.mark.parametrize("n", [19, 21, 38, 42, 168, 336])
def test_builtin_marginal_balance_is_exact_at_multiples(n: int) -> None:
    cov = build_balanced_catalog(n).coverage()
    for position, inventory_size in (("onset", 19), ("nucleus", 21), ("coda", 8)):
        if n % inventory_size == 0:
            counts = cov[position]
            assert len(counts) == inventory_size
            assert set(counts.values()) == {n // inventory_size}


def test_short_calibrations_cannot_cover_the_inventory_and_say_so() -> None:
    """A 10-item list cannot exercise 19 onsets. The catalogue must not pretend otherwise."""
    cov = build_balanced_catalog(10).coverage()
    assert len(cov["onset"]) == 10
    assert len(cov["nucleus"]) == 10
    assert len(cov["coda"]) == 8
    assert build_balanced_catalog(10).balance_score()["onset"] < 1.0
    assert build_balanced_catalog(19).balance_score()["onset"] == pytest.approx(1.0)


def test_builtin_uses_only_neutralised_codas() -> None:
    codas = {s.decomposition.coda for s in build_balanced_catalog()}
    assert codas == set(BUILTIN_CODAS)
    assert len(codas) == 8


def test_builtin_covers_the_whole_onset_and_nucleus_inventory() -> None:
    full = build_balanced_catalog()
    assert {s.decomposition.onset for s in full} == set(ONSET_JAMO)
    assert {s.decomposition.nucleus for s in full} == set(NUCLEUS_JAMO)


def test_builtin_catalog_rejects_out_of_range_sizes() -> None:
    with pytest.raises(ValueError, match="must be in 1"):
        build_balanced_catalog(0)
    with pytest.raises(ValueError, match="must be in 1"):
        build_balanced_catalog(DESIGN_SPACE_SIZE + 1)


def test_builtin_carries_no_third_party_licence_and_flags_uncalibrated_audio() -> None:
    prov = build_balanced_catalog(10).provenance
    assert prov["license"] == "none (generated from the Hangul inventory)"
    assert "not clinical" in prov["audio"]


def test_phoneme_balanced_subset_beats_random_on_coverage() -> None:
    """The E5 balanced arm must actually be more balanced than the random arm."""
    pool = build_balanced_catalog(500)
    # Shuffle the pool so the native order gives the greedy selector no free advantage.
    shuffled = random_subset(pool, 500, seed=1)
    balanced = phoneme_balanced_subset(shuffled, 19)
    rand = random_subset(shuffled, 19, seed=3)
    assert len(balanced.coverage()["onset"]) >= len(rand.coverage()["onset"])
    assert balanced.balance_score()["onset"] >= rand.balance_score()["onset"]


def test_phoneme_balanced_subset_is_deterministic() -> None:
    pool = random_subset(build_balanced_catalog(300), 300, seed=5)
    a = [s.stimulus_id for s in phoneme_balanced_subset(pool, 40)]
    b = [s.stimulus_id for s in phoneme_balanced_subset(pool, 40)]
    assert a == b


def test_random_subset_is_seed_reproducible_and_seed_sensitive() -> None:
    pool = build_balanced_catalog(200)
    assert [s.stimulus_id for s in random_subset(pool, 25, seed=11)] == [
        s.stimulus_id for s in random_subset(pool, 25, seed=11)
    ]
    assert [s.stimulus_id for s in random_subset(pool, 25, seed=11)] != [
        s.stimulus_id for s in random_subset(pool, 25, seed=12)
    ]


def test_subset_helpers_validate_sizes() -> None:
    pool = build_balanced_catalog(20)
    with pytest.raises(ValueError, match="must be positive"):
        phoneme_balanced_subset(pool, 0)
    with pytest.raises(ValueError, match="must be positive"):
        random_subset(pool, 0, seed=1)
    with pytest.raises(ValueError, match="non-negative"):
        pool.head(-1)


def test_catalog_from_explicit_syllable_list() -> None:
    cat = catalog_from_syllables(["강", "산", "물"])
    assert [s.syllable for s in cat] == ["강", "산", "물"]
    assert cat[0].structure == "CVC"
    assert len(cat) == 3
