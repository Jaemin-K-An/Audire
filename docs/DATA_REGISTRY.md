# AUDIRE Data Registry and Provenance

## 1. Primary Korean monosyllabic calibration data
**Korean Monosyllabic Speech Perception Test Dataset** — 726 utterances, 2 speakers (363 each), 16 kHz, approximately 157 MB. Metadata includes syllable structure, duration, pitch, F1/F2/F3 and audio. License: **CC BY-NC-ND 4.0**. The dataset card says intended use/scope should be communicated to Woojae Han before use.

Use in AUDIRE:
- browser/desktop calibration stimuli
- phoneme-position parsing tests
- acoustic-feature sensitivity analyses
- reproducible stimulus identifiers

Do **not**:
- commit audio to this public repository
- publish modified audio or a derived audio corpus
- imply that the public dataset includes the 72 participants' row-level perceptual responses

Source: https://huggingface.co/datasets/K-University-AIED/korean_monosyllabic_speech

## 2. Korean sentence-level audio for end-to-end evaluation
**Zeroth-Korean**, OpenSLR SLR40 / Hugging Face mirror. Approximately 51.6 h training + 1.2 h test; 22,263 train utterances, 457 test utterances; CC BY 4.0.

Use in AUDIRE:
- ASR regression testing
- Korean word timestamping
- sentence-level jamo decomposition
- selective-caption output regression
- noise/SNR simulation on a permissively licensed corpus

Prefer the test split for CI-sized evaluation and a pinned sample manifest for deterministic tests.

Sources:
- https://www.openslr.org/40/
- https://huggingface.co/datasets/kresnik/zeroth_korean

## 3. Auxiliary audiological dataset
**A FAIR and Open-Access Database of Audiological Perceptual Measures**, Zenodo record 17091997. The record exposes General Information, source/Excel and SQL archives.

Use in AUDIRE only for:
- audiological schema design
- sensitivity analyses across hearing/perceptual measures
- checking plausible correlations among general audiological measures

Do not use its labels as Korean phoneme-confusion ground truth. Language/task mismatch must be explicit in every report.

Source: https://zenodo.org/records/17091997

## 4. 2026 Korean hearing-loss error studies
Joo et al. (2026), DOI **10.21848/asr.250214**: 72 older adults assigned to normal, mild, moderate, severe hearing groups; 726 Korean monosyllabic stimuli at MCL. Total error rates rose with hearing loss severity; errors were classified at onset/nucleus/coda and by substitution/addition/omission/compound pattern.

Ma et al. (2026), DOI **10.21848/asr.250216**: the same general experimental family analyzed confusion matrices and perceptual similarity/distance by hearing group. These publications are **reference/prior evidence**, not a license to invent or redistribute participant-level raw responses.

Published examples useful as sanity checks include stronger confusion within phonetic classes and hearing-level-dependent increases in similarity. Any numerical transcription from article tables must be stored with a `source_page/table` field and verified by a second pass.

## 5. KS-MWL-A / WRS reference
The 2015 WRS test-retest paper describes the Korean Standard Monosyllabic Word Lists for Adults (KS-MWL-A), with 4 lists × 50 words, and SRT measured as the level for 50% correct response. It reports decreasing test-retest reliability as lists are shortened. Use this to constrain calibration-shortening experiments; do not assume that a custom 10-word calibration is clinically equivalent to a standardized WRS.

Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC4582455/

## Provenance rules
1. Every downloaded source gets a machine-readable manifest: source URL/DOI, retrieval date, version/revision, license, file checksum.
2. `data/raw/` and `data/processed/` are ignored by Git.
3. Synthetic data must carry `is_synthetic=true` and can never be presented as clinical observations.
4. Human participant data is private and de-identified; no audiograms, WRS, SRT, responses, names or identifiers in this public repo.
5. A published aggregate may be manually transcribed only when its source table/figure is named and a verification checksum/test is added.
