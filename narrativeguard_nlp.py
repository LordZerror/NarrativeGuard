"""
NarrativeGuard NLP Analysis Pipeline
=====================================
Performs gender, severity, and hedging NLP analysis on MIMIC-IV discharge summaries.

Usage
-----
Single file:
    python narrativeguard_nlp.py --input data/file.csv --output-dir outputs/

Batch (all CSVs in a directory, then aggregate):
    python narrativeguard_nlp.py --input-dir data/csvs/ --output-dir outputs/

Batch with optional Gemma-based hedging:
    python narrativeguard_nlp.py --input-dir data/ --output-dir outputs/ --use-gemma --gemma-model /Users/kaushalpatil/Library/Caches/llama.cpp/unsloth_gemma-3-1b-it-GGUF_gemma-3-1b-it-Q4_K_M.gguf

Aggregate only (if files already processed):
    python narrativeguard_nlp.py --aggregate-only --output-dir outputs/

Options
-------
--input          Path to a single CSV file
--input-dir      Directory containing multiple CSV files
--output-dir     Where to write per-file results and the final aggregate (default: ./outputs)
--use-gemma      Enable Gemma-based hedging analysis (slow; requires llama-cpp-python)
--gemma-model    Path to the GGUF model file
--workers        Number of parallel workers for batch processing (default: 1)
--resume         Skip files whose output parquet already exists
--aggregate-only Skip processing, just merge existing output parquets
--no-plots       Skip generating visualisation PNGs
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import spacy
from spacy.matcher import Matcher
from tqdm.auto import tqdm

tqdm.pandas()
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG / CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

TEXT_COL   = "TEXT"
LABEL_COL  = "LABEL"
ID_COLS    = ["SUBJECT_ID", "HADM_ID", "NOTE_SEQ"]
CHUNK_SIZE = 5_000

# ─── Section Parsing ──────────────────────────────────────────────────────────

_HEADER_NORM = {
    r"History\s+of\s+Present\s+Illness": "History of Present Illness",
    r"Brief\s+Hospital\s+Course":         "Brief Hospital Course",
    r"Past\s+Medical\s+History":          "Past Medical History",
    r"Discharge\s+Diagnosis":             "Discharge Diagnosis",
    r"Discharge\s+Condition":             "Discharge Condition",
    r"Discharge\s+Instructions":          "Discharge Instructions",
    r"Pertinent\s+Results":               "Pertinent Results",
    r"Physical\s+Exam":                   "Physical Exam",
    r"Medications\s+on\s+Admission":      "Medications on Admission",
    r"Discharge\s+Medications":           "Discharge Medications",
    r"Social\s+History":                  "Social History",
    r"Family\s+History":                  "Family History",
}

_SEC_STOP = (
    r"(?=(?:Chief Complaint|Major Surgical|History of Present Illness|"
    r"Past Medical History|Social History|Family History|Physical Exam|"
    r"Pertinent Results|Brief Hospital Course|Medications on Admission|"
    r"Discharge Medications|Discharge Diagnosis|Discharge Condition|"
    r"Discharge Instructions|Followup Instructions|$))"
)

_SECTION_RE: Dict[str, re.Pattern] = {
    "chief_complaint":    re.compile(r"Chief Complaint\s*:(.*?)"           + _SEC_STOP, re.DOTALL | re.IGNORECASE),
    "history_present":    re.compile(r"History of Present Illness\s*:(.*?)"+ _SEC_STOP, re.DOTALL | re.IGNORECASE),
    "past_medical":       re.compile(r"Past Medical History\s*:(.*?)"      + _SEC_STOP, re.DOTALL | re.IGNORECASE),
    "social_history":     re.compile(r"Social History\s*:(.*?)"            + _SEC_STOP, re.DOTALL | re.IGNORECASE),
    "family_history":     re.compile(r"Family History\s*:(.*?)"            + _SEC_STOP, re.DOTALL | re.IGNORECASE),
    "hospital_course":    re.compile(r"Brief Hospital Course\s*:(.*?)"     + _SEC_STOP, re.DOTALL | re.IGNORECASE),
    "discharge_dx":       re.compile(r"Discharge Diagnosis\s*:(.*?)"       + _SEC_STOP, re.DOTALL | re.IGNORECASE),
    "discharge_instruct": re.compile(r"Discharge Instructions\s*:(.*?)"    + _SEC_STOP, re.DOTALL | re.IGNORECASE),
}

_SEX_RE = re.compile(r"Sex:\s*([MF])", re.IGNORECASE)


# ─── Gender ───────────────────────────────────────────────────────────────────

_FEMALE_PRONOUNS     = {"she", "her", "hers", "herself"}
_MALE_PRONOUNS       = {"he", "him", "his", "himself"}
_NON_BINARY_PRONOUNS = {"they", "them", "their", "theirs", "themself"}

_PATIENT_ANCHORS = {
    "patient", "pt", "admitted", "presented", "complains", "reports",
    "denies", "history", "hpi", "exam", "discharge", "readmit",
}
_EXCLUSION_TOKENS = {
    "doctor", "dr", "physician", "nurse", "brother", "sister",
    "mother", "father", "spouse", "husband", "wife", "son", "daughter",
}

_GENDER_THRESHOLD = 0.60


# ─── Severity ─────────────────────────────────────────────────────────────────

_SEVERITY_LEVELS: Dict[str, int] = {
    "critical": 5, "life-threatening": 5, "emergent": 5, "catastrophic": 5, "fatal": 5, "lethal": 5,
    "severe": 4, "serious": 4, "significant": 4, "profound": 4, "extensive": 4,
    "major": 4, "marked": 4, "extreme": 4,
    "moderate": 3, "considerable": 3, "substantial": 3, "progressive": 3,
    "worsening": 3, "worsened": 3, "deteriorating": 3,
    "mild": 2, "minimal": 2, "minor": 2, "slight": 2, "trace": 2, "low-grade": 2,
    "absent": 1, "no": 1, "without": 1, "negative": 1, "resolved": 1,
    "improved": 1, "stable": 1, "unremarkable": 1, "within normal limits": 1, "wnl": 1,
}

_CLINICAL_CONCEPTS = {
    "pain", "discomfort", "distension", "dyspnea", "shortness of breath", "sob",
    "edema", "swelling", "nausea", "vomiting", "diarrhea", "fever", "chills",
    "fatigue", "weakness", "confusion", "dizziness", "headache", "bleeding",
    "hematuria", "hemoptysis", "melena", "jaundice", "ascites", "cough",
    "orthopnea", "syncope", "abdominal pain", "abd pain", "chest pain", "tenderness",
    "cirrhosis", "hcv", "hiv", "copd", "heart failure", "sbp", "sepsis",
    "pneumonia", "uti", "aki", "hepatitis", "encephalopathy", "hyponatremia",
    "hyperkalemia", "anemia", "thrombocytopenia",
    "distended", "tympanitic", "tender", "nodular", "elevated",
    "hypertension", "hypotension", "tachycardia", "bradycardia",
}

_NEGATION_TERMS = {
    "no", "not", "without", "denies", "denied", "negative", "absent",
    "none", "neither", "nor", "never",
}


# ─── Hedging ──────────────────────────────────────────────────────────────────

_HEDGING_LEXICON: Dict[str, List[str]] = {
    "epistemic_uncertainty": [
        "appears", "appears to", "seems", "suggest", "suggests",
        "possible", "possibly", "probable", "probably", "likely", "unlikely",
        "may", "might", "could", "uncertain", "unclear",
        "question of", "cannot exclude", "rule out", "r/o",
        "differential", "presumed", "presumptive", "suspected",
        "consistent with", "compatible with", "thought to", "believed to",
        "it is unclear", "cannot be confirmed", "appears to be",
        "possibly related", "likely due", "working diagnosis",
    ],
    "patient_attribution": [
        "per patient", "patient reports", "patient states", "patient claims",
        "patient denies", "per the patient", "patient-reported",
        "patient alleges", "reportedly", "as reported by",
        "patient notes", "pt reports", "pt states", "pt denies",
        "she reports", "he reports", "she denies", "he denies",
        "she states", "he states", "she notes", "he notes",
        "she feels", "he feels", "pt c/o", "patient c/o",
        "reports that", "states that", "notes that", "denies that",
        "per report", "per ed report",
    ],
    "clinician_skepticism": [
        "however", "but states", "although reports", "claims to",
        "denies any", "no objective evidence", "without objective",
        "no clear", "questionable", "questionably", "dubious",
        "unverified", "not confirmed", "not corroborated",
        "difficult to assess", "unclear history", "history limited",
        "poor historian", "unreliable historian", "limited reliability",
        "secondary gain", "drug-seeking", "malingering",
        "noncompliant", "non-compliant", "med non-compliance",
        "self-discontinuing", "does not follow", "refused",
        "despite counseling", "despite recommendations",
    ],
}

# Counterfactual pronoun swap maps
_F2M = {
    "she": "he",   "She": "He",   "SHE": "HE",
    "her": "him",  "Her": "Him",  "HER": "HIM",
    "hers": "his", "Hers": "His", "HERS": "HIS",
    "herself": "himself", "Herself": "Himself", "HERSELF": "HIMSELF",
}
_M2F = {v: k for k, v in _F2M.items()}


# MODULE-LEVEL COMPILED OBJECTS  (compiled once per process)

def _build_statics():
    """Compile all regex patterns and build spaCy objects. Called once per worker."""
    global _nlp, _gender_matcher
    global _SEVERITY_RE, _CONCEPT_RE, _NEGATION_RE
    global _HEDGE_RE, _WORD_RE, _PRONOUN_RE

    _nlp = spacy.blank("en")
    _nlp.max_length = 2_000_000

    _gender_matcher = Matcher(_nlp.vocab)
    _gender_matcher.add("FEMALE",     [[{"LOWER": {"IN": list(_FEMALE_PRONOUNS)}}]])
    _gender_matcher.add("MALE",       [[{"LOWER": {"IN": list(_MALE_PRONOUNS)}}]])
    _gender_matcher.add("NON_BINARY", [[{"LOWER": {"IN": list(_NON_BINARY_PRONOUNS)}}]])

    _SEVERITY_RE = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in sorted(_SEVERITY_LEVELS, key=len, reverse=True)) + r")\b",
        re.IGNORECASE,
    )
    _CONCEPT_RE = re.compile(
        r"\b(" + "|".join(re.escape(c) for c in sorted(_CLINICAL_CONCEPTS, key=len, reverse=True)) + r")\b",
        re.IGNORECASE,
    )
    _NEGATION_RE = re.compile(
        r"\b(" + "|".join(re.escape(n) for n in _NEGATION_TERMS) + r")\b",
        re.IGNORECASE,
    )
    _HEDGE_RE = {
        cat: re.compile(
            r"\b(" + "|".join(re.escape(p) for p in sorted(phrases, key=len, reverse=True)) + r")\b",
            re.IGNORECASE,
        )
        for cat, phrases in _HEDGING_LEXICON.items()
    }
    _WORD_RE = re.compile(r"\b\w+\b")

    _all_pronouns = list(_F2M.keys()) + list(_M2F.keys())
    _PRONOUN_RE = re.compile(
        r"\b(" + "|".join(re.escape(p) for p in sorted(_all_pronouns, key=len, reverse=True)) + r")\b"
    )


# Initialise statics at import time (applies when workers import this module too)
_build_statics()

# SECTION 2 — TEXT CLEANING & SECTION PARSING

def clean_note(raw: str) -> str:
    text = raw.replace("\r\n", "\n").replace("\t", " ")
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\b_{3,}\b", "[REDACTED]", text)
    for pat, repl in _HEADER_NORM.items():
        text = re.sub(pat, repl, text)
    return text.strip()


def _extract_section(text: str, pattern: re.Pattern) -> str:
    m = pattern.search(text)
    if not m:
        return ""
    content = m.group(1)
    content = re.sub(r" {2,}", " ", content)
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip()


def parse_row(text: str) -> dict:
    cleaned = clean_note(text)
    out = {"cleaned_text": cleaned}
    m = _SEX_RE.search(cleaned)
    out["header_sex"] = m.group(1).upper() if m else ""
    for sec, pat in _SECTION_RE.items():
        out[f"sec_{sec}"] = _extract_section(cleaned, pat)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — GENDER ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def _context_tokens(doc, idx: int, window: int = 5) -> List[str]:
    lo = max(0, idx - window)
    hi = min(len(doc), idx + window + 1)
    return [doc[i].lower_ for i in range(lo, hi) if i != idx]


def _is_patient_ref(ctx: List[str]) -> bool:
    return bool(set(ctx) & _PATIENT_ANCHORS) and not bool(set(ctx) & _EXCLUSION_TOKENS)


def analyse_gender(text: str) -> dict:
    if not text.strip():
        return dict(gender_label="unknown", gender_confidence=0.0,
                    patient_female_ct=0, patient_male_ct=0, patient_non_binary_ct=0,
                    raw_female_ct=0, raw_male_ct=0, raw_non_binary_ct=0)

    doc = _nlp(text.lower())
    pf = pm = pn = rf = rm = rn = 0

    for match_id, start, _ in _gender_matcher(doc):
        label = _nlp.vocab.strings[match_id]
        ctx   = _context_tokens(doc, start)
        is_pt = _is_patient_ref(ctx)
        if label == "FEMALE":
            rf += 1
            if is_pt: pf += 1
        elif label == "MALE":
            rm += 1
            if is_pt: pm += 1
        else:
            rn += 1
            if is_pt: pn += 1

    total = pf + pm + pn
    if total == 0:
        label, conf = "unknown", 0.0
    else:
        best_ct = max(pf, pm, pn)
        conf = best_ct / total
        if conf < _GENDER_THRESHOLD:
            label = "ambiguous"
        elif pf == best_ct:
            label = "female"
        elif pm == best_ct:
            label = "male"
        else:
            label = "non-binary"

    return dict(
        gender_label=label, gender_confidence=round(conf, 4),
        patient_female_ct=pf, patient_male_ct=pm, patient_non_binary_ct=pn,
        raw_female_ct=rf, raw_male_ct=rm, raw_non_binary_ct=rn,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — SEVERITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyse_severity(text: str) -> dict:
    empty = dict(severity_score=np.nan, severity_max=np.nan,
                 severity_count=0, severity_high_ct=0,
                 severity_low_ct=0, negated_severity_ct=0)
    if not text.strip():
        return empty

    text_lower = text.lower()
    scores: List[int] = []
    high_ct = low_ct = negated_ct = 0

    for m in _SEVERITY_RE.finditer(text_lower):
        modifier = m.group(1)
        base_lvl = _SEVERITY_LEVELS.get(modifier, 0)
        if base_lvl == 0:
            continue

        start, end = m.start(), m.end()
        window_before = text_lower[max(0, start - 30): start]
        is_negated    = bool(_NEGATION_RE.search(window_before))
        effective_lvl = 1 if is_negated else base_lvl
        scores.append(effective_lvl)

        if is_negated:
            negated_ct += 1
        elif effective_lvl >= 4:
            high_ct += 1
        elif effective_lvl <= 2:
            low_ct += 1

    if not scores:
        return empty

    return dict(
        severity_score=round(float(np.mean(scores)), 4),
        severity_max=int(max(scores)),
        severity_count=len(scores),
        severity_high_ct=high_ct,
        severity_low_ct=low_ct,
        negated_severity_ct=negated_ct,
    )


def _severity_text(row) -> str:
    return " ".join(filter(None, [
        row.get("sec_chief_complaint", ""),
        row.get("sec_history_present", ""),
        row.get("sec_hospital_course", ""),
    ]))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — HEDGING ANALYSIS (rule-based)
# ─────────────────────────────────────────────────────────────────────────────

def analyse_hedging(text: str) -> dict:
    if not text.strip():
        return dict(hedge_epistemic_ct=0, hedge_attribution_ct=0,
                    hedge_skepticism_ct=0, hedge_total_ct=0,
                    hedge_skepticism_flag=0, hedge_density=0.0)

    text_lower  = text.lower()
    word_count  = len(_WORD_RE.findall(text_lower))
    epist_ct    = len(_HEDGE_RE["epistemic_uncertainty"].findall(text_lower))
    attrib_ct   = len(_HEDGE_RE["patient_attribution"].findall(text_lower))
    skeptic_ct  = len(_HEDGE_RE["clinician_skepticism"].findall(text_lower))
    total_ct    = epist_ct + attrib_ct + skeptic_ct
    density     = round(total_ct / (word_count / 100), 4) if word_count > 0 else 0.0

    return dict(
        hedge_epistemic_ct=epist_ct,
        hedge_attribution_ct=attrib_ct,
        hedge_skepticism_ct=skeptic_ct,
        hedge_total_ct=total_ct,
        hedge_skepticism_flag=int(skeptic_ct > 0),
        hedge_density=density,
    )


def _hedging_text(row) -> str:
    return " ".join(filter(None, [
        row.get("sec_chief_complaint", ""),
        row.get("sec_history_present", ""),
        row.get("sec_hospital_course", ""),
    ]))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5b — GEMMA HEDGING (optional; loads model lazily)
# ─────────────────────────────────────────────────────────────────────────────

_llm = None  # loaded lazily


def _load_gemma(model_path: str):
    global _llm
    if _llm is not None:
        return _llm
    try:
        from llama_cpp import Llama
        print(f"Loading Gemma model from {model_path} …")
        _llm = Llama(model_path=model_path, n_ctx=2048, n_gpu_layers=-1,
                     n_threads=4, n_batch=512, verbose=False)
        print("Gemma model loaded.")
        return _llm
    except ImportError:
        raise RuntimeError("llama-cpp-python is not installed. Run: pip install llama-cpp-python")


def get_gemma_score(text: str, model_path: str) -> float:
    llm = _load_gemma(model_path)
    if not text or not text.strip():
        return 0.0
    try:
        snippet = text[:1700] + ("…" if len(text) > 1700 else "")
        prompt = (
            f"<bos><start_of_turn>user\n"
            f"You are an expert clinical NLP annotator. Analyse this discharge note section "
            f"for hedging language (may, might, possibly, suggests, appears, likely, etc.).\n\n"
            f'Return ONLY a valid JSON object: {{"hedging_score": 0.75}}\n\n'
            f"0.0 = no hedging | 0.3-0.5 = mild | 0.6-0.8 = moderate | 0.9-1.0 = very high\n\n"
            f"Text:\n---\n{snippet}\n---\n<end_of_turn>\n<start_of_turn>model\n"
        )
        output = llm(prompt, max_tokens=120, temperature=0.05, top_p=0.95,
                     stop=["<end_of_turn>", "\n\n"])
        raw = output["choices"][0]["text"].strip()
        raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw, flags=re.MULTILINE | re.DOTALL).strip()
        json_match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if json_match:
            raw = json_match.group(0)
        result = json.loads(raw)
        return max(0.0, min(1.0, float(result.get("hedging_score", 0.0))))
    except Exception:
        return 0.0


def _gemma_hedging_text(row) -> str:
    """Feed Gemma the family history, social history, and chief complaint sections."""
    return " ".join(filter(None, [
        row.get("sec_family_history", ""),
        row.get("sec_social_history", ""),
        row.get("sec_chief_complaint", ""),
    ]))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — COUNTERFACTUAL TEXT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def _swap_pronouns(text: str, swap_map: Dict[str, str]) -> str:
    return _PRONOUN_RE.sub(lambda m: swap_map.get(m.group(), m.group()), text)


def generate_counterfactual(text: str, gender_label: str) -> str:
    if gender_label == "female":
        return _swap_pronouns(text, _F2M)
    elif gender_label == "male":
        return _swap_pronouns(text, _M2F)
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# CORE: PROCESS ONE FILE
# ─────────────────────────────────────────────────────────────────────────────

def process_file(
    csv_path: Path,
    out_dir: Path,
    use_gemma: bool = False,
    gemma_model: Optional[str] = None,
    resume: bool = True,
) -> Path:
    """
    Run the full NLP pipeline on one CSV and write the enriched parquet to out_dir.
    Returns the path to the output parquet file.
    """
    csv_path = Path(csv_path)
    out_dir  = Path(out_dir)
    stem     = csv_path.stem

    file_out_dir = out_dir / stem
    file_out_dir.mkdir(parents=True, exist_ok=True)

    final_parquet = file_out_dir / "nlp_results.parquet"
    final_csv     = file_out_dir / "nlp_results.csv"

    # ── Resume shortcut ──────────────────────────────────────────────────────
    if resume and final_parquet.exists():
        print(f"[SKIP] {stem} — output already exists at {final_parquet}")
        return final_parquet

    ckpt_gender   = file_out_dir / "checkpoint_gender.parquet"
    ckpt_severity = file_out_dir / "checkpoint_severity.parquet"
    ckpt_hedging  = file_out_dir / "checkpoint_hedging.parquet"
    ckpt_gemma    = file_out_dir / "checkpoint_gemma.parquet"

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Processing: {csv_path.name}")
    print(f"{'='*60}")

    t_start = time.time()
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Loaded {len(df):,} rows × {df.shape[1]} columns")
    df.columns = [c.upper() for c in df.columns]
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)

    # ── 2. Text cleaning & section parsing ───────────────────────────────────
    print("  Parsing sections …")
    parsed     = df[TEXT_COL].progress_apply(parse_row)
    parsed_df  = pd.json_normalize(parsed)
    df         = pd.concat([df.reset_index(drop=True), parsed_df], axis=1)

    # ── 3. Gender ─────────────────────────────────────────────────────────────
    if ckpt_gender.exists():
        print(f"  [resume] Loading gender checkpoint")
        gender_df = pd.read_parquet(ckpt_gender)
    else:
        print("  Running gender analysis …")
        t0 = time.time()
        gender_results = df["cleaned_text"].progress_apply(analyse_gender)
        gender_df      = pd.json_normalize(gender_results)
        gender_df.to_parquet(ckpt_gender, index=False)
        print(f"    Done in {time.time()-t0:.1f}s")

    df = pd.concat([df.reset_index(drop=True), gender_df], axis=1)

    # ── 4. Severity ───────────────────────────────────────────────────────────
    if ckpt_severity.exists():
        print(f"  [resume] Loading severity checkpoint")
        severity_df = pd.read_parquet(ckpt_severity)
    else:
        print("  Running severity analysis …")
        t0      = time.time()
        sev_txt = df.progress_apply(_severity_text, axis=1)
        severity_results = sev_txt.progress_apply(analyse_severity)
        severity_df      = pd.json_normalize(severity_results)
        severity_df.to_parquet(ckpt_severity, index=False)
        print(f"    Done in {time.time()-t0:.1f}s")

    df = pd.concat([df.reset_index(drop=True), severity_df], axis=1)

    # ── 5. Hedging (rule-based) ────────────────────────────────────────────────
    if ckpt_hedging.exists():
        print(f"  [resume] Loading hedging checkpoint")
        hedging_df = pd.read_parquet(ckpt_hedging)
    else:
        print("  Running hedging analysis …")
        t0        = time.time()
        hedge_txt = df.progress_apply(_hedging_text, axis=1)
        hedging_results = hedge_txt.progress_apply(analyse_hedging)
        hedging_df      = pd.json_normalize(hedging_results)
        hedging_df.to_parquet(ckpt_hedging, index=False)
        print(f"    Done in {time.time()-t0:.1f}s")

    existing_hedge = [c for c in df.columns if c.startswith("hedge_")]
    if existing_hedge:
        df = df.drop(columns=existing_hedge)
    df = pd.concat([df.reset_index(drop=True), hedging_df.reset_index(drop=True)], axis=1)

    # ── 5b. Gemma hedging (optional) ──────────────────────────────────────────
    if use_gemma and gemma_model:
        if ckpt_gemma.exists():
            print(f"  [resume] Loading Gemma checkpoint")
            gemma_df = pd.read_parquet(ckpt_gemma)
        else:
            print("  Running Gemma hedging analysis …")
            t0       = time.time()
            g_text   = df.progress_apply(_gemma_hedging_text, axis=1)
            scores   = g_text.progress_apply(lambda t: get_gemma_score(t, gemma_model))
            gemma_df = pd.DataFrame({"gemma_hedging_score": scores})
            gemma_df.to_parquet(ckpt_gemma, index=False)
            print(f"    Done in {time.time()-t0:.1f}s")

        if "gemma_hedging_score" in df.columns:
            df = df.drop(columns=["gemma_hedging_score"])
        df = pd.concat([df.reset_index(drop=True), gemma_df.reset_index(drop=True)], axis=1)

    # ── 9. Counterfactual text ─────────────────────────────────────────────────
    print("  Generating counterfactual texts …")
    mask = df["gender_label"].isin(["female", "male"])
    df["counterfactual_text"] = ""
    df.loc[mask, "counterfactual_text"] = df[mask].progress_apply(
        lambda r: generate_counterfactual(r["cleaned_text"], r["gender_label"]), axis=1
    )

    # ── 6. Column ordering & save ──────────────────────────────────────────────
    ID_AND_LABEL  = [c for c in df.columns if c in [
        "SUBJECT_ID", "HADM_ID", "NOTE_SEQ", "ADMITTIME", "DISCHTIME",
        "OUTPUT_LABEL", "ADMISSION_TYPE", "INSURANCE", "LANGUAGE",
        "MARITAL_STATUS", "RACE", "ADMISSION_LOCATION", "DISCHARGE_LOCATION",
    ]]
    GENDER_COLS   = [c for c in df.columns if c.startswith("gender_") or c == "header_sex"]
    SEVERITY_COLS = [c for c in df.columns if c.startswith("severity_")]
    HEDGING_COLS  = [c for c in df.columns if c.startswith("hedge_") or c == "gemma_hedging_score"]
    SECTION_COLS  = [c for c in df.columns if c.startswith("sec_")]
    TEXT_COLS     = ["cleaned_text", "counterfactual_text"]

    ordered_cols = ID_AND_LABEL + GENDER_COLS + SEVERITY_COLS + HEDGING_COLS + SECTION_COLS + TEXT_COLS
    extra        = [c for c in df.columns if c not in ordered_cols]
    final_df     = df[ordered_cols + extra]

    # Add a source column so we know which file each row came from after aggregation
    final_df.insert(0, "_source_file", csv_path.name)

    final_df.to_parquet(final_parquet, index=False)
    final_df.to_csv(final_csv, index=False)

    # ── Focused output: text + counterfactual + key IDs ───────────────────────
    focus_cols = [c for c in ["HADM_ID", "OUTPUT_LABEL", "cleaned_text", "counterfactual_text"]
                  if c in final_df.columns]
    focus_csv  = file_out_dir / "final_text_output.csv"
    final_df[focus_cols].to_csv(focus_csv, index=False)
    print(f"  Saved focused CSV   → {focus_csv}")

    elapsed = time.time() - t_start
    print(f"  Saved → {final_parquet}  ({final_parquet.stat().st_size / 1e6:.1f} MB)")
    print(f"  Total time for {csv_path.name}: {elapsed:.1f}s")

    return final_parquet


# ─────────────────────────────────────────────────────────────────────────────
# BATCH PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def _worker(args):
    """Unpacks args tuple for multiprocessing (top-level so it's picklable)."""
    csv_path, out_dir, use_gemma, gemma_model, resume = args
    try:
        return str(process_file(csv_path, out_dir, use_gemma, gemma_model, resume)), None
    except Exception as e:
        return str(csv_path), str(e)


def process_batch(
    csv_dir: Path,
    out_dir: Path,
    use_gemma: bool = False,
    gemma_model: Optional[str] = None,
    workers: int = 1,
    resume: bool = True,
) -> List[Path]:
    """
    Process every CSV in csv_dir.  With workers > 1 files are processed in
    parallel using separate processes (safe: each worker gets its own spaCy
    and regex objects).

    NOTE: Gemma model loading in multiprocessing requires workers=1 (the model
    cannot be safely shared across processes).  The function will force
    workers=1 when use_gemma=True.
    """
    csv_files = sorted(Path(csv_dir).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")

    print(f"Found {len(csv_files)} CSV files to process.")

    if use_gemma and workers > 1:
        print("WARNING: Gemma analysis is incompatible with parallel workers. Falling back to workers=1.")
        workers = 1

    out_parquets: List[Path] = []

    if workers == 1:
        for csv_path in tqdm(csv_files, desc="Files", unit="file"):
            out_path = process_file(csv_path, out_dir, use_gemma, gemma_model, resume)
            out_parquets.append(out_path)
    else:
        args_list = [(csv, out_dir, use_gemma, gemma_model, resume) for csv in csv_files]
        print(f"Spawning {workers} parallel workers …")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_worker, a): a[0] for a in args_list}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Files"):
                result_path, error = future.result()
                if error:
                    print(f"\n[ERROR] {futures[future].name}: {error}")
                else:
                    out_parquets.append(Path(result_path))

    return out_parquets


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_results(out_dir: Path) -> pd.DataFrame:
    """
    Find all per-file nlp_results.parquet files under out_dir and concatenate
    them into a single DataFrame.  Saves two aggregate artefacts:
      - aggregate/all_files_nlp_results.parquet
      - aggregate/all_files_nlp_results.csv
      - aggregate/summary_stats.csv          (per-file summary)
      - aggregate/gender_bias_report.csv     (cross-file bias metrics)
    """
    parquet_files = sorted(out_dir.rglob("nlp_results.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No nlp_results.parquet files found under {out_dir}")

    print(f"\nAggregating {len(parquet_files)} result files …")

    frames = []
    per_file_summaries = []

    for pf in tqdm(parquet_files, desc="Loading", unit="file"):
        df = pd.read_parquet(pf)
        frames.append(df)

        # Per-file summary statistics
        summary = {"source_file": pf.parent.name, "n_rows": len(df)}
        for col in ["severity_score", "hedge_total_ct", "hedge_density",
                    "gender_confidence", "hedge_skepticism_flag", "gemma_hedging_score"]:
            if col in df.columns:
                summary[f"{col}_mean"] = round(df[col].mean(), 4)
        if "gender_label" in df.columns:
            vc = df["gender_label"].value_counts(normalize=True)
            for g in ["female", "male", "unknown", "ambiguous"]:
                summary[f"pct_{g}"] = round(vc.get(g, 0.0), 4)
        if "OUTPUT_LABEL" in df.columns:
            summary["readmit_rate"] = round(df["OUTPUT_LABEL"].mean(), 4)
        per_file_summaries.append(summary)

    aggregate_dir = out_dir / "aggregate"
    aggregate_dir.mkdir(exist_ok=True)

    # ── Concatenate ──────────────────────────────────────────────────────────
    all_df = pd.concat(frames, ignore_index=True)
    print(f"  Total rows: {len(all_df):,}  |  Columns: {all_df.shape[1]}")

    # ── Save aggregate data ───────────────────────────────────────────────────
    agg_parquet = aggregate_dir / "all_files_nlp_results.parquet"
    agg_csv     = aggregate_dir / "all_files_nlp_results.csv"
    all_df.to_parquet(agg_parquet, index=False)
    all_df.to_csv(agg_csv, index=False)
    print(f"  Saved aggregate parquet → {agg_parquet} ({agg_parquet.stat().st_size / 1e6:.1f} MB)")
    print(f"  Saved aggregate CSV     → {agg_csv}")

    # ── Per-file summary ──────────────────────────────────────────────────────
    summary_df = pd.DataFrame(per_file_summaries)
    summary_path = aggregate_dir / "summary_stats.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved per-file summary  → {summary_path}")

    # ── Focused aggregate: text + counterfactual + key IDs ───────────────────
    focus_cols = [c for c in ["_source_file", "HADM_ID", "OUTPUT_LABEL",
                               "cleaned_text", "counterfactual_text"]
                  if c in all_df.columns]
    focus_agg_csv = aggregate_dir / "final_text_output.csv"
    all_df[focus_cols].to_csv(focus_agg_csv, index=False)
    print(f"  Saved focused CSV       → {focus_agg_csv}")

    # ── Cross-file gender bias report ─────────────────────────────────────────
    _write_bias_report(all_df, aggregate_dir)

    return all_df


def _write_bias_report(df: pd.DataFrame, out_dir: Path):
    """Write a gender bias summary table and, optionally, plots."""
    if "gender_label" not in df.columns:
        return

    bias_path = out_dir / "gender_bias_report.csv"
    bg = df[df["gender_label"].isin(["female", "male"])].copy()

    if bg.empty:
        return

    metrics = {}
    for g in ["female", "male"]:
        sub = bg[bg["gender_label"] == g]
        m   = {"N": len(sub)}
        for col in ["severity_score", "severity_high_ct", "hedge_total_ct",
                    "hedge_skepticism_ct", "hedge_density", "hedge_skepticism_flag",
                    "gender_confidence"]:
            if col in sub.columns:
                m[f"{col}_mean"] = round(sub[col].mean(), 4)
        if "OUTPUT_LABEL" in sub.columns:
            m["readmission_rate"] = round(sub["OUTPUT_LABEL"].mean(), 4)
        if "gemma_hedging_score" in sub.columns:
            m["gemma_hedging_mean"] = round(sub["gemma_hedging_score"].mean(), 4)
        metrics[g] = m

    bias_df = pd.DataFrame(metrics).T
    bias_df.index.name = "gender"
    bias_df.to_csv(bias_path)
    print(f"  Saved gender bias report→ {bias_path}")

    # Numeric disparity ratios (female / male)
    print("\n  Gender disparity ratios (female / male):")
    numeric_cols = bias_df.select_dtypes(include="number").columns.tolist()
    for col in numeric_cols:
        if "female" in bias_df.index and "male" in bias_df.index:
            f_val = bias_df.loc["female", col]
            m_val = bias_df.loc["male", col]
            if m_val and m_val != 0:
                ratio = f_val / m_val
                flag  = " ← DISPARITY" if abs(ratio - 1.0) > 0.10 else ""
                print(f"    {col:35s}: {ratio:.3f}{flag}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="NarrativeGuard NLP pipeline — single file or batch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--input",      type=Path, help="Path to a single CSV file")
    mode.add_argument("--input-dir",  type=Path, help="Directory containing CSV files")
    mode.add_argument("--aggregate-only", action="store_true",
                      help="Skip processing; merge existing output parquets only")

    p.add_argument("--output-dir", type=Path, default=Path("outputs"),
                   help="Root output directory (default: ./outputs)")
    p.add_argument("--use-gemma",  action="store_true",
                   help="Enable Gemma LLM hedging analysis")
    p.add_argument("--gemma-model", type=str,
                   default="/content/google_gemma-3-1b-it-Q4_K_M.gguf",
                   help="Path to the GGUF model file")
    p.add_argument("--workers",  type=int, default=1,
                   help="Parallel workers for batch mode (default: 1, Gemma forces 1)")
    p.add_argument("--resume",   action="store_true", default=True,
                   help="Skip files whose output already exists (default: True)")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="Force reprocessing even if output already exists")
    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.time()

    if args.aggregate_only:
        aggregate_results(args.output_dir)

    elif args.input:
        if not args.input.exists():
            parser.error(f"File not found: {args.input}")
        process_file(
            csv_path    = args.input,
            out_dir     = args.output_dir,
            use_gemma   = args.use_gemma,
            gemma_model = args.gemma_model,
            resume      = args.resume,
        )
        # Auto-aggregate even for a single file (creates the aggregate/ folder)
        aggregate_results(args.output_dir)

    elif args.input_dir:
        if not args.input_dir.is_dir():
            parser.error(f"Not a directory: {args.input_dir}")
        process_batch(
            csv_dir     = args.input_dir,
            out_dir     = args.output_dir,
            use_gemma   = args.use_gemma,
            gemma_model = args.gemma_model,
            workers     = args.workers,
            resume      = args.resume,
        )
        aggregate_results(args.output_dir)

    else:
        parser.print_help()
        sys.exit(1)

    print(f"\nTotal wall time: {time.time() - t_total:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()