# NarrativeGuard: Mitigating Gender Bias in Clinical NLP Models

## Overview
NarrativeGuard is a fairness-focused NLP framework designed to detect and mitigate **gender-based narrative bias** in clinical text used for **hospital readmission prediction**.

Modern transformer-based models like ClinicalBERT achieve high predictive performance, but they may unintentionally learn **spurious correlations** from biased clinical documentation. This project investigates how **subjective language in discharge summaries** impacts model predictions and introduces methods to improve fairness without sacrificing clinical relevance.

---

## Motivation
Clinical notes often reflect subtle linguistic biases:
- Women’s symptoms may be documented with **less certainty or more hedging**
- Subjective phrasing (e.g., *"claims pain"* vs *"reports pain"*) can influence model interpretation

This leads to a critical question:

> Do NLP models learn **clinical signals**, or do they learn **biased narrative patterns**?

NarrativeGuard addresses this by combining **fairness evaluation, causal analysis, and data augmentation**.

---

## Dataset
- **MIMIC-IV-Note Dataset**
  - 300K+ discharge summaries
  - Real-world ICU patient records

### Key Variables
- Clinical text (discharge summaries)
- 30-day readmission label (binary)
- Gender (protected attribute)

---

## Methodology

### 1. Feature Extraction (NLP Pipeline)

We extract three categories of linguistic signals:

#### Gender Markers
- Pronoun-based extraction using regex + spaCy dependency parsing

#### Severity Indicators
- Clinical keyword detection (e.g., *acute, severe, mild*)
- UMLS-based entity linking

#### Hedging / Confidence Signals
- Rule-based + LLM-based detection of uncertainty and skepticism  
- Example: *"likely", "possibly", "claims"*

---

### 2. Bias Evaluation

We evaluate fairness using standard metrics:

#### Equalized Odds
- Compare True Positive Rate (TPR) and False Positive Rate (FPR) across genders

#### Recall Parity
- Measures disparity in detecting true readmissions

#### Key Finding
- ~22% recall gap between male and female patients  
→ Indicates under-detection of high-risk female patients

---

### 3. Counterfactual Analysis

To isolate bias:

- Generate **gender-swapped versions** of clinical notes
- Keep all medical information constant
- Measure prediction consistency

This helps determine whether:
- Predictions depend on **clinical signals**
- Or on **gendered language patterns**

---

### 4. Bias Mitigation

We propose **Counterfactual Data Augmentation**:

- Gender-swapped text
- Neutralized subjective language
- Combined transformations

**Goal:**  
Encourage the model to learn **medical relevance**, not **linguistic bias**

---

## Results

- Significant disparity in:
  - True Positive Rate (TPR)
  - Recall across genders
- Higher **hedging density** observed in female patient notes
- Evidence that models rely on **non-clinical linguistic signals**

---

## Tech Stack

- **NLP**: spaCy, regex, UMLS  
- **Models**: ClinicalBERT, LLM-based evaluation  
- **Data Processing**: Pandas, NumPy  
- **Explainability**: LIME  
- **Dataset**: MIMIC-IV  

---

## Key Contributions

- Identified **narrative bias** in clinical ML pipelines  
- Designed **counterfactual fairness evaluation**  
- Built **multi-stage NLP pipeline** for linguistic feature extraction  
- Proposed **data-centric bias mitigation strategy**

---

## Future Work

- Extend to other protected attributes (race, insurance type)  
- Deploy fairness-aware training objectives  
- Evaluate real-world clinical decision impact  
- Integrate causal inference frameworks  

---

## Why This Matters

Bias in healthcare AI is not just a technical issue—it has real consequences:

- Underestimating patient risk  
- Unequal treatment decisions  
- Reduced trust in AI systems  

NarrativeGuard aims to build **fair, reliable, and clinically meaningful AI systems**.
