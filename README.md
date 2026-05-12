# Resume Matching Engine
**Redrob AI Campus Hackathon — Individual Competition**

A from-scratch Resume Matching Engine that ranks candidates against job descriptions using TF-IDF vectors and cosine similarity — **no external libraries** (pure Python stdlib only).

---

## Problem Summary

Given 10 Indian university student resumes with noisy skill data and 3 Job Descriptions from Korean tech companies (Kakao, Naver, Line), build a program that:

1. **Normalizes** noisy resume skills (typos, aliases, multi-word phrases)
2. **Deduplicates** canonical skills per resume
3. **Builds a shared vocabulary** from all resume skills (alphabetically sorted)
4. **Computes TF-IDF vectors** for each resume
5. **Builds binary vectors** for each JD over the same vocabulary
6. **Ranks Top-3 candidates** per JD using cosine similarity

---

## Results

```
JD1 — Kakao (Seoul) (ML Engineer)
Sneha Patel(0.57), Karan Mehta(0.53), Arjun Sharma(0.40)

JD2 — Naver (Seongnam) (Backend Engineer)
Rahul Gupta(0.81), Ananya Krishnan(0.28), Deepika Rao(0.19)

JD3 — Line (Seoul) (Frontend Engineer)
Aditya Kumar(0.67), Priya Nair(0.58), Ananya Krishnan(0.35)
```

---

## Formulas Used

```
TF(skill, resume) = 1 / N
    N = total unique skills in resume (after deduplication)

IDF(skill) = ln( 10 / df(skill) )
    df(skill) = number of resumes containing the skill
    No smoothing. Natural logarithm.

TF-IDF = TF × IDF

Cosine(A, B) = (A · B) / (|A| × |B|)
    A = Resume TF-IDF vector
    B = JD binary vector
    |A| = Euclidean norm of A
```

---

## Pipeline

| Step | Task | Description |
|------|------|-------------|
| 1 | Normalize Skills | Split on commas → lowercase → alias map → discard unknowns |
| 2 | Deduplicate | Each canonical skill appears at most once per resume |
| 3 | Build Vocabulary | Union of all resume skills, sorted alphabetically (48 skills) |
| 4 | Compute TF-IDF | TF = 1/N, IDF = ln(10/df), no external libs |
| 5 | Build JD Vectors | Binary 0/1 over same vocabulary |
| 6 | Rank | Cosine similarity → Top 3 per JD, ties broken alphabetically |

---

## Normalization Details

Key normalizations handled:

| Raw Token | Canonical |
|-----------|-----------|
| `Pyhton` | `python` |
| `MachineLearning` | `machine_learning` |
| `Deep-learning` | `deep_learning` |
| `Spring Boot` | `spring_boot` |
| `REST api` | `rest_api` |
| `kubernates` | `kubernetes` |
| `matplotlib`, `tableau`, `Power-BI` | `data_visualization` |
| `Sklearn` | `machine_learning` |
| `Algoritms` | `algorithms` |
| `Data Structure` | `data_structures` |
| `TypeScrpit` | `typescript` |

Multi-word phrases (e.g. `Spring Boot`, `feature engineering`, `competitive programming`) survive comma-splitting as single tokens and are matched directly in the alias map.

---

## Vocabulary (48 skills)

```
algorithms, android, aws, bert, ci_cd, clustering, competitive_programming, cpp,
data_structures, data_visualization, deep_learning, docker, feature_engineering,
figma, firebase, flask, graphql, html_css, java, javascript, jest, keras, kotlin,
kubernetes, machine_learning, microservices, mongodb, mysql, nlp, nodejs, numpy,
pandas, postgresql, python, r, react, redux, regression, rest_api, spring_boot,
sql, statistics, tailwind, tensorflow, typescript, ui_ux, vue, xgboost
```

Note: `pytorch` and `redis` appear in JD skills but not in any resume — they are **not** in the vocabulary and score 0 in JD vectors.

---

## Usage

```bash
python resume_matching_engine.py
```

**Requirements:** Python 3.10+, stdlib only (`math`, `collections`)

---

## Constraints

- No external libraries (numpy, pandas, scikit-learn, etc.)
- SKILL_ALIASES used exactly as specified
- IDF uses natural log, no smoothing
- TF uses post-deduplication count (always 1/N)
- Ties broken alphabetically by candidate name

---

## File Structure

```
resume-matching-engine/
├── resume_matching_engine.py   # Complete solution
└── README.md                   # This file
```
