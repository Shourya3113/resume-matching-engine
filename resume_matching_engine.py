"""
Resume Matching Engine
Redrob AI Campus Hackathon — Individual Competition

Pipeline:
  1. Normalize noisy resume skill data
  2. Deduplicate skills per resume
  3. Build shared vocabulary (alphabetically sorted)
  4. Compute TF-IDF vectors for resumes
  5. Build binary vectors for JDs over the same vocabulary
  6. Compute cosine similarity and rank Top-3 per JD

Formulas:
  TF(skill, resume) = 1 / N           (N = total unique skills after dedup)
  IDF(skill)        = ln(10 / df)      (df = resumes containing skill; no smoothing)
  TF-IDF            = TF × IDF
  Cosine(A, B)      = (A · B) / (|A| × |B|)

Author: Shourya Solanki
"""

import math
from collections import OrderedDict


# ─────────────────────────────────────────────
#  SKILL ALIAS MAP  (exact as provided in spec)
# ─────────────────────────────────────────────
SKILL_ALIASES = {
    # Languages
    "python":               "python",
    "pyhton":               "python",
    "java":                 "java",
    "javascript":           "javascript",
    "javascrpit":           "javascript",
    "js":                   "javascript",
    "typescript":           "typescript",
    "typescrpit":           "typescript",
    "c++":                  "cpp",
    "cpp":                  "cpp",
    "r":                    "r",
    "kotlin":               "kotlin",

    # ML / Data
    "machinelearning":      "machine_learning",
    "machine learning":     "machine_learning",
    "ml":                   "machine_learning",
    "sklearn":              "machine_learning",
    "deeplearning":         "deep_learning",
    "deep learning":        "deep_learning",
    "deep-learning":        "deep_learning",
    "tensorflow":           "tensorflow",
    "pytorch":              "pytorch",
    "keras":                "keras",
    "nlp":                  "nlp",
    "bert":                 "bert",
    "xgboost":              "xgboost",
    "feature engineering":  "feature_engineering",
    "statistics":           "statistics",
    "stats":                "statistics",
    "regression":           "regression",
    "clustering":           "clustering",
    "data-viz":             "data_visualization",
    "data visualization":   "data_visualization",
    "data viz":             "data_visualization",
    "matplotlib":           "data_visualization",
    "tableau":              "data_visualization",
    "power-bi":             "data_visualization",
    "power bi":             "data_visualization",
    "powerbi":              "data_visualization",
    "pandas":               "pandas",
    "numpy":                "numpy",

    # Web — Frontend
    "react":                "react",
    "reacts":               "react",
    "reactjs":              "react",
    "vue":                  "vue",
    "vue.js":               "vue",
    "vuejs":                "vue",
    "redux":                "redux",
    "tailwind":             "tailwind",
    "html/css":             "html_css",
    "html css":             "html_css",
    "html":                 "html_css",
    "css":                  "html_css",
    "jest":                 "jest",
    "graphql":              "graphql",

    # Web — Backend
    "node.js":              "nodejs",
    "nodejs":               "nodejs",
    "node js":              "nodejs",
    "flask":                "flask",
    "spring boot":          "spring_boot",
    "springboot":           "spring_boot",
    "rest api":             "rest_api",
    "rest":                 "rest_api",
    "restapi":              "rest_api",
    "microservices":        "microservices",

    # Databases
    "sql":                  "sql",
    "mysql":                "mysql",
    "mysq":                 "mysql",
    "postgresql":           "postgresql",
    "postgres":             "postgresql",
    "mongodb":              "mongodb",
    "redis":                "redis",

    # DevOps / Cloud
    "docker":               "docker",
    "kubernetes":           "kubernetes",
    "kubernates":           "kubernetes",
    "k8s":                  "kubernetes",
    "ci/cd":                "ci_cd",
    "cicd":                 "ci_cd",
    "ci cd":                "ci_cd",
    "aws":                  "aws",

    # Mobile
    "android":              "android",
    "firebase":             "firebase",

    # CS Fundamentals
    "algorithms":           "algorithms",
    "algoritms":            "algorithms",
    "data structure":       "data_structures",
    "data structures":      "data_structures",
    "competitive programming": "competitive_programming",

    # Design
    "ui/ux":                "ui_ux",
    "ui ux":                "ui_ux",
    "figma":                "figma",
}

# ─────────────────────────────────────────────
#  RESUME DATASET  (10 candidates)
# ─────────────────────────────────────────────
RESUMES = [
    {"id": "01", "name": "Arjun Sharma",
     "raw": "Pyhton, MachineLearning, SQL, pandas, numpy, Deep-learning"},
    {"id": "02", "name": "Priya Nair",
     "raw": "JavaScrpit, Reacts, Node.JS, MongoDb, REST api, HTML/CSS"},
    {"id": "03", "name": "Rahul Gupta",
     "raw": "Java, Spring Boot, MySql, Microservices, Docker, kubernates"},
    {"id": "04", "name": "Sneha Patel",
     "raw": "Python, TensorFlow, Keras, NLP, BERT, data-viz, matplotlib"},
    {"id": "05", "name": "Vikram Singh",
     "raw": "C++, Algoritms, Data Structure, competitive programming, python"},
    {"id": "06", "name": "Ananya Krishnan",
     "raw": "javascript, vue.js, python, flask, PostgreSQL, AWS, CI/CD"},
    {"id": "07", "name": "Karan Mehta",
     "raw": "Python, Sklearn, XGboost, feature engineering, SQL, tableau"},
    {"id": "08", "name": "Deepika Rao",
     "raw": "Java, Android, Kotlin, Firebase, REST, UI/UX, figma"},
    {"id": "09", "name": "Aditya Kumar",
     "raw": "Reactjs, TypeScrpit, GraphQL, redux, tailwind, nodejs, jest"},
    {"id": "10", "name": "Meera Iyer",
     "raw": "python, R, statistics, ML, regression, clustering, Power-BI"},
]

# ─────────────────────────────────────────────
#  JOB DESCRIPTION DATASET  (3 JDs)
# ─────────────────────────────────────────────
JDS = [
    {
        "id": "JD1", "company": "Kakao (Seoul)", "role": "ML Engineer",
        "skills": ("Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, "
                   "SQL, Data Visualization, NLP, BERT, Feature Engineering, Statistics"),
    },
    {
        "id": "JD2", "company": "Naver (Seongnam)", "role": "Backend Engineer",
        "skills": ("Java, Spring Boot, MySQL, PostgreSQL, Microservices, "
                   "Docker, Kubernetes, REST API, CI/CD, Redis"),
    },
    {
        "id": "JD3", "company": "Line (Seoul)", "role": "Frontend Engineer",
        "skills": ("JavaScript, React, Vue, TypeScript, REST API, HTML/CSS, "
                   "Node.js, GraphQL, Redux, Jest, AWS"),
    },
]


# ─────────────────────────────────────────────
#  STEP 1 & 2: NORMALIZE + DEDUPLICATE
# ─────────────────────────────────────────────
def normalize_skills(raw_string: str) -> list[str]:
    """
    1. Split on commas
    2. Strip whitespace, lowercase each token
    3. Look up in SKILL_ALIASES (multi-word phrases match because they survive
       comma-splitting as single tokens, e.g. 'Spring Boot', 'REST api')
    4. Discard tokens not in the alias map
    5. Deduplicate while preserving first-seen order
    """
    tokens = [t.strip().lower() for t in raw_string.split(",")]
    seen   = OrderedDict()
    for token in tokens:
        canonical = SKILL_ALIASES.get(token)
        if canonical and canonical not in seen:
            seen[canonical] = True
    return list(seen.keys())


# ─────────────────────────────────────────────
#  STEP 3: BUILD VOCABULARY
# ─────────────────────────────────────────────
def build_vocabulary(normalized_resumes: list[list[str]]) -> list[str]:
    """Collect all unique canonical skills from resumes, sort alphabetically."""
    vocab_set = set()
    for skills in normalized_resumes:
        vocab_set.update(skills)
    return sorted(vocab_set)


# ─────────────────────────────────────────────
#  STEP 4: TF-IDF VECTORS
# ─────────────────────────────────────────────
def compute_tfidf(normalized_resumes: list[list[str]],
                  vocab: list[str]) -> list[list[float]]:
    """
    For each resume compute a TF-IDF vector over the shared vocabulary.
      TF(skill, resume) = 1 / N          (N = unique skills in resume)
      IDF(skill)        = ln(10 / df)    (df = #resumes containing skill)
      TF-IDF            = TF × IDF
    """
    N_docs = len(normalized_resumes)        # always 10
    vocab_idx = {skill: i for i, skill in enumerate(vocab)}

    # Document frequency
    df = {skill: 0 for skill in vocab}
    for skills in normalized_resumes:
        for skill in skills:
            if skill in df:
                df[skill] += 1

    # IDF per skill
    idf = {skill: math.log(N_docs / df[skill]) for skill in vocab if df[skill] > 0}

    # TF-IDF matrix
    tfidf_matrix = []
    for skills in normalized_resumes:
        N = len(skills)
        tf = 1.0 / N
        vec = [0.0] * len(vocab)
        for skill in skills:
            if skill in vocab_idx:
                vec[vocab_idx[skill]] = tf * idf[skill]
        tfidf_matrix.append(vec)

    return tfidf_matrix


# ─────────────────────────────────────────────
#  STEP 5: JD BINARY VECTORS
# ─────────────────────────────────────────────
def build_jd_vector(jd_skills_raw: str, vocab: list[str]) -> list[int]:
    """
    Normalize JD skills using the same alias map, then build a binary vector
    over the shared vocabulary (1 if skill in JD and in vocab, else 0).
    """
    vocab_idx  = {skill: i for i, skill in enumerate(vocab)}
    jd_skills  = normalize_skills(jd_skills_raw)
    vec        = [0] * len(vocab)
    for skill in jd_skills:
        if skill in vocab_idx:
            vec[vocab_idx[skill]] = 1
    return vec


# ─────────────────────────────────────────────
#  STEP 6: COSINE SIMILARITY
# ─────────────────────────────────────────────
def cosine_similarity(a: list[float], b: list[int]) -> float:
    """
    Cosine(A, B) = (A · B) / (|A| × |B|)
    |A| = Euclidean norm of TF-IDF vector
    |B| = Euclidean norm of binary JD vector = sqrt(#skills in JD ∩ vocab)
    Returns 0.0 if either norm is zero.
    """
    dot   = sum(ai * bi for ai, bi in zip(a, b))
    norm_a = math.sqrt(sum(ai * ai for ai in a))
    norm_b = math.sqrt(sum(bi * bi for bi in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ─────────────────────────────────────────────
#  MAIN: RANK TOP-3 PER JD
# ─────────────────────────────────────────────
def rank_candidates(tfidf_matrix, jd_vec, names):
    scores = []
    for i, vec in enumerate(tfidf_matrix):
        sim = cosine_similarity(vec, jd_vec)
        scores.append((names[i], sim))
    # Sort: descending score, then alphabetical name for ties
    scores.sort(key=lambda x: (-x[1], x[0]))
    return scores[:3]


def main():
    # ── Normalize & deduplicate all resumes
    normalized = [normalize_skills(r["raw"]) for r in RESUMES]
    names      = [r["name"] for r in RESUMES]

    print("=" * 60)
    print("STEP 1–2: Normalized & Deduplicated Skills")
    print("=" * 60)
    for r, skills in zip(RESUMES, normalized):
        print(f"  {r['id']} {r['name']:20s} ({len(skills)} skills): {skills}")

    # ── Build vocabulary
    vocab = build_vocabulary(normalized)
    print(f"\nSTEP 3: Vocabulary — {len(vocab)} unique skills (alphabetical)")
    print(f"  {vocab}")

    # ── Compute TF-IDF
    tfidf_matrix = compute_tfidf(normalized, vocab)

    print("\nSTEP 4: TF-IDF Vectors (non-zero entries only)")
    vocab_idx = {s: i for i, s in enumerate(vocab)}
    for r, skills, vec in zip(RESUMES, normalized, tfidf_matrix):
        print(f"  {r['name']}:")
        for skill in skills:
            idx = vocab_idx[skill]
            print(f"    {skill:30s} = {vec[idx]:.6f}")

    # ── Build JD vectors & rank
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    for jd in JDS:
        jd_vec  = build_jd_vector(jd["skills"], vocab)
        jd_skills_active = [vocab[i] for i, v in enumerate(jd_vec) if v]
        top3    = rank_candidates(tfidf_matrix, jd_vec, names)

        print(f"\n{jd['id']} — {jd['company']} ({jd['role']})")
        print(f"  JD vocab skills ({len(jd_skills_active)}): {jd_skills_active}")
        result_str = ", ".join(f"{name}({score:.2f})" for name, score in top3)
        print(f"  {result_str}")


if __name__ == "__main__":
    main()
