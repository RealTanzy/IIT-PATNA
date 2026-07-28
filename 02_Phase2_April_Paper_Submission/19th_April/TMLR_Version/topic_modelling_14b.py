"""
NMF Topic Modelling on 14B Qwen-2.5 paired results (3,659 questions).
Outputs: per-topic accuracy tables, NPMI coherence, difficulty categorization,
and example questions for the TMLR paper revision.
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from itertools import combinations

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(BASE))
DATA = os.path.join(REPO, "14b_dataset_check", "results_14b")


def load_json(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    return data


def load_paired():
    """Load and pair all 14B results. Returns list of dicts."""
    paired = []

    # StrategyQA: join on question text
    sq_astar = load_json(os.path.join(DATA, "strategyqa", "astar", "results.json"))
    sq_cot = load_json(os.path.join(DATA, "strategyqa", "cot", "results.json"))
    sq_cot_lookup = {r["question"]: r for r in sq_cot}
    for r in sq_astar:
        q = r["question"]
        if q in sq_cot_lookup:
            cr = sq_cot_lookup[q]
            paired.append({
                "text": q,
                "dataset": "StrategyQA",
                "astar_correct": r["correct"],
                "cot_correct": cr["correct"],
                "gold": r["gold"],
                "astar_pred": r["pred"],
                "cot_pred": cr["pred"],
            })

    # HotpotQA: join on question text
    hp_astar = load_json(os.path.join(DATA, "hotpotqa", "astar", "results.json"))
    hp_cot = load_json(os.path.join(DATA, "hotpotqa", "cot", "results.json"))
    hp_cot_lookup = {r["question"]: r for r in hp_cot}
    for r in hp_astar:
        q = r["question"]
        if q in hp_cot_lookup:
            cr = hp_cot_lookup[q]
            paired.append({
                "text": q,
                "dataset": "HotpotQA",
                "astar_correct": r["correct"],
                "cot_correct": cr["correct"],
                "gold": r["gold"],
                "astar_pred": r["pred"],
                "cot_pred": cr["pred"],
            })

    # LogicQA: join on id, use context as text
    lq_astar = load_json(os.path.join(DATA, "logicqa", "astar", "results.json"))
    lq_cot = load_json(os.path.join(DATA, "logicqa", "cot", "results.json"))
    lq_cot_lookup = {r["id"]: r for r in lq_cot}
    for r in lq_astar:
        rid = r["id"]
        if rid in lq_cot_lookup:
            cr = lq_cot_lookup[rid]
            paired.append({
                "text": r.get("context", ""),
                "dataset": "LogicQA",
                "astar_correct": r["correct"],
                "cot_correct": cr["correct"],
                "gold": str(r["gold"]),
                "astar_pred": str(r["pred"]),
                "cot_pred": str(cr.get("pred", "")),
            })

    return paired


def compute_npmi(tfidf_matrix, feature_names, top_terms_per_topic, n_top=10):
    """Compute NPMI coherence with Laplace smoothing (standard for short texts)."""
    binary = (tfidf_matrix > 0).astype(int)
    n_docs = binary.shape[0]
    vocab_to_idx = {w: i for i, w in enumerate(feature_names)}

    topic_scores = []
    for terms in top_terms_per_topic:
        terms = terms[:n_top]
        pair_scores = []
        for w1, w2 in combinations(terms, 2):
            i1, i2 = vocab_to_idx.get(w1), vocab_to_idx.get(w2)
            if i1 is None or i2 is None:
                continue
            col1 = np.array(binary[:, i1].todense()).flatten()
            col2 = np.array(binary[:, i2].todense()).flatten()
            co = np.sum(col1 * col2)
            p1 = np.sum(col1) / n_docs
            p2 = np.sum(col2) / n_docs
            p12 = (co + 1) / (n_docs + 1)
            if p1 == 0 or p2 == 0:
                continue
            pmi = np.log(p12 / (p1 * p2))
            npmi = pmi / (-np.log(p12))
            pair_scores.append(npmi)
        if pair_scores:
            topic_scores.append(np.mean(pair_scores))
        else:
            topic_scores.append(0)
    return np.mean(topic_scores), topic_scores


def run_nmf(texts, k=8):
    """Run TF-IDF + NMF pipeline. Returns vectorizer, tfidf_matrix, nmf model, W, H."""
    vectorizer = TfidfVectorizer(
        max_df=0.80, min_df=3, max_features=5000,
        ngram_range=(1, 2), stop_words="english"
    )
    tfidf = vectorizer.fit_transform(texts)
    nmf = NMF(n_components=k, random_state=42, init="nndsvda", max_iter=500)
    W = nmf.fit_transform(tfidf)  # doc-topic
    H = nmf.components_            # topic-term
    return vectorizer, tfidf, nmf, W, H


def get_top_terms(H, feature_names, n=15):
    """Get top terms per topic."""
    topics = []
    for i, row in enumerate(H):
        top_idx = row.argsort()[::-1][:n]
        terms = [(feature_names[j], row[j]) for j in top_idx]
        topics.append(terms)
    return topics


def main():
    print("=" * 70)
    print("NMF TOPIC MODELLING — 14B Qwen-2.5 Dataset (3,659 paired questions)")
    print("=" * 70)

    # Load data
    paired = load_paired()
    df = pd.DataFrame(paired)
    print(f"\nTotal paired questions: {len(df)}")
    print(f"  StrategyQA: {len(df[df.dataset == 'StrategyQA'])}")
    print(f"  HotpotQA:   {len(df[df.dataset == 'HotpotQA'])}")
    print(f"  LogicQA:    {len(df[df.dataset == 'LogicQA'])}")

    # Run NMF
    print("\n--- Running TF-IDF + NMF (k=8) ---")
    texts = df["text"].tolist()
    vectorizer, tfidf, nmf, W, H = run_nmf(texts, k=8)
    feature_names = vectorizer.get_feature_names_out()
    df["topic"] = W.argmax(axis=1)

    # Top terms per topic
    top_terms = get_top_terms(H, feature_names, n=15)
    print("\n--- TOP TERMS PER TOPIC ---")
    for i, terms in enumerate(top_terms):
        term_str = ", ".join(f"{t[0]}({t[1]:.3f})" for t in terms[:10])
        print(f"  Topic {i}: {term_str}")

    # Per-topic accuracy
    print("\n--- PER-TOPIC ACCURACY ---")
    print(f"{'Topic':<8} {'Cov%':<7} {'A*%':<7} {'CoT%':<7} {'Gap':<7} {'W':<5} | {'SQ_A*':<6} {'SQ_C':<6} {'HP_A*':<6} {'HP_C':<6} {'LQ_A*':<6} {'LQ_C':<6}")
    print("-" * 100)

    topic_stats = []
    for t in range(8):
        mask = df["topic"] == t
        subset = df[mask]
        n = len(subset)
        cov = 100.0 * n / len(df)
        a_acc = 100.0 * subset["astar_correct"].sum() / n if n > 0 else 0
        c_acc = 100.0 * subset["cot_correct"].sum() / n if n > 0 else 0
        gap = a_acc - c_acc
        winner = "A*" if gap > 0 else ("CoT" if gap < 0 else "Tie")

        # Per-dataset
        ds_stats = {}
        for ds in ["StrategyQA", "HotpotQA", "LogicQA"]:
            ds_mask = mask & (df["dataset"] == ds)
            ds_sub = df[ds_mask]
            if len(ds_sub) > 0:
                ds_stats[ds] = (
                    100.0 * ds_sub["astar_correct"].sum() / len(ds_sub),
                    100.0 * ds_sub["cot_correct"].sum() / len(ds_sub),
                )
            else:
                ds_stats[ds] = (None, None)

        sq = ds_stats["StrategyQA"]
        hp = ds_stats["HotpotQA"]
        lq = ds_stats["LogicQA"]

        def fmt(v):
            return f"{v:.0f}" if v is not None else "---"

        print(f"  T{t:<5} {cov:<7.1f} {a_acc:<7.1f} {c_acc:<7.1f} {gap:+.1f}{'':3} {winner:<5} | "
              f"{fmt(sq[0]):<6} {fmt(sq[1]):<6} {fmt(hp[0]):<6} {fmt(hp[1]):<6} {fmt(lq[0]):<6} {fmt(lq[1]):<6}")

        topic_stats.append({
            "topic": t, "coverage": cov, "astar_acc": a_acc, "cot_acc": c_acc,
            "gap": gap, "winner": winner, "n": n,
            "sq_astar": sq[0], "sq_cot": sq[1],
            "hp_astar": hp[0], "hp_cot": hp[1],
            "lq_astar": lq[0], "lq_cot": lq[1],
        })

    # Dominance summary
    astar_topics = [s for s in topic_stats if s["winner"] == "A*"]
    cot_topics = [s for s in topic_stats if s["winner"] == "CoT"]
    tie_topics = [s for s in topic_stats if s["winner"] == "Tie"]
    astar_cov = sum(s["coverage"] for s in astar_topics)
    cot_cov = sum(s["coverage"] for s in cot_topics)
    tie_cov = sum(s["coverage"] for s in tie_topics)

    print(f"\n--- DOMINANCE SUMMARY ---")
    print(f"  A*-dominant: {len(astar_topics)} topics, {astar_cov:.1f}% coverage")
    print(f"  CoT-dominant: {len(cot_topics)} topics, {cot_cov:.1f}% coverage")
    if tie_topics:
        print(f"  Tie: {len(tie_topics)} topics, {tie_cov:.1f}% coverage")

    # Easy/Medium/Hard categorization
    print("\n--- DIFFICULTY CATEGORIZATION ---")
    df["difficulty"] = "Hard"
    df.loc[df["astar_correct"] & df["cot_correct"], "difficulty"] = "Easy"
    df.loc[df["astar_correct"] & ~df["cot_correct"], "difficulty"] = "Medium-A*"
    df.loc[~df["astar_correct"] & df["cot_correct"], "difficulty"] = "Medium-CoT"

    print(f"\nOverall:")
    for d in ["Easy", "Medium-A*", "Medium-CoT", "Hard"]:
        n = len(df[df["difficulty"] == d])
        print(f"  {d:<12}: {n:>5} ({100*n/len(df):.1f}%)")

    print(f"\nPer-topic difficulty distribution:")
    print(f"{'Topic':<8} {'Easy%':<8} {'Med-A*%':<9} {'Med-CoT%':<10} {'Hard%':<8}")
    print("-" * 50)
    for t in range(8):
        mask = df["topic"] == t
        subset = df[mask]
        n = len(subset)
        if n == 0:
            continue
        easy = 100 * len(subset[subset["difficulty"] == "Easy"]) / n
        med_a = 100 * len(subset[subset["difficulty"] == "Medium-A*"]) / n
        med_c = 100 * len(subset[subset["difficulty"] == "Medium-CoT"]) / n
        hard = 100 * len(subset[subset["difficulty"] == "Hard"]) / n
        print(f"  T{t:<5} {easy:<8.1f} {med_a:<9.1f} {med_c:<10.1f} {hard:<8.1f}")

    # NPMI coherence
    print("\n--- NPMI COHERENCE ---")
    top_term_lists = [[t[0] for t in terms[:10]] for terms in top_terms]
    mean_npmi, per_topic_npmi = compute_npmi(tfidf, feature_names, top_term_lists, n_top=10)
    print(f"  Mean NPMI (k=8): {mean_npmi:.4f}")
    for i, s in enumerate(per_topic_npmi):
        print(f"    Topic {i}: {s:.4f}")

    # k-sweep
    print("\n--- COHERENCE K-SWEEP ---")
    for k in [6, 7, 8, 9, 10]:
        _, _, _, W_k, H_k = run_nmf(texts, k=k)
        tt = get_top_terms(H_k, feature_names, n=10)
        ttl = [[t[0] for t in terms] for terms in tt]
        mn, _ = compute_npmi(tfidf, feature_names, ttl, n_top=10)
        print(f"  k={k}: NPMI={mn:.4f}")

    # Example questions per topic
    print("\n--- EXAMPLE QUESTIONS PER TOPIC ---")
    for t in range(8):
        mask = df["topic"] == t
        subset = df[mask]
        print(f"\n  Topic {t} (n={len(subset)}, cov={100*len(subset)/len(df):.1f}%):")

        # A* wins case
        astar_wins = subset[subset["astar_correct"] & ~subset["cot_correct"]]
        if len(astar_wins) > 0:
            ex = astar_wins.iloc[0]
            print(f"    [A* wins] ({ex['dataset']}) {ex['text'][:120]}")
            print(f"              Gold: {ex['gold']} | A*: {ex['astar_pred']} | CoT: {ex['cot_pred']}")

        # CoT wins case
        cot_wins = subset[~subset["astar_correct"] & subset["cot_correct"]]
        if len(cot_wins) > 0:
            ex = cot_wins.iloc[0]
            print(f"    [CoT wins] ({ex['dataset']}) {ex['text'][:120]}")
            print(f"               Gold: {ex['gold']} | A*: {ex['astar_pred']} | CoT: {ex['cot_pred']}")

        # Both correct (Easy)
        both_correct = subset[subset["astar_correct"] & subset["cot_correct"]]
        if len(both_correct) > 0:
            ex = both_correct.iloc[0]
            print(f"    [Both OK]  ({ex['dataset']}) {ex['text'][:120]}")

        # Both wrong (Hard)
        both_wrong = subset[~subset["astar_correct"] & ~subset["cot_correct"]]
        if len(both_wrong) > 0:
            ex = both_wrong.iloc[0]
            print(f"    [Both wrong] ({ex['dataset']}) {ex['text'][:120]}")

    # Save results to JSON
    output = {
        "total_questions": len(df),
        "datasets": {
            "StrategyQA": len(df[df.dataset == "StrategyQA"]),
            "HotpotQA": len(df[df.dataset == "HotpotQA"]),
            "LogicQA": len(df[df.dataset == "LogicQA"]),
        },
        "topic_stats": topic_stats,
        "dominance": {
            "astar_topics": len(astar_topics),
            "astar_coverage": astar_cov,
            "cot_topics": len(cot_topics),
            "cot_coverage": cot_cov,
        },
        "top_terms": {f"topic_{i}": [t[0] for t in terms[:10]] for i, terms in enumerate(top_terms)},
        "npmi_k8": mean_npmi,
        "difficulty_overall": {
            d: int(len(df[df["difficulty"] == d])) for d in ["Easy", "Medium-A*", "Medium-CoT", "Hard"]
        },
    }
    out_path = os.path.join(BASE, "topic_modelling_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
