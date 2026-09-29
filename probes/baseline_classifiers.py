#!/usr/bin/env python3
"""Why not just train a classifier? For most of our benchmark predicates, you should.

The first question anyone asks a tool like this is what it buys over a cheap model,
or over SQL. The honest answer depends entirely on the predicate, and this measures
the split rather than asserting it.

Every predicate used to benchmark this project so far — `ArrDelay > DepDelay`,
`OriginState == DestState`, `TaxiOut + TaxiIn > 30` — is a **deterministic function
of the columns we send**. That is exactly what made them good for evaluation: ground
truth comes from the table, with nothing to annotate and nothing to argue about. It
also means a decision tree on those columns should reach ~100% for free, and if it
does, a decision model is the wrong tool for them. A `WHERE` clause is.

So this compares, on identical labelled data:

  majority        the floor
  logistic        logistic regression on the projected columns
  tree            gradient-boosted trees on the same
  jev             our measured numbers

and then does the same on predicates that are **not** computable from what the model
is shown: the title of a film is given, and the question is about the film. Ground
truth comes from a genre flag that is withheld from the input. A text classifier can
learn title-surface correlations; it cannot know what the film is about. That is
where a decision model should earn its place, and if it does not, that is worth
knowing too.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jev_solo import datasets  # noqa: E402
from jev_solo.recalibrate import balanced_accuracy  # noqa: E402


def encode_features(rows, cols_are_numeric):
    """Numeric columns as floats, categorical as label codes."""
    out = []
    for j, numeric in enumerate(cols_are_numeric):
        col = [r[j] for r in rows]
        if numeric:
            vals = []
            for v in col:
                try:
                    vals.append(float(v))
                except ValueError:
                    vals.append(np.nan)
            out.append(np.asarray(vals, dtype=float))
        else:
            uniq = {v: i for i, v in enumerate(sorted(set(col)))}
            out.append(np.asarray([uniq[v] for v in col], dtype=float))
    X = np.vstack(out).T
    return np.nan_to_num(X, nan=-1.0)


def run_classifiers(X, y, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=seed,
                                          stratify=y if len(set(y)) > 1 else None)
    out = {}
    maj = float(np.bincount(ytr.astype(int)).argmax())
    out["majority"] = balanced_accuracy(np.full(len(yte), maj), yte)
    try:
        sc = StandardScaler().fit(Xtr)
        lr = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr)
        out["logistic"] = balanced_accuracy(lr.predict(sc.transform(Xte)), yte)
    except Exception as exc:
        out["logistic"] = float("nan")
        print(f"      logistic failed: {exc}")
    try:
        gb = HistGradientBoostingClassifier(random_state=seed).fit(Xtr, ytr)
        out["tree"] = balanced_accuracy(gb.predict(Xte), yte)
    except Exception as exc:
        out["tree"] = float("nan")
        print(f"      tree failed: {exc}")
    return out


def run_text_classifier(texts, y, seed=0):
    """TF-IDF over the same text the model is shown."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    ttr, tte, ytr, yte = train_test_split(texts, y, test_size=0.5, random_state=seed,
                                          stratify=y if len(set(y)) > 1 else None)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2)
    Xtr = vec.fit_transform(ttr)
    lr = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    return balanced_accuracy(lr.predict(vec.transform(tte)), yte)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4000)
    ap.add_argument("--out", default="results/baselines.json")
    args = ap.parse_args()
    records = []

    print("## computable predicates: is a classifier on the same columns enough?")
    print(f"   {'table':8s} {'predicate':12s} {'n':>5s} {'base':>6s} "
          f"{'majority':>9s} {'logistic':>9s} {'tree':>7s}")
    for tname in ("flight", "movies"):
        table = datasets.get(tname)
        header, rows_all = table.load(40_000)
        for pname, pred in table.predicates.items():
            idx = [header.index(c) for c in pred.cols]
            X_rows, y = [], []
            for r in rows_all:
                vals = [r[i] for i in idx]
                if any(v in ("", "NA", "NULL") for v in vals):
                    continue
                try:
                    y.append(int(bool(pred.truth(vals))))
                except Exception:
                    continue
                X_rows.append(vals)
                if len(y) >= args.rows:
                    break
            y = np.asarray(y)
            if len(set(y.tolist())) < 2:
                continue
            numeric = []
            for j in range(len(idx)):
                try:
                    float(X_rows[0][j])
                    numeric.append(True)
                except ValueError:
                    numeric.append(False)
            res = run_classifiers(encode_features(X_rows, numeric), y)
            print(f"   {tname:8s} {pname:12s} {len(y):5d} {y.mean():6.3f} "
                  f"{res['majority']*100:8.1f}% {res['logistic']*100:8.1f}% "
                  f"{res['tree']*100:6.1f}%")
            records.append({"kind": "computable", "table": tname, "predicate": pname,
                            "n": int(len(y)), "base_rate": float(y.mean()), **res})

    print("\n## semantic predicates: the model sees only the title; truth is a")
    print("   withheld genre flag, so the columns shown do not determine the answer")
    table = datasets.get("movies")
    header, rows_all = table.load(40_000)
    ti = header.index("title")
    print(f"   {'predicate':16s} {'n':>5s} {'base':>6s} {'majority':>9s} "
          f"{'tfidf on title':>15s}")
    for flag in ("Action", "Comedy", "Drama", "Horror"):
        fi = header.index(flag)
        texts, y = [], []
        for r in rows_all:
            if not r[ti] or r[fi] in ("", "NA"):
                continue
            texts.append(r[ti])
            y.append(int(r[fi] in ("1", "1.0", "True")))
            if len(y) >= args.rows:
                break
        y = np.asarray(y)
        if len(set(y.tolist())) < 2:
            continue
        maj = balanced_accuracy(
            np.full(len(y) // 2, float(np.bincount(y.astype(int)).argmax())),
            y[len(y) // 2:])
        tf = run_text_classifier(texts, y)
        print(f"   is_{flag:13s} {len(y):5d} {y.mean():6.3f} {maj*100:8.1f}% "
              f"{tf*100:14.1f}%")
        records.append({"kind": "semantic", "table": "movies",
                        "predicate": f"is_{flag}", "n": int(len(y)),
                        "base_rate": float(y.mean()), "majority": maj, "tfidf": tf})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"records": records}, indent=2))
    print(f"\n# wrote {args.out}")


if __name__ == "__main__":
    main()
