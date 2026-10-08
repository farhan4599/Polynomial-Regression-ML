import os
import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso, Ridge, LogisticRegression
from sklearn.model_selection import KFold, cross_val_predict

ROLL = "BT2024140"
DATA = "data"
OUT = "predictions"
RES = "results"
SEED = 1

os.makedirs(OUT, exist_ok=True)
os.makedirs(RES, exist_ok=True)


def load_data(v):
    train = pd.read_csv(f"{DATA}/{ROLL}_train_{v}.csv")
    test = pd.read_csv(f"{DATA}/{ROLL}_test_{v}.csv")

    X = train.drop(columns="y").values
    y = train["y"].values
    Xt = test.values

    return X, y, Xt


def get_r2(y, pred, weights=None):
    if weights is None:
        weights = np.ones(len(y))

    mean = np.average(y, weights=weights)

    num = np.sum(weights * (y - pred) ** 2)
    den = np.sum(weights * (y - mean) ** 2)

    return 1 - num / den


def get_weights(X_train, X_test):
    def make_features(X):
        clipped = (np.abs(X) == 1).astype(float)

        return np.hstack([
            X,
            X ** 2,
            clipped,
            clipped.sum(axis=1, keepdims=True),
            (X ** 2).sum(axis=1, keepdims=True)
        ])

    X = np.vstack([make_features(X_train), make_features(X_test)])
    labels = np.r_[
        np.zeros(len(X_train)),
        np.ones(len(X_test))
    ]

    model = LogisticRegression(max_iter=2000)

    prob = cross_val_predict(
        model,
        X,
        labels,
        cv=5,
        method="predict_proba"
    )[:, 1]

    prob = prob[:len(X_train)]

    weights = prob / (1 - prob)
    weights *= len(X_train) / len(X_test)

    weights = np.minimum(weights, 20)
    weights /= weights.mean()

    return weights


def get_cv_score(model_func, X, y, weights, folds):
    pred = np.zeros(len(y))

    for train_idx, val_idx in folds:
        model = model_func()
        model.fit(X[train_idx], y[train_idx])
        pred[val_idx] = model.predict(X[val_idx])

    normal_r2 = get_r2(y, pred)
    weighted_r2 = get_r2(y, pred, weights)

    return normal_r2, weighted_r2


def run_model(v, model_name, model_func, degrees, params, shift_aware=False):
    X, y, Xt = load_data(v)

    if shift_aware:
        weights = get_weights(X, Xt)
    else:
        weights = np.ones(len(y))

    folds = [
        list(
            KFold(
                n_splits=5,
                shuffle=True,
                random_state=SEED + i
            ).split(X)
        )
        for i in range(3)
    ]

    results = []

    for degree in degrees:
        poly = PolynomialFeatures(
            degree=degree,
            include_bias=False
        )

        P = poly.fit_transform(X)

        for param in params:
            scores = []

            for fold in folds:
                scores.append(
                    get_cv_score(
                        lambda: model_func(param),
                        P,
                        y,
                        weights,
                        fold
                    )
                )

            scores = np.array(scores)

            results.append({
                "degree": degree,
                "param": param,
                "cv_r2": scores[:, 0].mean(),
                "cv_r2_weighted": scores[:, 1].mean()
            })

            print(
                v,
                degree,
                param,
                round(scores[:, 0].mean(), 4),
                round(scores[:, 1].mean(), 4)
            )

    results = pd.DataFrame(results)

    if shift_aware:
        best_row = results.loc[
            results["cv_r2_weighted"].idxmax()
        ]
    else:
        best_row = results.loc[
            results["cv_r2"].idxmax()
        ]

    best_degree = int(best_row["degree"])
    best_param = float(best_row["param"])

    poly = PolynomialFeatures(
        degree=best_degree,
        include_bias=False
    )

    P = poly.fit_transform(X)
    Pt = poly.transform(Xt)

    model = model_func(best_param)
    model.fit(P, y)

    pred = model.predict(Pt)

    pd.DataFrame({
        "y": pred
    }).to_csv(
        f"{OUT}/{ROLL}_pred_{v}.csv",
        index=False
    )

    results.to_csv(
        f"{RES}/cv_{v}.csv",
        index=False
    )

    summary = {
        "problem": v,
        "model": model_name,
        "degree": best_degree,
        "param": best_param,
        "cv_r2": float(best_row["cv_r2"]),
        "cv_r2_weighted": float(best_row["cv_r2_weighted"]),
        "train_r2": float(
            get_r2(y, model.predict(P))
        )
    }

    with open(
        f"{RES}/summary_{v}.json",
        "w"
    ) as f:
        json.dump(summary, f, indent=2)

    print("chosen:", v, best_degree, best_param)


if __name__ == "__main__":
    run_model(
        "var1",
        "Lasso",
        lambda a: Lasso(
            alpha=a,
            max_iter=100000,
            tol=1e-7
        ),
        [3, 4, 5, 6],
        [0.001, 0.002, 0.003, 0.005, 0.008],
        True
    )

    run_model(
        "var2",
        "Ridge",
        lambda a: Ridge(alpha=a),
        [4, 5, 6, 7, 8, 9, 10, 12, 14],
        [1e-6, 1e-4, 1e-2, 1e-1, 1.0],
        False
    )