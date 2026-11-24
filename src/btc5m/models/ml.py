"""Logistic regression, LightGBM, XGBoost and ensembles, each with optional post-hoc
calibration fitted on the (chronologically later) validation block only."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from btc5m.models.base import Model, clip, logit, sigmoid


class Calibrator:
    """Platt (logistic on logit p) or isotonic calibration."""

    def __init__(self, method: str = "platt"):
        self.method = method
        self.m = None

    def fit(self, p, y):
        if self.method == "isotonic":
            self.m = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(p, y)
        else:
            self.m = LogisticRegression(C=1e3, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
        return self

    def __call__(self, p):
        if self.m is None:
            return clip(p)
        if self.method == "isotonic":
            return clip(self.m.predict(p))
        return clip(self.m.predict_proba(logit(p).reshape(-1, 1))[:, 1])


class LogReg(Model):
    """Per-checkpoint L2 logistic regression; C chosen on the validation block."""

    name = "logreg"

    def __init__(self, features: list[str], Cs=(0.01, 0.1, 1.0)):
        self.features = list(features)
        self.Cs = Cs
        self.models: dict = {}
        self.chosen: dict = {}

    def _pipe(self, C):
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=C, max_iter=2000))

    def fit(self, train, val=None):
        for k, g in train.groupby("k"):
            X, y = g[self.features].clip(-50, 50), g["y"].to_numpy()
            best, best_ll = None, np.inf
            if val is not None and len(val):
                v = val[val["k"] == k]
                for C in self.Cs:
                    m = self._pipe(C).fit(X, y)
                    p = clip(m.predict_proba(v[self.features].clip(-50, 50))[:, 1])
                    ll = -np.mean(v["y"] * np.log(p) + (1 - v["y"]) * np.log(1 - p))
                    if ll < best_ll:
                        best, best_ll = C, ll
                # refit on train+val with the chosen C
                gv = pd.concat([g, v])
                self.models[float(k)] = self._pipe(best).fit(gv[self.features].clip(-50, 50), gv["y"].to_numpy())
            else:
                best = self.Cs[-1]
                self.models[float(k)] = self._pipe(best).fit(X, y)
            self.chosen[float(k)] = best
        return self

    def predict(self, df):
        out = np.full(len(df), 0.5)
        ks = df["k"].to_numpy()
        for k, m in self.models.items():
            mask = ks == k
            if mask.any():
                out[mask] = m.predict_proba(df.loc[mask, self.features].clip(-50, 50))[:, 1]
        return clip(out)

    def coefficients(self) -> dict:
        res = {}
        for k, m in self.models.items():
            lr = m[-1]
            res[k] = dict(zip(self.features, lr.coef_[0].tolist()))
        return res

    def params(self):
        return {"C": self.chosen}


class LGBM(Model):
    """Pooled LightGBM across checkpoints (``k`` is a feature); early stopping on validation;
    Platt calibration on the validation block."""

    name = "lightgbm"

    def __init__(self, features: list[str], params: dict | None = None, calibrate: str | None = "platt"):
        self.features = list(features)
        self.p = {
            "objective": "binary", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 200,
            "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0,
            "verbose": -1, "seed": 11, "num_threads": 4,
        }
        self.p.update(params or {})
        self.calibrate = calibrate
        self.cal = None

    def fit(self, train, val=None):
        dtr = lgb.Dataset(train[self.features], train["y"])
        if val is not None and len(val):
            dv = lgb.Dataset(val[self.features], val["y"], reference=dtr)
            self.m = lgb.train(self.p, dtr, num_boost_round=2000, valid_sets=[dv], callbacks=[lgb.early_stopping(100, verbose=False)])
            if self.calibrate:
                self.cal = Calibrator(self.calibrate).fit(self.m.predict(val[self.features], num_iteration=self.m.best_iteration), val["y"].to_numpy())
        else:
            self.m = lgb.train(self.p, dtr, num_boost_round=300)
        return self

    def predict(self, df):
        p = self.m.predict(df[self.features], num_iteration=getattr(self.m, "best_iteration", None) or None)
        return self.cal(p) if self.cal else clip(p)

    def importance(self) -> dict:
        gain = self.m.feature_importance("gain")
        return dict(sorted(zip(self.features, (gain / gain.sum()).tolist()), key=lambda x: -x[1]))

    def params(self):
        return {"best_iteration": int(getattr(self.m, "best_iteration", 0) or 0)}


class XGB(Model):
    name = "xgboost"

    def __init__(self, features: list[str], params: dict | None = None, calibrate: str | None = "platt"):
        self.features = list(features)
        self.p = {
            "objective": "binary:logistic", "eval_metric": "logloss", "eta": 0.03, "max_depth": 4,
            "min_child_weight": 50, "subsample": 0.8, "colsample_bytree": 0.7, "lambda": 10.0, "nthread": 4, "seed": 13,
        }
        self.p.update(params or {})
        self.calibrate = calibrate
        self.cal = None

    def fit(self, train, val=None):
        dtr = xgb.DMatrix(train[self.features], label=train["y"])
        if val is not None and len(val):
            dv = xgb.DMatrix(val[self.features], label=val["y"])
            self.m = xgb.train(self.p, dtr, num_boost_round=2000, evals=[(dv, "val")], early_stopping_rounds=100, verbose_eval=False)
            if self.calibrate:
                pv = self.m.predict(dv, iteration_range=(0, self.m.best_iteration + 1))
                self.cal = Calibrator(self.calibrate).fit(pv, val["y"].to_numpy())
        else:
            self.m = xgb.train(self.p, dtr, num_boost_round=300)
        return self

    def predict(self, df):
        d = xgb.DMatrix(df[self.features])
        it = getattr(self.m, "best_iteration", None)
        p = self.m.predict(d, iteration_range=(0, it + 1)) if it is not None else self.m.predict(d)
        return self.cal(p) if self.cal else clip(p)

    def params(self):
        return {"best_iteration": int(getattr(self.m, "best_iteration", 0) or 0)}


class Ensemble(Model):
    """Logit-average of member models; weights fitted on validation by logistic stacking
    (non-negative weights are not enforced; reported for transparency)."""

    name = "ensemble"

    def __init__(self, members: list[Model], stack: bool = True):
        self.members = members
        self.stack = stack
        self.w = None

    def fit(self, train, val=None):
        for m in self.members:
            m.fit(train, val)
        if self.stack and val is not None and len(val):
            X = np.column_stack([logit(m.predict(val)) for m in self.members])
            lr = LogisticRegression(C=1.0, max_iter=1000).fit(X, val["y"].to_numpy())
            self.w = (lr.coef_[0], lr.intercept_[0])
        return self

    def predict(self, df):
        X = np.column_stack([logit(m.predict(df)) for m in self.members])
        if self.w is None:
            return clip(sigmoid(X.mean(axis=1)))
        return clip(sigmoid(X @ self.w[0] + self.w[1]))

    def params(self):
        return {"members": [m.name for m in self.members], "weights": None if self.w is None else [float(x) for x in self.w[0]] + [float(self.w[1])]}
