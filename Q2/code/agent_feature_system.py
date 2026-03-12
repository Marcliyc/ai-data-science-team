from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import seaborn as sns


BASE_COLS = [
    "trade_date",
    "underlying",
    "start_time",
    "end_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


@dataclass
class AgentConfig:
    data_dir: str = "Q2/data"
    output_dir: str = "Q2/output"
    label_col: str = "Y1"
    top_k: int = 50
    corr_threshold: float = 0.92
    missing_threshold: float = 0.35
    random_state: int = 42


class FinancialFeatureAgentSystem:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.fig_dir = self.output_dir / "figures"
        self.report_dir = self.output_dir / "reports"
        self.log_dir = self.output_dir / "logs"
        for d in [self.fig_dir, self.report_dir, self.log_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _log(self, agent_name: str, action: str, detail: Dict):
        rec = {
            "agent": agent_name,
            "action": action,
            "detail": detail,
            "ts": pd.Timestamp.utcnow().isoformat(),
        }
        with (self.log_dir / "agent_run_log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def load_data(self) -> pd.DataFrame:
        self._log("orchestrator", "load_data", {"data_dir": self.config.data_dir})
        files = sorted(Path(self.config.data_dir).glob("*.pq"))
        if not files:
            raise FileNotFoundError(f"No parquet files found in {self.config.data_dir}")
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.sort_values(["trade_date", "underlying"]).reset_index(drop=True)
        self._log("orchestrator", "data_loaded", {"shape": list(df.shape)})
        return df

    def _get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if c.startswith("X")]

    def diagnose_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_cols = self._get_feature_cols(df)
        rows = []
        for col in feature_cols:
            s = df[col]
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outlier_ratio = ((s < lower) | (s > upper)).mean()
            rows.append(
                {
                    "feature": col,
                    "missing_ratio": float(s.isna().mean()),
                    "mean": float(s.mean()),
                    "std": float(s.std()),
                    "skew": float(s.skew()),
                    "kurtosis": float(s.kurtosis()),
                    "outlier_ratio_iqr": float(outlier_ratio),
                }
            )
        report = pd.DataFrame(rows).sort_values("feature")
        report.to_csv(self.report_dir / "feature_diagnosis_report.csv", index=False)
        self._log("diagnosis_agent", "diagnosis_complete", {"n_features": len(feature_cols)})

        plt.figure(figsize=(12, 4))
        sns.histplot(report["missing_ratio"], bins=30)
        plt.title("Feature missing ratio distribution")
        plt.tight_layout()
        plt.savefig(self.fig_dir / "missing_ratio_distribution.png", dpi=180)
        plt.close()

        return report

    def clean_features(self, df: pd.DataFrame, diagnosis: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        feature_cols = self._get_feature_cols(df)
        clean_df = df.copy()
        actions = []
        kept_features = []

        for _, row in diagnosis.iterrows():
            feature = row["feature"]
            missing_ratio = row["missing_ratio"]
            outlier_ratio = row["outlier_ratio_iqr"]
            action = []

            if missing_ratio > self.config.missing_threshold:
                action.append("drop_high_missing")
                actions.append({"feature": feature, "action": "+".join(action), "kept": False})
                continue

            kept_features.append(feature)

            if missing_ratio > 0:
                med = clean_df[feature].median()
                clean_df[feature] = clean_df[feature].fillna(med)
                action.append("median_impute")

            if outlier_ratio > 0.08:
                lo, hi = clean_df[feature].quantile([0.01, 0.99])
                clean_df[feature] = clean_df[feature].clip(lo, hi)
                action.append("winsorize_1_99")

            if not action:
                action.append("keep")

            actions.append({"feature": feature, "action": "+".join(action), "kept": True})

        actions_df = pd.DataFrame(actions)
        actions_df.to_csv(self.report_dir / "feature_cleaning_actions.csv", index=False)
        self._log("cleaning_agent", "cleaning_complete", {"kept": len(kept_features), "dropped": len(feature_cols) - len(kept_features)})

        selected_cols = BASE_COLS + [self.config.label_col] + kept_features
        clean_df = clean_df[selected_cols]
        return clean_df, actions_df

    def _prepare_binary_target(self, s: pd.Series) -> pd.Series:
        # Positive class: 1, Others(-1/0) as 0
        return (s == 1).astype(int)

    def evaluate_and_select_features(self, df: pd.DataFrame) -> Tuple[List[str], pd.DataFrame, Dict]:
        all_features = [c for c in df.columns if c.startswith("X")]
        target = self._prepare_binary_target(df[self.config.label_col])

        split_idx = int(len(df) * 0.8)
        train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
        y_train = target.iloc[:split_idx]
        y_test = target.iloc[split_idx:]

        X_train = train_df[all_features]
        X_test = test_df[all_features]

        mi = mutual_info_classif(X_train, y_train, random_state=self.config.random_state)
        mi_df = pd.DataFrame({"feature": all_features, "mi": mi}).sort_values("mi", ascending=False)

        rf = RandomForestClassifier(
            n_estimators=180,
            max_depth=8,
            random_state=self.config.random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        rf.fit(X_train, y_train)
        rf_df = pd.DataFrame({"feature": all_features, "rf_importance": rf.feature_importances_})

        score_df = mi_df.merge(rf_df, on="feature")
        score_df["mi_rank"] = score_df["mi"].rank(ascending=False)
        score_df["rf_rank"] = score_df["rf_importance"].rank(ascending=False)
        score_df["combined_rank"] = (score_df["mi_rank"] + score_df["rf_rank"]) / 2
        score_df = score_df.sort_values("combined_rank")

        initial_candidates = score_df["feature"].head(120).tolist()

        corr = X_train[initial_candidates].corr().abs()
        chosen = []
        for feat in initial_candidates:
            if not chosen:
                chosen.append(feat)
                continue
            if corr.loc[feat, chosen].max() < self.config.corr_threshold:
                chosen.append(feat)
            if len(chosen) >= self.config.top_k:
                break

        top_df = score_df[score_df["feature"].isin(chosen)].copy().sort_values("combined_rank")
        top_df.to_csv(self.report_dir / "top50_features_with_scores.csv", index=False)

        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=600, class_weight="balanced", random_state=self.config.random_state)),
            ]
        )
        model.fit(X_train[chosen], y_train)
        prob = model.predict_proba(X_test[chosen])[:, 1]
        pred = (prob >= 0.5).astype(int)

        metrics = {
            "auc": float(roc_auc_score(y_test, prob)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
        }

        fpr, tpr, _ = roc_curve(y_test, prob)
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, label=f"ROC AUC={metrics['auc']:.4f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.fig_dir / "roc_curve.png", dpi=180)
        plt.close()

        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax)
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(self.fig_dir / "confusion_matrix.png", dpi=180)
        plt.close()

        top20 = top_df.head(20)
        plt.figure(figsize=(8, 6))
        sns.barplot(data=top20, y="feature", x="rf_importance")
        plt.title("Top 20 features by RF importance (within selected 50)")
        plt.tight_layout()
        plt.savefig(self.fig_dir / "top20_feature_importance.png", dpi=180)
        plt.close()

        leak_checks = {
            "split_mode": "time_based_80_20",
            "train_max_date": str(train_df["trade_date"].max().date()),
            "test_min_date": str(test_df["trade_date"].min().date()),
            "temporal_order_ok": bool(train_df["trade_date"].max() <= test_df["trade_date"].min()),
            "max_feature_target_corr_in_train": float(
                pd.concat([X_train[chosen], y_train.rename("target")], axis=1)
                .corr()["target"]
                .drop("target")
                .abs()
                .max()
            ),
        }

        with (self.report_dir / "model_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        with (self.report_dir / "leakage_checks.json").open("w", encoding="utf-8") as f:
            json.dump(leak_checks, f, ensure_ascii=False, indent=2)

        self._log("selection_agent", "selection_complete", {"top_k": len(chosen), "metrics": metrics})
        return chosen, top_df, {"metrics": metrics, "leakage": leak_checks}

    def run(self) -> Dict:
        df = self.load_data()
        diagnosis = self.diagnose_features(df)
        clean_df, actions = self.clean_features(df, diagnosis)
        top_features, top_df, results = self.evaluate_and_select_features(clean_df)

        summary = {
            "label": self.config.label_col,
            "n_rows": int(len(df)),
            "n_features_raw": int(len([c for c in df.columns if c.startswith('X')])),
            "n_features_after_cleaning": int(actions["kept"].sum()),
            "top50_features": top_features,
            **results,
        }
        with (self.report_dir / "run_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary


def run_agent_system() -> Dict:
    system = FinancialFeatureAgentSystem(AgentConfig())
    return system.run()
