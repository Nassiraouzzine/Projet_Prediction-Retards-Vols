"""
PIPELINE FINAL — Prédiction des Retards de Vols
=================================================
Relie les 3 binômes en un seul script :
  Binôme 1 → Preprocessing
  Binôme 2 → Machine Learning (LR, RF, SVM, XGBoost)
  Binôme 3 → Deep Learning (MLP, LSTM, GRU, Transformer)
             → Évalue les 4 modèles et choisit le meilleur par ROC-AUC

Usage
-----
    # Depuis le début (preprocessing inclus) :
    python pipeline_automatique_final1.py --flights flights.csv --airlines airlines.csv --airports airports.csv

    # CSV nettoyé existe déjà :
    python pipeline_automatique_final1.py --skip_preprocessing

    # Dossier DL personnalisé :
    python pipeline_automatique_final1.py --skip_preprocessing --dl_folder mon_dossier_dl
"""

from __future__ import annotations

import argparse
import logging
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    ConfusionMatrixDisplay, classification_report,
)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SEED = 42

# ══════════════════════════════════════════════════════════════════════════════
# BINÔME 1 — PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    flights_path:  str = "flights.csv"
    airlines_path: str = "airlines.csv"
    airports_path: str = "airports.csv"
    output_path:   str = "flights_cleaned_encoded.csv"
    drop_cancelled: bool = True
    drop_diverted:  bool = True
    leakage_cols: List[str] = field(default_factory=lambda: [
        "CANCELLATION_REASON", "AIR_SYSTEM_DELAY", "SECURITY_DELAY",
        "AIRLINE_DELAY", "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY",
        "WHEELS_OFF", "WHEELS_ON", "TAXI_IN", "TAXI_OUT",
        "DEPARTURE_TIME", "ARRIVAL_TIME", "ELAPSED_TIME", "AIR_TIME",
    ])
    outlier_cols: List[str] = field(default_factory=lambda: [
        "DEPARTURE_DELAY", "ARRIVAL_DELAY", "DISTANCE", "SCHEDULED_TIME",
    ])
    iqr_multiplier: float = 1.5
    delay_bins: List[int] = field(default_factory=lambda: [15, 30])
    merge_airlines: bool = True
    merge_airports: bool = True
    post_merge_drop: List[str] = field(default_factory=lambda: [
        "AIRLINE_NAME", "AIRPORT", "CITY", "STATE", "COUNTRY", "TAIL_NUMBER",
    ])
    ohe_cols: List[str] = field(default_factory=lambda: [
        "AIRLINE", "DAY_PERIOD", "SEASON", "WEATHER_RISK",
    ])
    freq_encode_cols: List[str] = field(default_factory=lambda: [
        "ORIGIN_AIRPORT", "DESTINATION_AIRPORT",
    ])


class FlightPreprocessingPipeline:

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.flights = self.airlines = self.airports = self.df = None

    def load(self):
        log.info("── [B1] Étape 1 · Chargement ────────────────────────────────")
        self.flights  = pd.read_csv(self.cfg.flights_path)
        self.airlines = pd.read_csv(self.cfg.airlines_path)
        self.airports = pd.read_csv(self.cfg.airports_path)
        log.info("Flights %s | Airlines %s | Airports %s",
                 self.flights.shape, self.airlines.shape, self.airports.shape)
        return self

    def audit(self):
        log.info("── [B1] Étape 2 · Audit ─────────────────────────────────────")
        df = self.flights
        missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        missing_pct = missing_pct[missing_pct > 0]
        if missing_pct.empty:
            log.info("Aucune valeur manquante.")
        else:
            log.info("Valeurs manquantes (%%):\n%s", missing_pct.to_string())
        log.info("Doublons : %d", df.duplicated().sum())
        return self

    def clean(self):
        log.info("── [B1] Étape 3 · Nettoyage ─────────────────────────────────")
        df = self.flights.copy()
        n0 = len(df)
        if self.cfg.drop_cancelled:
            df = df[df["CANCELLED"] == 0]
        if self.cfg.drop_diverted:
            df = df[df["DIVERTED"] == 0]
        log.info("Après suppression annulés/déviés : %d (supprimé %d)", len(df), n0 - len(df))
        existing_leakage = [c for c in self.cfg.leakage_cols if c in df.columns]
        df = df.drop(columns=existing_leakage)
        df = df.dropna(subset=["ARRIVAL_DELAY"])
        for col in ["DIVERTED", "CANCELLED"]:
            if col in df.columns:
                df = df.drop(columns=[col])
        self.df = df
        log.info("Shape après nettoyage : %s", self.df.shape)
        return self

    def treat_outliers(self):
        log.info("── [B1] Étape 4 · Outliers ──────────────────────────────────")
        df = self.df
        k  = self.cfg.iqr_multiplier
        for col in self.cfg.outlier_cols:
            if col not in df.columns:
                continue
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR    = Q3 - Q1
            lo, hi = Q1 - k * IQR, Q3 + k * IQR
            df[col] = np.clip(df[col], lo, hi)
            log.info("  %-22s capped to [%.1f, %.1f]", col, lo, hi)
        self.df = df
        return self

    def engineer_target(self):
        log.info("── [B1] Étape 5 · Création DELAY_CLASS ──────────────────────")
        low, mid = self.cfg.delay_bins
        def categorize(x):
            if x <= low:   return 0
            elif x <= mid: return 1
            return 2
        self.df["DELAY_CLASS"] = self.df["ARRIVAL_DELAY"].apply(categorize)
        log.info("Distribution:\n%s",
                 self.df["DELAY_CLASS"].value_counts().sort_index().to_string())
        return self

    def engineer_features(self):
        log.info("── [B1] Étape 6 · Feature Engineering ──────────────────────")
        df = self.df.copy()
        df["DEP_HOUR"] = df["SCHEDULED_DEPARTURE"] // 100
        df["ARR_HOUR"] = df["SCHEDULED_ARRIVAL"]   // 100
        def get_period(h):
            if h < 6:    return "night"
            elif h < 12: return "morning"
            elif h < 18: return "afternoon"
            else:        return "evening"
        def get_season(m):
            if m in [12,1,2]:  return "winter"
            elif m in [3,4,5]: return "spring"
            elif m in [6,7,8]: return "summer"
            else:              return "autumn"
        df["DAY_PERIOD"]      = df["DEP_HOUR"].apply(get_period)
        df["SEASON"]          = df["MONTH"].apply(get_season)
        df["AIRPORT_TRAFFIC"] = df["ORIGIN_AIRPORT"].map(df["ORIGIN_AIRPORT"].value_counts())
        df["WEATHER_RISK"]    = df["SEASON"].map({"winter":"high","summer":"medium"}).fillna("low")
        df["IS_WEEKEND"]      = df["DAY_OF_WEEK"].apply(lambda x: 1 if x in [6,7] else 0)
        self.df = df
        log.info("Shape après feature engineering : %s", self.df.shape)
        return self

    def merge_lookup_tables(self):
        log.info("── [B1] Étape 7 · Merge ─────────────────────────────────────")
        df = self.df
        if self.cfg.merge_airlines and self.airlines is not None:
            df = df.merge(self.airlines, left_on="AIRLINE", right_on="IATA_CODE", how="left")
            df = df.drop(columns=["IATA_CODE"], errors="ignore")
            df = df.rename(columns={"AIRLINE_y":"AIRLINE_NAME","AIRLINE_x":"AIRLINE"})
        if self.cfg.merge_airports and self.airports is not None:
            df = df.merge(self.airports, left_on="ORIGIN_AIRPORT", right_on="IATA_CODE", how="left")
            df = df.drop(columns=["IATA_CODE"], errors="ignore")
        to_drop = [c for c in self.cfg.post_merge_drop if c in df.columns]
        df = df.drop(columns=to_drop)
        self.df = df
        log.info("Shape après merge : %s", self.df.shape)
        return self

    def encode(self):
        log.info("── [B1] Étape 8 · Encodage ──────────────────────────────────")
        df = self.df
        ohe_present = [c for c in self.cfg.ohe_cols if c in df.columns]
        if ohe_present:
            df = pd.get_dummies(df, columns=ohe_present, drop_first=True)
        for col in self.cfg.freq_encode_cols:
            if col not in df.columns:
                continue
            df[f"{col}_FREQ"] = df[col].map(df[col].value_counts())
            df = df.drop(columns=[col])
        if "ARRIVAL_DELAY" in df.columns:
            df = df.drop(columns=["ARRIVAL_DELAY"])
        before = len(df)
        df = df.dropna()
        log.info("Lignes supprimées (dropna) : %d", before - len(df))
        bool_cols = df.select_dtypes(include="bool").columns.tolist()
        if bool_cols:
            df[bool_cols] = df[bool_cols].astype(int)
        self.df = df
        log.info("Shape finale encodée : %s", self.df.shape)
        return self

    def export(self):
        log.info("── [B1] Étape 9 · Export ────────────────────────────────────")
        out = Path(self.cfg.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(out, index=False)
        log.info("✅ Sauvegardé : %s  (%d lignes × %d colonnes)", out, *self.df.shape)
        return self

    def run(self) -> pd.DataFrame:
        (self.load().audit().clean().treat_outliers()
             .engineer_target().engineer_features()
             .merge_lookup_tables().encode().export())
        log.info("✅ [BINÔME 1] Preprocessing terminé !")
        return self.df


# ══════════════════════════════════════════════════════════════════════════════
# BINÔME 2 — MACHINE LEARNING
# ══════════════════════════════════════════════════════════════════════════════

def pipeline_ML(csv_path: str):
    log.info("\n" + "="*60)
    log.info("🤖 [BINÔME 2] Machine Learning")
    log.info("="*60)

    df = pd.read_csv(csv_path)
    y  = df["DELAY_CLASS"]
    X  = df.drop(columns=["DELAY_CLASS"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    log.info("Train : %d | Test : %d", len(X_train), len(X_test))

    modeles = {}

    # Logistic Regression
    log.info("⏳ Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
    lr.fit(X_train_scaled, y_train)
    modeles["Logistic Regression"] = (lr, X_test_scaled)
    log.info("✅ Logistic Regression terminée")

    # Random Forest
    log.info("⏳ Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=SEED,
                                n_jobs=-1, class_weight="balanced")
    rf.fit(X_train, y_train)
    modeles["Random Forest"] = (rf, X_test)
    log.info("✅ Random Forest terminé")

    # SVM
    log.info("⏳ SVM (15 000 lignes)...")
    X_svm, _, y_svm, _ = train_test_split(
        X_train_scaled, y_train, train_size=15000, random_state=SEED, stratify=y_train
    )
    svm = SVC(kernel="rbf", probability=True, random_state=SEED, class_weight="balanced")
    svm.fit(X_svm, y_svm)
    modeles["SVM (RBF)"] = (svm, X_test_scaled)
    log.info("✅ SVM terminé")

    # XGBoost
    log.info("⏳ XGBoost...")
    xgb = XGBClassifier(n_estimators=200, random_state=SEED, n_jobs=-1,
                        eval_metric="mlogloss", verbosity=0)
    xgb.fit(X_train, y_train)
    modeles["XGBoost"] = (xgb, X_test)
    log.info("✅ XGBoost terminé")

    # GridSearchCV
    log.info("⏳ GridSearchCV XGBoost (16 combinaisons × 3 folds)...")
    param_grid = {
        "n_estimators" : [100, 200],
        "max_depth"    : [4, 6],
        "learning_rate": [0.05, 0.1],
        "subsample"    : [0.8, 1.0],
    }
    X_tune, _, y_tune, _ = train_test_split(
        X_train, y_train, train_size=30000, random_state=SEED, stratify=y_train
    )
    grid = GridSearchCV(
        XGBClassifier(random_state=SEED, verbosity=0, eval_metric="mlogloss"),
        param_grid, cv=3, scoring="f1_macro", n_jobs=-1, verbose=0
    )
    grid.fit(X_tune, y_tune)
    best_xgb = XGBClassifier(**grid.best_params_, random_state=SEED,
                              verbosity=0, eval_metric="mlogloss")
    best_xgb.fit(X_train, y_train)
    modeles["XGBoost (tuned)"] = (best_xgb, X_test)
    log.info("✅ XGBoost tuned — meilleurs params : %s", grid.best_params_)

    # Évaluation
    os.makedirs("results", exist_ok=True)
    os.makedirs("models",  exist_ok=True)
    resultats_ml = {}
    y_test_bin   = label_binarize(y_test, classes=[0, 1, 2])

    for nom, (model, X_te) in modeles.items():
        y_pred  = model.predict(X_te)
        y_proba = model.predict_proba(X_te)
        acc     = (y_pred == y_test).mean()
        auc     = roc_auc_score(y_test_bin, y_proba, multi_class="ovr")
        fn      = ((y_pred == 0) & (y_test != 0)).sum()
        fn_rate = fn / (y_test != 0).sum() * 100
        resultats_ml[nom] = {
            "Accuracy"      : round(acc, 4),
            "ROC-AUC"       : round(auc, 4),
            "Faux Négatifs" : fn,
            "Taux FN %"     : round(fn_rate, 1),
        }
        log.info("  %-22s → Accuracy: %.4f | ROC-AUC: %.4f | FN: %.1f%%",
                 nom, acc, auc, fn_rate)

        fig, ax = plt.subplots(figsize=(6, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred,
            display_labels=["A l'heure", "+15min", "+30min"],
            cmap="Blues", ax=ax
        )
        ax.set_title(f"Matrice de Confusion — {nom}")
        plt.tight_layout()
        plt.savefig(f"results/confusion_ML_{nom.replace(' ', '_')}.png", dpi=150)
        plt.close()

    df_ml      = pd.DataFrame(resultats_ml).T
    best_name  = df_ml["ROC-AUC"].idxmax()
    best_model = modeles[best_name][0]

    joblib.dump(best_model, "models/best_model_ML.pkl")
    joblib.dump(scaler,     "models/scaler_ML.pkl")
    df_ml.to_csv("results/resultats_ML.csv")
    log.info("✅ [BINÔME 2] Meilleur modèle ML : %s sauvegardé", best_name)

    return modeles, df_ml, best_name, y_test, X_test, X_test_scaled


# ══════════════════════════════════════════════════════════════════════════════
# BINÔME 3 — DEEP LEARNING
# Charge les 4 modèles depuis dl_models/ et choisit le meilleur par ROC-AUC
# ══════════════════════════════════════════════════════════════════════════════

def pipeline_DL(csv_path: str,
                dl_folder: str = "dl_models",
                nrows:     int = 50000):

    log.info("\n" + "="*60)
    log.info("🧠 [BINÔME 3] Deep Learning — MLP + LSTM + GRU + Transformer")
    log.info("="*60)

    # Chemins des fichiers dans dl_models/
    scaler_path   = os.path.join(dl_folder, "scaler.pkl")
    features_path = os.path.join(dl_folder, "features.pkl")

    # Vérification fichiers partagés obligatoires
    for p in [scaler_path, features_path]:
        if not os.path.exists(p):
            log.warning("⚠️  Fichier manquant : %s", p)
            log.warning("   → Ajoutez la cellule de sauvegarde dans le notebook DL.")
            return None, None, None

    try:
        import torch
        import torch.nn as nn

        # ── Chargement des artefacts partagés ────────────────────────────
        features_list = joblib.load(features_path)
        sc            = joblib.load(scaler_path)
        N_FEAT        = len(features_list)
        SEQ_LEN       = N_FEAT

        # ── Chargement et préparation des données ────────────────────────
        df_new  = pd.read_csv(csv_path, nrows=nrows)
        missing = [f for f in features_list if f not in df_new.columns]
        if missing:
            log.warning("⚠️  Colonnes manquantes : %s", missing)
            return None, None, None

        X_new  = df_new[features_list].values.astype(np.float32)
        X_new  = sc.transform(X_new)
        y_true = df_new["DELAY_CLASS"].values if "DELAY_CLASS" in df_new.columns else None

        # ── Définition des architectures (identiques au notebook) ────────
        class MLP(nn.Module):
            def __init__(self, in_dim, hidden1=256, hidden2=128, hidden3=64,
                         dropout=0.3, n_classes=3):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hidden1), nn.BatchNorm1d(hidden1),
                    nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(hidden1, hidden2), nn.BatchNorm1d(hidden2),
                    nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(hidden2, hidden3), nn.BatchNorm1d(hidden3),
                    nn.ReLU(), nn.Dropout(dropout / 2),
                    nn.Linear(hidden3, n_classes),
                )
            def forward(self, x): return self.net(x)

        class RNNModel(nn.Module):
            def __init__(self, input_size=1, hidden_size=64, num_layers=2,
                         dropout=0.3, n_classes=3, rnn_type="LSTM"):
                super().__init__()
                RNN = nn.LSTM if rnn_type == "LSTM" else nn.GRU
                self.rnn = RNN(input_size, hidden_size, num_layers=num_layers,
                               batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Sequential(
                    nn.Linear(hidden_size, 32), nn.ReLU(),
                    nn.Dropout(dropout / 2), nn.Linear(32, n_classes)
                )
            def forward(self, x):
                out, _ = self.rnn(x)
                return self.fc(self.dropout(out[:, -1, :]))

        class TransformerClassifier(nn.Module):
            def __init__(self, seq_len, d_model=32, nhead=4,
                         num_layers=2, dropout=0.2, n_classes=3):
                super().__init__()
                self.input_proj = nn.Linear(1, d_model)
                self.pos_emb    = nn.Embedding(seq_len, d_model)
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead,
                    dim_feedforward=d_model * 4,
                    dropout=dropout, batch_first=True)
                self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
                self.norm    = nn.LayerNorm(d_model)
                self.dropout = nn.Dropout(dropout)
                self.fc      = nn.Linear(d_model, n_classes)
            def forward(self, x):
                pos = torch.arange(x.size(1))
                x   = self.input_proj(x) + self.pos_emb(pos)
                x   = self.norm(self.encoder(x))
                return self.fc(self.dropout(x.mean(dim=1)))

        # ── Lecture automatique de l'architecture MLP depuis le checkpoint ──
        # Optuna a choisi des hidden sizes différents des valeurs par défaut
        # → on lit les vraies dimensions directement dans le fichier .pt
        def load_mlp_from_checkpoint(pt_file, in_dim):
            sd = torch.load(pt_file, map_location="cpu")
            h1 = sd["net.0.weight"].shape[0]   # première couche Linear
            h2 = sd["net.4.weight"].shape[0]   # deuxième couche Linear
            h3 = sd["net.8.weight"].shape[0]   # troisième couche Linear
            log.info("   MLP architecture détectée → h1=%d  h2=%d  h3=%d", h1, h2, h3)
            model = MLP(in_dim, hidden1=h1, hidden2=h2, hidden3=h3)
            model.load_state_dict(sd)
            return model

        # ── Registre des 4 modèles ────────────────────────────────────────
        mlp_pt = os.path.join(dl_folder, "mlp_best_model.pt")
        model_registry = {
            "MLP":         (mlp_pt,
                            load_mlp_from_checkpoint(mlp_pt, N_FEAT) if os.path.exists(mlp_pt) else None,
                            "flat"),
            "LSTM":        (os.path.join(dl_folder, "lstm_model.pt"),
                            RNNModel(rnn_type="LSTM"),
                            "seq"),
            "GRU":         (os.path.join(dl_folder, "gru_model.pt"),
                            RNNModel(rnn_type="GRU"),
                            "seq"),
            "Transformer": (os.path.join(dl_folder, "transformer_model.pt"),
                            TransformerClassifier(SEQ_LEN),
                            "seq"),
        }

        # ── Évaluation de chaque modèle disponible ────────────────────────
        os.makedirs("results", exist_ok=True)
        results_dl   = {}
        y_test_bin   = label_binarize(y_true, classes=[0, 1, 2]) if y_true is not None else None

        for name, (pt_file, model, input_type) in model_registry.items():
            if model is None:
                log.warning("⚠️  Modèle %s non initialisé — ignoré", name)
                continue
            if not os.path.exists(pt_file):
                log.warning("⚠️  %s introuvable dans %s — ignoré", pt_file, dl_folder)
                continue

            log.info("⏳ Chargement %s ...", name)
            # MLP already loaded via load_mlp_from_checkpoint — skip re-loading
            if name != "MLP":
                model.load_state_dict(torch.load(pt_file, map_location="cpu"))
            model.eval()

            # Format d'entrée : flat pour MLP, séquence pour LSTM/GRU/Transformer
            tensor_in = torch.tensor(X_new)
            if input_type == "seq":
                tensor_in = tensor_in.unsqueeze(-1)   # (N, seq_len, 1)

            with torch.no_grad():
                probs = torch.softmax(model(tensor_in), dim=1).numpy()
                preds = probs.argmax(axis=1)

            acc, auc = None, None
            if y_true is not None:
                n     = min(len(preds), len(y_true))
                acc   = (preds[:n] == y_true[:n]).mean()
                auc   = roc_auc_score(y_test_bin[:n], probs[:n], multi_class="ovr")
                log.info("  ✅ %-15s → Accuracy: %.4f | ROC-AUC: %.4f", name, acc, auc)

            results_dl[name] = {
                "Accuracy": round(float(acc), 4) if acc is not None else None,
                "ROC-AUC":  round(float(auc), 4) if auc is not None else None,
                "probs":    probs,
                "preds":    preds,
            }

        if not results_dl:
            log.warning("⚠️  Aucun modèle DL trouvé dans %s", dl_folder)
            return None, None, None

        # ── Choix du meilleur modèle DL par ROC-AUC ──────────────────────
        best_dl_name = max(results_dl,
                           key=lambda k: results_dl[k]["ROC-AUC"] or 0)
        best_dl_auc  = results_dl[best_dl_name]["ROC-AUC"]
        best_dl_acc  = results_dl[best_dl_name]["Accuracy"]
        log.info("🏆 Meilleur modèle DL : %s  (ROC-AUC: %.4f)", best_dl_name, best_dl_auc)

        # ── Sauvegarde du tableau récapitulatif ───────────────────────────
        df_dl_summary = pd.DataFrame({
            k: {"Accuracy": v["Accuracy"], "ROC-AUC": v["ROC-AUC"]}
            for k, v in results_dl.items()
        }).T
        df_dl_summary.to_csv("results/resultats_DL.csv")
        log.info("✅ results/resultats_DL.csv sauvegardé")

        return df_dl_summary, best_dl_name, best_dl_acc, best_dl_auc

    except Exception as e:
        log.warning("⚠️  Erreur DL : %s", str(e))
        import traceback; traceback.print_exc()
        return None, None, None


# ══════════════════════════════════════════════════════════════════════════════
# GRAPHIQUES COMPARATIFS
# ══════════════════════════════════════════════════════════════════════════════

def generer_graphiques(modeles, y_test, df_ml, best_ml_name,
                       df_dl=None, best_dl_name=None,
                       best_dl_acc=None, best_dl_auc=None):

    log.info("\n" + "="*60)
    log.info("📊 Génération des graphiques")
    log.info("="*60)

    os.makedirs("results", exist_ok=True)
    y_test_bin  = label_binarize(y_test, classes=[0, 1, 2])
    class_names = ["A l'heure", "Retard +15min", "Retard +30min"]
    colors      = ["blue", "green", "orange", "red", "purple"]

    # ROC Curves ML
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, cname in enumerate(class_names):
        ax = axes[i]
        for (nom, (model, X_te)), color in zip(modeles.items(), colors):
            y_proba = model.predict_proba(X_te)
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            auc = roc_auc_score(y_test_bin[:, i], y_proba[:, i])
            ax.plot(fpr, tpr, label=f"{nom} (AUC={auc:.3f})", color=color)
        ax.plot([0,1],[0,1], "k--", alpha=0.5)
        ax.set_title(f"ROC — {cname}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    plt.suptitle("Courbes ROC — Modèles ML", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/roc_curves.png", dpi=150)
    plt.close()
    log.info("✅ results/roc_curves.png")

    # Precision-Recall ML
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, cname in enumerate(class_names):
        ax = axes[i]
        for (nom, (model, X_te)), color in zip(modeles.items(), colors):
            y_proba = model.predict_proba(X_te)
            prec, rec, _ = precision_recall_curve(y_test_bin[:, i], y_proba[:, i])
            ap = average_precision_score(y_test_bin[:, i], y_proba[:, i])
            ax.plot(rec, prec, label=f"{nom} (AP={ap:.3f})", color=color)
        ax.set_title(f"Precision-Recall — {cname}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    plt.suptitle("Courbes Precision-Recall — Modèles ML", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/precision_recall_curves.png", dpi=150)
    plt.close()
    log.info("✅ results/precision_recall_curves.png")

    # Comparaison finale ML vs DL (tous les modèles DL + tous les modèles ML)
    noms_ml  = list(df_ml.index)
    accs_ml  = list(df_ml["Accuracy"].astype(float))
    aucs_ml  = list(df_ml["ROC-AUC"].astype(float))

    if df_dl is not None:
        noms_dl  = list(df_dl.index)
        accs_dl  = list(df_dl["Accuracy"].astype(float))
        aucs_dl  = list(df_dl["ROC-AUC"].astype(float))
        all_noms = noms_ml + noms_dl
        all_accs = accs_ml + accs_dl
        all_aucs = aucs_ml + aucs_dl
        colors_bar = (["steelblue"] * len(noms_ml) + ["crimson"] * len(noms_dl))
    else:
        all_noms   = noms_ml
        all_accs   = accs_ml
        all_aucs   = aucs_ml
        colors_bar = ["steelblue"] * len(noms_ml)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    bars0 = axes[0].bar(all_noms, all_accs, color=colors_bar, edgecolor="black")
    axes[0].set_title("Accuracy — ML (bleu) vs DL (rouge)", fontweight="bold")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(min(all_accs) - 0.05, 1.0)
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars0, all_accs):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.002,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    bars1 = axes[1].bar(all_noms, all_aucs, color=colors_bar, edgecolor="black")
    axes[1].set_title("ROC-AUC — ML (bleu) vs DL (rouge)", fontweight="bold")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].set_ylim(min(all_aucs) - 0.05, 1.0)
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars1, all_aucs):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.002,
                     f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Comparaison finale — Tous les modèles ML vs DL",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/comparaison_ML_DL.png", dpi=150)
    plt.close()
    log.info("✅ results/comparaison_ML_DL.png")


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE FINAL
# ══════════════════════════════════════════════════════════════════════════════

def pipeline_final(flights_path:       str  = "flights.csv",
                   airlines_path:      str  = "airlines.csv",
                   airports_path:      str  = "airports.csv",
                   cleaned_csv:        str  = "flights_cleaned_encoded.csv",
                   skip_preprocessing: bool = False,
                   dl_folder:          str  = "dl_models"):

    print("\n" + "="*60)
    print("🚀 PIPELINE FINAL — Prédiction des Retards de Vols")
    print("   Binôme 1 (Preprocessing) + Binôme 2 (ML) + Binôme 3 (DL)")
    print("="*60)

    # ── BINÔME 1 ──────────────────────────────────────────────
    if skip_preprocessing and os.path.exists(cleaned_csv):
        log.info("⏭️  Preprocessing ignoré — fichier existant : %s", cleaned_csv)
    else:
        print("\n📦 BINÔME 1 — Preprocessing...")
        cfg = PipelineConfig(
            flights_path  = flights_path,
            airlines_path = airlines_path,
            airports_path = airports_path,
            output_path   = cleaned_csv,
        )
        FlightPreprocessingPipeline(cfg).run()

    # ── BINÔME 2 ──────────────────────────────────────────────
    print("\n🤖 BINÔME 2 — Machine Learning...")
    modeles, df_ml, best_ml_name, y_test, X_test, X_test_scaled = pipeline_ML(cleaned_csv)

    # ── BINÔME 3 ──────────────────────────────────────────────
    print("\n🧠 BINÔME 3 — Deep Learning...")
    dl_result = pipeline_DL(cleaned_csv, dl_folder=dl_folder)

    df_dl, best_dl_name, best_dl_acc, best_dl_auc = (
        dl_result if dl_result[0] is not None else (None, None, None, None)
    )

    # ── Graphiques ────────────────────────────────────────────
    generer_graphiques(modeles, y_test, df_ml, best_ml_name,
                       df_dl, best_dl_name, best_dl_acc, best_dl_auc)

    # ── Résumé final ──────────────────────────────────────────
    print("\n" + "="*60)
    print("🎉 PIPELINE FINAL TERMINÉ !")
    print("="*60)

    print("\n📊 RÉSULTATS ML :")
    print(df_ml[["Accuracy", "ROC-AUC"]].to_string())
    print(f"\n🏆 Meilleur modèle ML : {best_ml_name}"
          f"  (ROC-AUC: {df_ml.loc[best_ml_name, 'ROC-AUC']:.4f})")

    if df_dl is not None:
        print("\n📊 RÉSULTATS DL :")
        print(df_dl[["Accuracy", "ROC-AUC"]].to_string())
        print(f"\n🏆 Meilleur modèle DL : {best_dl_name}"
              f"  (ROC-AUC: {best_dl_auc:.4f})")

        best_auc_ml = float(df_ml["ROC-AUC"].max())
        if best_dl_auc > best_auc_ml:
            print(f"\n✅ Le Deep Learning ({best_dl_name}) surpasse le Machine Learning ({best_ml_name}) !")
        else:
            print(f"\n✅ Le Machine Learning ({best_ml_name}) surpasse le Deep Learning ({best_dl_name}) !")
    else:
        print(f"\n⚠️  Modèles DL non trouvés dans '{dl_folder}/'")
        print("   → Vérifiez que les fichiers suivants existent :")
        for f in ["mlp_best_model.pt", "lstm_model.pt",
                  "gru_model.pt", "transformer_model.pt",
                  "scaler.pkl", "features.pkl"]:
            status = "✅" if os.path.exists(os.path.join(dl_folder, f)) else "❌"
            print(f"      {status}  {dl_folder}/{f}")

    print("\n📁 Fichiers générés :")
    for f in [
        cleaned_csv,
        "models/best_model_ML.pkl",
        "results/resultats_ML.csv",
        "results/resultats_DL.csv",
        "results/roc_curves.png",
        "results/precision_recall_curves.png",
        "results/comparaison_ML_DL.png",
    ]:
        status = "✅" if os.path.exists(f) else "⚠️ "
        print(f"   {status}  {f}")

    return df_ml, df_dl


# ══════════════════════════════════════════════════════════════════════════════
# LANCEMENT
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Pipeline Final — Prédiction Retards de Vols")
    p.add_argument("--flights",  default="flights.csv",                 help="Chemin flights.csv")
    p.add_argument("--airlines", default="airlines.csv",                help="Chemin airlines.csv")
    p.add_argument("--airports", default="airports.csv",                help="Chemin airports.csv")
    p.add_argument("--output",   default="flights_cleaned_encoded.csv", help="CSV nettoyé")
    p.add_argument("--skip_preprocessing", action="store_true",
                   help="Ignorer preprocessing si CSV nettoyé existe déjà")
    p.add_argument("--dl_folder", default="dl_models",
                   help="Dossier contenant les modèles DL (défaut: dl_models/)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline_final(
        flights_path       = args.flights,
        airlines_path      = args.airlines,
        airports_path      = args.airports,
        cleaned_csv        = args.output,
        skip_preprocessing = args.skip_preprocessing,
        dl_folder          = args.dl_folder,
    )