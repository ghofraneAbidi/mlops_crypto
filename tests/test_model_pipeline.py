import sys
import os
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from model_pipeline import (
    train_model,
    evaluate_model,
    save_model,
    load_model
)

# Permet à pytest de trouver model_pipeline.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# --------------------------------------------------
# TEST 1 : Entraînement des modèles
# --------------------------------------------------

def test_train_model():
    # Petites données artificielles pour le test
    X_btc = np.random.rand(20, 4)
    y_btc = np.array([0, 1] * 10)

    X_eth = np.random.rand(20, 4)
    y_eth = np.array([0, 1] * 10)

    model_btc, model_eth = train_model(
        X_btc,
        y_btc,
        X_eth,
        y_eth
    )

    assert model_btc is not None
    assert model_eth is not None


# --------------------------------------------------
# TEST 2 : Évaluation d'un modèle
# --------------------------------------------------

def test_evaluate_model():
    X = np.random.rand(30, 4)
    y = np.array([0, 1] * 15)

    model = XGBClassifier(
        n_estimators=5,
        max_depth=2,
        random_state=42
    )

    model.fit(X, y)

    standard, confidence = evaluate_model(model, X, y)

    assert "accuracy" in standard
    assert "precision" in standard
    assert "recall" in standard
    assert "f1" in standard

    assert 0 <= standard["accuracy"] <= 1


# --------------------------------------------------
# TEST 3 : Sauvegarde et rechargement
# --------------------------------------------------

def test_save_and_load_model(tmp_path):
    X = np.random.rand(20, 4)
    y = np.array([0, 1] * 10)

    model_btc = XGBClassifier(n_estimators=5, random_state=42)
    model_eth = XGBClassifier(n_estimators=5, random_state=42)

    model_btc.fit(X, y)
    model_eth.fit(X, y)

    scaler_btc = StandardScaler().fit(X)
    scaler_eth = StandardScaler().fit(X)

    feature_cols = ["feature1", "feature2", "feature3", "feature4"]

    model_dir = str(tmp_path)

    save_model(
        model_btc,
        scaler_btc,
        model_eth,
        scaler_eth,
        feature_cols,
        model_dir=model_dir
    )

    loaded = load_model(model_dir=model_dir)

    assert loaded is not None

    # Vérifie que les fichiers ont réellement été créés
    assert os.path.exists(os.path.join(model_dir, "bitcoin_best_model.pkl"))
    assert os.path.exists(os.path.join(model_dir, "bitcoin_scaler.pkl"))
    assert os.path.exists(os.path.join(model_dir, "ethereum_best_model.pkl"))
    assert os.path.exists(os.path.join(model_dir, "ethereum_scaler.pkl"))
    assert os.path.exists(os.path.join(model_dir, "feature_columns.pkl"))
