import pandas as pd
import numpy as np
import yfinance as yf
import os
import joblib


from sklearn.preprocessing import StandardScaler


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from datetime import datetime, timedelta


def prepare_data(days_back=365):
    """
    Récupère et prépare les données Bitcoin et Ethereum
    depuis Yahoo Finance.
    """

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)

    cryptos = {"BTC-USD": "BTC", "ETH-USD": "ETH"}

    all_data = []

    for ticker, symbol in cryptos.items():

        df = yf.download(ticker, start=start_date, end=end_date, progress=False)

        # Gestion des colonnes MultiIndex de yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        df = df.loc[:, ~df.columns.duplicated()]

        # Suppression des lignes invalides
        df = df.dropna(subset=["Close", "High", "Low", "Open"])

        # Ajout du symbole BTC ou ETH
        df["symbol"] = symbol

        all_data.append(df)

    # Regrouper BTC et ETH
    data = pd.concat(all_data, ignore_index=True)

    # Trier chronologiquement
    data = data.sort_values(["symbol", "Date"]).reset_index(drop=True)

    return data


def calculate_rsi(data, window=14):
    """Calcule le Relative Strength Index (RSI)."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calcule le MACD et sa ligne de signal."""
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()

    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_histogram = macd - macd_signal

    return macd, macd_signal, macd_histogram


def engineer_features(df):
    """
    Crée les indicateurs techniques utilisés par le modèle.
    """

    df = df.copy()
    df = df.sort_values(["symbol", "Date"]).reset_index(drop=True)

    feature_dfs = []

    for symbol in df["symbol"].unique():

        crypto_df = df[df["symbol"] == symbol].copy()

        # Price features
        crypto_df["Daily_Return"] = (
            (crypto_df["Close"] - crypto_df["Open"]) / crypto_df["Open"]
        ) * 100

        crypto_df["Price_Change"] = crypto_df["Close"].pct_change() * 100

        crypto_df["Volatility"] = (
            (crypto_df["High"] - crypto_df["Low"]) / crypto_df["Close"]
        ) * 100

        # Lagged features
        for lag in [1, 2, 3, 5, 7]:
            crypto_df[f"Close_Lag_{lag}"] = crypto_df["Close"].shift(lag)

        # Moving averages
        crypto_df["MA_7"] = crypto_df["Close"].rolling(window=7).mean()
        crypto_df["MA_20"] = crypto_df["Close"].rolling(window=20).mean()
        crypto_df["MA_30"] = crypto_df["Close"].rolling(window=30).mean()
        crypto_df["MA_50"] = crypto_df["Close"].rolling(window=50).mean()

        # Moving average ratios
        crypto_df["MA_Ratio_7_30"] = crypto_df["MA_7"] / crypto_df["MA_30"]
        crypto_df["Price_to_MA7"] = crypto_df["Close"] / crypto_df["MA_7"]
        crypto_df["Price_to_MA30"] = crypto_df["Close"] / crypto_df["MA_30"]

        # Bollinger Bands
        crypto_df["Std_20"] = crypto_df["Close"].rolling(window=20).std()
        crypto_df["Upper_BB"] = crypto_df["MA_20"] + 2 * crypto_df["Std_20"]
        crypto_df["Lower_BB"] = crypto_df["MA_20"] - 2 * crypto_df["Std_20"]

        crypto_df["BB_Position"] = (crypto_df["Close"] - crypto_df["Lower_BB"]) / (
            crypto_df["Upper_BB"] - crypto_df["Lower_BB"]
        )

        # Rate of Change
        crypto_df["ROC_5"] = (
            (crypto_df["Close"] - crypto_df["Close"].shift(5))
            / crypto_df["Close"].shift(5)
        ) * 100

        crypto_df["ROC_10"] = (
            (crypto_df["Close"] - crypto_df["Close"].shift(10))
            / crypto_df["Close"].shift(10)
        ) * 100

        # RSI
        crypto_df["RSI_14"] = calculate_rsi(crypto_df["Close"], window=14)

        # MACD
        crypto_df["MACD"], crypto_df["MACD_Signal"], crypto_df["MACD_Histogram"] = (
            calculate_macd(crypto_df["Close"])
        )

        # ATR
        crypto_df["TR1"] = crypto_df["High"] - crypto_df["Low"]
        crypto_df["TR2"] = abs(crypto_df["High"] - crypto_df["Close"].shift(1))
        crypto_df["TR3"] = abs(crypto_df["Low"] - crypto_df["Close"].shift(1))

        crypto_df["True_Range"] = crypto_df[["TR1", "TR2", "TR3"]].max(axis=1)

        crypto_df["ATR_14"] = crypto_df["True_Range"].rolling(window=14).mean()

        crypto_df.drop(["TR1", "TR2", "TR3", "True_Range"], axis=1, inplace=True)

        # Volume
        crypto_df["Volume_Change"] = crypto_df["Volume"].pct_change() * 100

        crypto_df["Volume_MA_7"] = crypto_df["Volume"].rolling(window=7).mean()

        crypto_df["Volume_Ratio"] = crypto_df["Volume"] / crypto_df["Volume_MA_7"]

        crypto_df["Volume_ROC_5"] = (
            (crypto_df["Volume"] - crypto_df["Volume"].shift(5))
            / crypto_df["Volume"].shift(5)
        ) * 100

        crypto_df["Volume_Spike"] = (
            crypto_df["Volume"] > crypto_df["Volume_MA_7"] * 1.5
        ).astype(int)

        # Additional indicators
        crypto_df["HL_Spread"] = crypto_df["High"] - crypto_df["Low"]

        crypto_df["Rolling_Volatility_7"] = (
            crypto_df["Price_Change"].rolling(window=7).std()
        )

        crypto_df["Rolling_Volatility_30"] = (
            crypto_df["Price_Change"].rolling(window=30).std()
        )

        crypto_df["MA_Cross_Signal"] = (crypto_df["MA_7"] > crypto_df["MA_30"]).astype(
            int
        )

        crypto_df["Distance_MA7"] = (
            (crypto_df["Close"] - crypto_df["MA_7"]) / crypto_df["MA_7"]
        ) * 100

        crypto_df["Distance_MA30"] = (
            (crypto_df["Close"] - crypto_df["MA_30"]) / crypto_df["MA_30"]
        ) * 100

        crypto_df["Price_Direction"] = (
            crypto_df["Close"] > crypto_df["Close"].shift(1)
        ).astype(int)

        crypto_df["Consecutive_Trend"] = crypto_df.groupby(
            (
                crypto_df["Price_Direction"] != crypto_df["Price_Direction"].shift()
            ).cumsum()
        )["Price_Direction"].transform("count")

        feature_dfs.append(crypto_df)

    df_with_features = pd.concat(feature_dfs, ignore_index=True)

    df_with_features = df_with_features.dropna().reset_index(drop=True)

    return df_with_features


def load_model(model_dir="models"):
    """
    Recharge les modèles, scalers et features avec joblib.
    """

    model_btc = joblib.load(os.path.join(model_dir, "bitcoin_best_model.pkl"))

    scaler_btc = joblib.load(os.path.join(model_dir, "bitcoin_scaler.pkl"))

    model_eth = joblib.load(os.path.join(model_dir, "ethereum_best_model.pkl"))

    scaler_eth = joblib.load(os.path.join(model_dir, "ethereum_scaler.pkl"))

    feature_cols = joblib.load(os.path.join(model_dir, "feature_columns.pkl"))

    return model_btc, scaler_btc, model_eth, scaler_eth, feature_cols


def create_target(df):
    """
    Crée la variable Target.

    Target = 1 : hausse du prix du lendemain > 0.5 %
    Target = 0 : sinon
    """

    df = df.copy()

    # Conversion de la date
    df["Date"] = pd.to_datetime(df["Date"])

    # Ordre chronologique
    df = df.sort_values(["Name", "Date"]).reset_index(drop=True)

    # Prix du lendemain
    df["Next_Close"] = df.groupby("Name")["Close"].shift(-1)

    # Variation en pourcentage
    df["Pct_Change"] = ((df["Next_Close"] - df["Close"]) / df["Close"]) * 100

    # Target
    df["Target"] = (df["Pct_Change"] > 0.5).astype(int)

    # Supprimer les dernières lignes sans lendemain
    df = df[df["Next_Close"].notna()].copy()

    # Colonnes temporaires inutiles
    df.drop(["Next_Close", "Pct_Change"], axis=1, inplace=True)

    return df


def engineer_training_features(df):
    """
    Crée les indicateurs techniques pour les données
    historiques utilisées pour l'entraînement.
    """

    df = df.copy()
    feature_dfs = []

    for name in df["Name"].unique():

        crypto_df = df[df["Name"] == name].copy()

        # Price features
        crypto_df["Daily_Return"] = (
            (crypto_df["Close"] - crypto_df["Open"]) / crypto_df["Open"]
        ) * 100

        crypto_df["Price_Change"] = crypto_df["Close"].pct_change() * 100

        crypto_df["Volatility"] = (
            (crypto_df["High"] - crypto_df["Low"]) / crypto_df["Close"]
        ) * 100

        # Lag features
        for lag in [1, 2, 3, 5, 7]:
            crypto_df[f"Close_Lag_{lag}"] = crypto_df["Close"].shift(lag)

        # Moving averages
        crypto_df["MA_7"] = crypto_df["Close"].rolling(7).mean()
        crypto_df["MA_20"] = crypto_df["Close"].rolling(20).mean()
        crypto_df["MA_30"] = crypto_df["Close"].rolling(30).mean()
        crypto_df["MA_50"] = crypto_df["Close"].rolling(50).mean()

        crypto_df["MA_Ratio_7_30"] = crypto_df["MA_7"] / crypto_df["MA_30"]

        crypto_df["Price_to_MA7"] = crypto_df["Close"] / crypto_df["MA_7"]

        crypto_df["Price_to_MA30"] = crypto_df["Close"] / crypto_df["MA_30"]

        # Bollinger Bands
        crypto_df["Std_20"] = crypto_df["Close"].rolling(20).std()

        crypto_df["Upper_BB"] = crypto_df["MA_20"] + 2 * crypto_df["Std_20"]

        crypto_df["Lower_BB"] = crypto_df["MA_20"] - 2 * crypto_df["Std_20"]

        crypto_df["BB_Position"] = (crypto_df["Close"] - crypto_df["Lower_BB"]) / (
            crypto_df["Upper_BB"] - crypto_df["Lower_BB"]
        )

        # ROC
        crypto_df["ROC_5"] = (
            (crypto_df["Close"] - crypto_df["Close"].shift(5))
            / crypto_df["Close"].shift(5)
        ) * 100

        crypto_df["ROC_10"] = (
            (crypto_df["Close"] - crypto_df["Close"].shift(10))
            / crypto_df["Close"].shift(10)
        ) * 100

        # RSI
        crypto_df["RSI_14"] = calculate_rsi(crypto_df["Close"], 14)

        # MACD
        crypto_df["MACD"], crypto_df["MACD_Signal"], crypto_df["MACD_Histogram"] = (
            calculate_macd(crypto_df["Close"])
        )

        # ATR
        crypto_df["TR1"] = crypto_df["High"] - crypto_df["Low"]

        crypto_df["TR2"] = abs(crypto_df["High"] - crypto_df["Close"].shift(1))

        crypto_df["TR3"] = abs(crypto_df["Low"] - crypto_df["Close"].shift(1))

        crypto_df["True_Range"] = crypto_df[["TR1", "TR2", "TR3"]].max(axis=1)

        crypto_df["ATR_14"] = crypto_df["True_Range"].rolling(14).mean()

        crypto_df.drop(["TR1", "TR2", "TR3", "True_Range"], axis=1, inplace=True)

        # Volume
        crypto_df["Volume_Change"] = crypto_df["Volume"].pct_change() * 100

        crypto_df["Volume_MA_7"] = crypto_df["Volume"].rolling(7).mean()

        crypto_df["Volume_Ratio"] = crypto_df["Volume"] / crypto_df["Volume_MA_7"]

        crypto_df["Volume_ROC_5"] = (
            (crypto_df["Volume"] - crypto_df["Volume"].shift(5))
            / crypto_df["Volume"].shift(5)
        ) * 100

        crypto_df["Volume_Spike"] = (
            crypto_df["Volume"] > crypto_df["Volume_MA_7"] * 1.5
        ).astype(int)

        # Additional indicators
        crypto_df["HL_Spread"] = crypto_df["High"] - crypto_df["Low"]

        crypto_df["Rolling_Volatility_7"] = crypto_df["Price_Change"].rolling(7).std()

        crypto_df["Rolling_Volatility_30"] = crypto_df["Price_Change"].rolling(30).std()

        crypto_df["MA_Cross_Signal"] = (crypto_df["MA_7"] > crypto_df["MA_30"]).astype(
            int
        )

        crypto_df["Distance_MA7"] = (
            (crypto_df["Close"] - crypto_df["MA_7"]) / crypto_df["MA_7"]
        ) * 100

        crypto_df["Distance_MA30"] = (
            (crypto_df["Close"] - crypto_df["MA_30"]) / crypto_df["MA_30"]
        ) * 100

        crypto_df["Price_Direction"] = (
            crypto_df["Close"] > crypto_df["Close"].shift(1)
        ).astype(int)

        crypto_df["Consecutive_Trend"] = crypto_df.groupby(
            (
                crypto_df["Price_Direction"] != crypto_df["Price_Direction"].shift()
            ).cumsum()
        )["Price_Direction"].transform("count")

        feature_dfs.append(crypto_df)

    df_features = pd.concat(feature_dfs, ignore_index=True)

    # Nettoyage
    df_features = df_features.dropna().reset_index(drop=True)

    df_features.replace([np.inf, -np.inf], np.nan, inplace=True)

    df_features = df_features.dropna().reset_index(drop=True)

    return df_features


def prepare_training_data(df_features):
    """
    Prépare les données d'entraînement pour Bitcoin et Ethereum
    et sélectionne les colonnes utilisées comme features.
    """

    btc_data = df_features[df_features["Name"] == "Bitcoin"].copy()

    eth_data = df_features[df_features["Name"] == "Ethereum"].copy()

    exclude_cols = ["SNo", "Name", "Symbol", "Date", "Target", "Marketcap"]

    feature_cols = [col for col in df_features.columns if col not in exclude_cols]

    return btc_data, eth_data, feature_cols


def split_and_scale_data(data, feature_cols, train_size=0.8):
    """
    Sépare chronologiquement les données en train/test
    puis applique StandardScaler.
    """

    split_idx = int(len(data) * train_size)

    train_data = data.iloc[:split_idx].copy()
    test_data = data.iloc[split_idx:].copy()

    X_train = train_data[feature_cols].values
    y_train = train_data["Target"].values

    X_test = test_data[feature_cols].values
    y_test = test_data["Target"].values

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (X_train_scaled, X_test_scaled, y_train, y_test, scaler)


def train_model(X_train_btc, y_train_btc, X_train_eth, y_train_eth):
    """
    Entraîne les modèles XGBoost pour Bitcoin et Ethereum.

    Parameters:
    - X_train_btc : données d'entraînement Bitcoin
    - y_train_btc : cible Bitcoin
    - X_train_eth : données d'entraînement Ethereum
    - y_train_eth : cible Ethereum

    Returns:
    - model_btc : modèle Bitcoin entraîné
    - model_eth : modèle Ethereum entraîné
    """

    best_params = {
        "n_estimators": 150,
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 1,
        "reg_alpha": 0.1,
        "reg_lambda": 1,
        "objective": "binary:logistic",
        "random_state": 0,
        "eval_metric": "logloss",
    }

    # Bitcoin
    model_btc = XGBClassifier(**best_params)
    model_btc.fit(X_train_btc, y_train_btc, verbose=False)

    # Ethereum
    model_eth = XGBClassifier(**best_params)
    model_eth.fit(X_train_eth, y_train_eth, verbose=False)

    return model_btc, model_eth


def apply_confidence_threshold(y_proba, threshold_high=0.70, threshold_low=0.30):
    """
    Applique les seuils de confiance :
    - UP = 1
    - DOWN = 0
    - HOLD = -1
    """

    predictions = np.full(len(y_proba), -1)

    predictions[y_proba > threshold_high] = 1
    predictions[y_proba < threshold_low] = 0

    return predictions


def evaluate_model(model, X_test, y_test, threshold_high=0.70, threshold_low=0.30):
    """
    Évalue un modèle avec Accuracy, Precision, Recall et F1-score.
    """

    # Prédictions standards
    y_pred = model.predict(X_test)

    standard_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    # Probabilité de la classe UP
    y_proba = model.predict_proba(X_test)[:, 1]

    # Prédictions avec seuil de confiance
    y_pred_confident = apply_confidence_threshold(
        y_proba, threshold_high, threshold_low
    )

    confident_mask = y_pred_confident != -1

    confident_metrics = {
        "accuracy": accuracy_score(
            y_test[confident_mask], y_pred_confident[confident_mask]
        ),
        "precision": precision_score(
            y_test[confident_mask], y_pred_confident[confident_mask], zero_division=0
        ),
        "recall": recall_score(
            y_test[confident_mask], y_pred_confident[confident_mask], zero_division=0
        ),
        "f1": f1_score(
            y_test[confident_mask], y_pred_confident[confident_mask], zero_division=0
        ),
        "coverage": (confident_mask.sum() / len(confident_mask)) * 100,
    }

    return standard_metrics, confident_metrics


def save_model(
    model_btc, scaler_btc, model_eth, scaler_eth, feature_cols, model_dir="models"
):
    """
    Sauvegarde les modèles, scalers et features avec joblib.
    """

    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(model_btc, os.path.join(model_dir, "bitcoin_best_model.pkl"))

    joblib.dump(scaler_btc, os.path.join(model_dir, "bitcoin_scaler.pkl"))

    joblib.dump(model_eth, os.path.join(model_dir, "ethereum_best_model.pkl"))

    joblib.dump(scaler_eth, os.path.join(model_dir, "ethereum_scaler.pkl"))

    joblib.dump(feature_cols, os.path.join(model_dir, "feature_columns.pkl"))

    print(f"Models saved successfully in: {model_dir}")
