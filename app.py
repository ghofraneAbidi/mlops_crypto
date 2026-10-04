from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

from model_pipeline import load_model


# Création de l'application FastAPI
app = FastAPI(
    title="Crypto Price Prediction API",
    description="API de prédiction Bitcoin / Ethereum",
    version="1.0"
)


# Chargement des modèles
try:
    model_btc, scaler_btc, model_eth, scaler_eth, feature_cols = load_model(
        "models_test"
    )
    print("Modèles chargés avec succès.")
except Exception as e:
    print(f"Erreur chargement modèles : {e}")
    model_btc = None
    model_eth = None


# Données reçues par l'API
class PredictionRequest(BaseModel):
    crypto: str
    features: list[float]


# Route de test
@app.get("/")
def home():
    return {
        "message": "Crypto Price Prediction API fonctionne"
    }


# Route de prédiction
@app.post("/predict")
def predict(data: PredictionRequest):

    try:
        crypto = data.crypto.upper()

        # Vérification du nombre de features
        if len(data.features) != len(feature_cols):
            raise HTTPException(
                status_code=400,
                detail=f"Il faut {len(feature_cols)} features."
            )

        X = np.array(data.features).reshape(1, -1)

        # Bitcoin
        if crypto == "BTC":
            X_scaled = scaler_btc.transform(X)
            prediction = model_btc.predict(X_scaled)[0]

        # Ethereum
        elif crypto == "ETH":
            X_scaled = scaler_eth.transform(X)
            prediction = model_eth.predict(X_scaled)[0]

        else:
            raise HTTPException(
                status_code=400,
                detail="crypto doit être BTC ou ETH"
            )

        return {
            "crypto": crypto,
            "prediction": int(prediction)
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
