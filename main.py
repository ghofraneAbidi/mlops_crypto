import pandas as pd
import argparse
from model_pipeline import (
    create_target,
    engineer_training_features,
    prepare_training_data,
    split_and_scale_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


def main():

    print("=" * 60)
    print("CRYPTO PRICE PREDICTION - TRAINING PIPELINE")
    print("=" * 60)

    # 1. Charger les données historiques
    print("\n1. Chargement des donnees...")

    df = pd.read_csv("data/combined_crypto_dataset.csv")

    print(f"Dataset : {df.shape}")

    # 2. Créer Target
    print("\n2. Creation de Target...")

    df = create_target(df)

    print(f"Dataset apres Target : {df.shape}")

    # 3. Feature Engineering
    print("\n3. Feature Engineering...")

    df_features = engineer_training_features(df)

    print(f"Dataset avec features : {df_features.shape}")

    # 4. Préparer BTC / ETH
    print("\n4. Preparation Bitcoin / Ethereum...")

    btc_data, eth_data, feature_cols = prepare_training_data(df_features)

    print(f"Bitcoin : {len(btc_data)} lignes")
    print(f"Ethereum : {len(eth_data)} lignes")
    print(f"Nombre de features : {len(feature_cols)}")

    # 5. Split + Scaling
    print("\n5. Split et Scaling...")

    X_train_btc, X_test_btc, y_train_btc, y_test_btc, scaler_btc = split_and_scale_data(
        btc_data, feature_cols
    )

    X_train_eth, X_test_eth, y_train_eth, y_test_eth, scaler_eth = split_and_scale_data(
        eth_data, feature_cols
    )

    print(f"BTC : Train={len(X_train_btc)} | " f"Test={len(X_test_btc)}")

    print(f"ETH : Train={len(X_train_eth)} | " f"Test={len(X_test_eth)}")

    # 6. Entraînement
    print("\n6. Entrainement des modeles...")

    model_btc, model_eth = train_model(
        X_train_btc, y_train_btc, X_train_eth, y_train_eth
    )

    print("Entrainement termine.")

    # 7. Evaluation Bitcoin
    print("\n7. Evaluation Bitcoin...")

    btc_standard, btc_confident = evaluate_model(model_btc, X_test_btc, y_test_btc)

    print("Standard :", btc_standard)
    print("Confiance :", btc_confident)

    # 8. Evaluation Ethereum
    print("\n8. Evaluation Ethereum...")

    eth_standard, eth_confident = evaluate_model(model_eth, X_test_eth, y_test_eth)

    print("Standard :", eth_standard)
    print("Confiance :", eth_confident)

    # 9. Sauvegarde des modeles
    print("\n9. Sauvegarde des modeles...")

    save_model(
        model_btc,
        scaler_btc,
        model_eth,
        scaler_eth,
        feature_cols,
        model_dir="models_test",
    )

    # 10. Rechargement des modeles
    print("\n10. Rechargement des modeles...")

    (
        loaded_btc_model,
        loaded_btc_scaler,
        loaded_eth_model,
        loaded_eth_scaler,
        loaded_feature_cols,
    ) = load_model(model_dir="models_test")

    print("Modeles recharges avec succes.")
    print(f"Nombre de features rechargees : {len(loaded_feature_cols)}")

    print("\n" + "=" * 60)
    print("PIPELINE TERMINE AVEC SUCCES")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline ML de prediction Bitcoin et Ethereum"
    )

    parser.add_argument(
        "--run", action="store_true", help="Executer le pipeline complet d'entrainement"
    )

    args = parser.parse_args()

    if args.run:
        main()
    else:
        parser.print_help()
