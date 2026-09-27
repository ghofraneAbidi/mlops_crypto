import argparse
import subprocess  # nosec B404
import sys

import pandas as pd
from prefect import flow, task

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

FILES = [
    "model_pipeline.py",
    "main.py",
    "pipeline_prefect.py",
]


def run_command(command):
    """Execute une commande et arrete la task si elle echoue."""
    subprocess.run(command, check=True)  # nosec B603


# ============================================================
# TASKS : SUIVI DU CODE
# ============================================================
@task(name="retrieve-project", log_prints=True)
def task_retrieve_project():
    """Recupere la derniere version du projet depuis GitHub."""
    print("Recuperation du projet depuis GitHub...")
    run_command(["git", "pull", "origin", "main"])
    print("Projet recupere avec succes.")


@task(name="install-dependencies", log_prints=True)
def task_install_dependencies():
    print("Verification / installation des dependances...")
    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-r",
            "requirements.txt",
        ]
    )
    print("Dependances OK.")


@task(name="format-code", log_prints=True)
def task_format_code():
    print("Formatage du code avec Black...")
    run_command(["black", *FILES])
    print("Formatage OK.")


@task(name="code-quality", log_prints=True)
def task_code_quality():
    print("Verification de la qualite avec Flake8...")
    run_command(["flake8", *FILES])
    print("Qualite du code OK.")


@task(name="security-check", log_prints=True)
def task_security():
    print("Verification de la securite avec Bandit...")
    run_command(["bandit", *FILES])
    print("Securite du code OK.")


@task(name="unit-tests", log_prints=True)
def task_unit_tests():
    print("Execution des tests unitaires avec Pytest...")
    run_command(
        [
            sys.executable,
            "-m",
            "pytest",
            "-v",
        ]
    )
    print("Tests unitaires OK.")


# ============================================================
# TASK : PREPARATION DES DONNEES
# ============================================================


@task(name="prepare-data", log_prints=True)
def task_prepare_data():
    print("Preparation des donnees...")

    df = pd.read_csv("data/combined_crypto_dataset.csv")
    print(f"Dataset initial : {df.shape}")

    df = create_target(df)
    df_features = engineer_training_features(df)

    btc_data, eth_data, feature_cols = prepare_training_data(df_features)

    (
        X_train_btc,
        X_test_btc,
        y_train_btc,
        y_test_btc,
        scaler_btc,
    ) = split_and_scale_data(btc_data, feature_cols)

    (
        X_train_eth,
        X_test_eth,
        y_train_eth,
        y_test_eth,
        scaler_eth,
    ) = split_and_scale_data(eth_data, feature_cols)

    print("Preparation terminee.")

    return {
        "X_train_btc": X_train_btc,
        "X_test_btc": X_test_btc,
        "y_train_btc": y_train_btc,
        "y_test_btc": y_test_btc,
        "scaler_btc": scaler_btc,
        "X_train_eth": X_train_eth,
        "X_test_eth": X_test_eth,
        "y_train_eth": y_train_eth,
        "y_test_eth": y_test_eth,
        "scaler_eth": scaler_eth,
        "feature_cols": feature_cols,
    }


# ============================================================
# TASK : ENTRAINEMENT
# ============================================================


@task(name="train-model", log_prints=True)
def task_train_model(data):
    print("Entrainement des modeles...")

    model_btc, model_eth = train_model(
        data["X_train_btc"],
        data["y_train_btc"],
        data["X_train_eth"],
        data["y_train_eth"],
    )

    print("Entrainement termine.")

    return model_btc, model_eth


# ============================================================
# TASK : SAUVEGARDE
# ============================================================


@task(name="save-model", log_prints=True)
def task_save_model(models, data):
    print("Sauvegarde des modeles...")

    model_btc, model_eth = models

    save_model(
        model_btc,
        data["scaler_btc"],
        model_eth,
        data["scaler_eth"],
        data["feature_cols"],
        model_dir="models_test",
    )

    print("Sauvegarde terminee.")


# ============================================================
# TASK : CHARGEMENT
# ============================================================


@task(name="load-model", log_prints=True)
def task_load_model():
    print("Chargement des modeles...")

    models = load_model(model_dir="models_test")

    print("Chargement termine.")

    return models


# ============================================================
# TASK : EVALUATION
# ============================================================


@task(name="evaluate-model", log_prints=True)
def task_evaluate_model(models, data):
    model_btc, model_eth = models

    print("Evaluation Bitcoin...")

    btc_standard, btc_confident = evaluate_model(
        model_btc,
        data["X_test_btc"],
        data["y_test_btc"],
    )

    print("BTC Standard :", btc_standard)
    print("BTC Confiance :", btc_confident)

    print("\nEvaluation Ethereum...")

    eth_standard, eth_confident = evaluate_model(
        model_eth,
        data["X_test_eth"],
        data["y_test_eth"],
    )

    print("ETH Standard :", eth_standard)
    print("ETH Confiance :", eth_confident)


# ============================================================
# FLOW CODE
# ============================================================


@flow(name="code", log_prints=True)
def flow_code():
    task_install_dependencies()
    task_format_code()
    task_code_quality()
    task_security()
    task_unit_tests()


# ============================================================
# FLOW TRAIN
# ============================================================


@flow(name="train", log_prints=True)
def flow_train():
    data = task_prepare_data()
    task_train_model(data)


# ============================================================
# FLOW EVALUATE
# ============================================================


@flow(name="evaluate", log_prints=True)
def flow_evaluate():
    data = task_prepare_data()
    loaded = task_load_model()

    model_btc = loaded[0]
    model_eth = loaded[2]

    task_evaluate_model(
        (model_btc, model_eth),
        data,
    )


# ============================================================
# FLOW ALL
# ============================================================


@flow(name="all", log_prints=True)
def flow_all():
    # Suivi du code
    task_retrieve_project()
    task_install_dependencies()
    task_format_code()
    task_code_quality()
    task_security()
    task_unit_tests()

    # Pipeline ML
    data = task_prepare_data()
    models = task_train_model(data)
    task_save_model(models, data)
    task_evaluate_model(models, data)


# ============================================================
# CLI
# ============================================================


FLOWS = {
    "all": flow_all,
    "code": flow_code,
    "train": flow_train,
    "entrainement": flow_train,
    "evaluate": flow_evaluate,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline MLOps Crypto orchestre avec Prefect"
    )

    parser.add_argument(
        "--flow",
        choices=FLOWS.keys(),
        required=True,
        help="Flow Prefect a executer",
    )

    args = parser.parse_args()

    FLOWS[args.flow]()
