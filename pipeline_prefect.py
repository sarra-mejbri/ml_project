import argparse
import subprocess
import sys
from pathlib import Path

from prefect import flow, task, get_run_logger


from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

PROJECT_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable  # python du venv actif
MODEL_PATH = PROJECT_DIR / "model.joblib"
DATA_PATH = str(PROJECT_DIR / "adult.csv")
REQUIREMENTS = PROJECT_DIR / "requirements.txt"


CODE_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


def _run(cmd, allow_fail=False):
    """Lance une commande shell et logge sa sortie."""
    logger = get_run_logger()
    logger.info("$ %s", " ".join(cmd))
    result = subprocess.run(  # nosec B603
        cmd, cwd=PROJECT_DIR, capture_output=True, text=True, check=False
    )
    if result.stdout:
        logger.info(result.stdout)
    if result.stderr:
        logger.warning(result.stderr)
    if result.returncode != 0 and not allow_fail:
        raise RuntimeError(f"Échec ({result.returncode}) : {' '.join(cmd)}")
    return result.returncode


# ---------------------------------------------------------------------------
# Tasks - dépendances
# ---------------------------------------------------------------------------
@task(name="install-dependencies")
def install_dependencies():
    _run([PYTHON, "-m", "pip", "install", "-q", "-r", str(REQUIREMENTS)])


# ---------------------------------------------------------------------------
# Tasks - suivi du code (étape 4)
# ---------------------------------------------------------------------------
@task(name="format-code")
def format_code():
    _run([PYTHON, "-m", "black", *CODE_FILES])


@task(name="code-quality")
def code_quality():
    # allow_fail=True : on signale les problèmes sans bloquer tout le pipeline
    _run(
        [PYTHON, "-m", "flake8", "--max-line-length=100", *CODE_FILES],
        allow_fail=True,
    )


@task(name="code-security")
def code_security():
    _run([PYTHON, "-m", "bandit", "-q", *CODE_FILES], allow_fail=True)


@task(name="unit-tests")
def unit_tests():
    _run([PYTHON, "-m", "pytest", "tests/", "-v"])


@task(name="prepare-data")
def task_prepare_data():
    logger = get_run_logger()
    X_train, X_test, y_train, y_test = prepare_data(DATA_PATH)
    logger.info("Données prêtes : train=%s, test=%s", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test


@task(name="train-model")
def task_train_model(X_train, y_train):
    return train_model(X_train, y_train)


@task(name="save-model")
def task_save_model(model):
    save_model(model, MODEL_PATH)
    get_run_logger().info("Modèle sauvegardé : %s", MODEL_PATH)


@task(name="load-model")
def task_load_model():
    return load_model(MODEL_PATH)


@task(name="evaluate-model")
def task_evaluate_model(model, X_test, y_test):
    metrics = evaluate_model(model, X_test, y_test)
    get_run_logger().info("Métriques : %s", metrics)
    return metrics


@flow(name="code", log_prints=True)
def flow_code():
    format_code()
    code_quality()
    code_security()
    unit_tests()


@flow(name="train", log_prints=True)
def flow_train():
    X_train, X_test, y_train, y_test = task_prepare_data()
    model = task_train_model(X_train, y_train)
    task_save_model(model)


@flow(name="evaluate", log_prints=True)
def flow_evaluate():
    # On a besoin du jeu de test pour évaluer le modèle chargé
    _, X_test, _, y_test = task_prepare_data()
    model = task_load_model()
    task_evaluate_model(model, X_test, y_test)


@flow(name="all", log_prints=True)
def flow_all():
    install_dependencies()
    flow_code()  # étapes code AVANT la data
    X_train, X_test, y_train, y_test = task_prepare_data()  # 2
    model = task_train_model(X_train, y_train)  # 3
    task_save_model(model)  # 4
    task_evaluate_model(model, X_test, y_test)  # 6


FLOWS = {
    "all": flow_all,
    "train": flow_train,
    "entrainement": flow_train,
    "evaluate": flow_evaluate,
    "code": flow_code,
}


def main():
    parser = argparse.ArgumentParser(description="Pipeline MLOps Prefect")
    parser.add_argument("--flow", required=True, choices=FLOWS.keys())
    args = parser.parse_args()
    FLOWS[args.flow]()


if __name__ == "__main__":
    main()
