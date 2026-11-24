import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_data(path):
    df = pd.read_csv(path, skipinitialspace=True)
    df = df.drop_duplicates()
    df = df.replace("?", np.nan)
    df = df.dropna(subset=["workclass", "occupation"]).reset_index(drop=True)

    df["vit_chez_parents"] = (df["relationship"] == "Own-child").astype(int)
    df["log_capital_gain"] = np.log1p(df["capital-gain"])
    df["log_capital_loss"] = np.log1p(df["capital-loss"])

    num_cols = ["age", "education-num", "log_capital_gain", "log_capital_loss"]
    num = pd.DataFrame(StandardScaler().fit_transform(df[num_cols]), columns=num_cols)
    num["age_carre"] = num["age"] ** 2

    dummies = pd.get_dummies(
        df[["workclass", "marital-status", "occupation"]], drop_first=True, dtype=int
    )

    x = pd.concat([num, df[["vit_chez_parents"]], dummies], axis=1)
    y = pd.cut(df["hours-per-week"], [0, 34, 45, 99], labels=[0, 1, 2]).astype(int)

    return train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)


def train_model(x_train, y_train):
    model = HistGradientBoostingClassifier(
        learning_rate=0.03,
        max_iter=700,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=5.0,
        random_state=42,
    )
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy :", round(accuracy, 4))
    return accuracy


def save_model(model, path):
    joblib.dump(model, path)
    print("Modele sauvegarde")


def load_model(path):
    model = joblib.load(path)
    print("Modele charge")
    return model
