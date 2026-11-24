"""Exemple de tests unitaires - adapte aux sorties réelles de tes fonctions."""

from model_pipeline import prepare_data, train_model


def test_prepare_data_shapes():
    X_train, X_test, y_train, y_test = prepare_data("adult.csv")
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_train) > 0


def test_train_model_predicts():
    X_train, X_test, y_train, _ = prepare_data("adult.csv")
    model = train_model(X_train, y_train)
    preds = model.predict(X_test[:5])
    assert len(preds) == 5
