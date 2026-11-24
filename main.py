import argparse
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)

parser = argparse.ArgumentParser()
parser.add_argument("--prepare", action="store_true")
parser.add_argument("--train", action="store_true")
parser.add_argument("--evaluate", action="store_true")
parser.add_argument("--save", action="store_true")
parser.add_argument("--load", action="store_true")
args = parser.parse_args()

x_train, x_test, y_train, y_test = prepare_data("adult.csv")

if args.prepare:
    print("Donnees preparees")
    print("x_train :", x_train.shape)
    print("x_test :", x_test.shape)

if args.train:
    model = train_model(x_train, y_train)
    print("Modele entraine")

if args.evaluate:
    model = train_model(x_train, y_train)
    evaluate_model(model, x_test, y_test)

if args.save:
    model = train_model(x_train, y_train)
    save_model(model, "model.joblib")

if args.load:
    model = load_model("model.joblib")
    evaluate_model(model, x_test, y_test)
