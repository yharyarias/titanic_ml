import click
import pandas as pd
import joblib
from titanic_ml.titanic_model import TitanicModel

@click.group()
def cli():
    """
    TitanicML: A CLI for training, evaluating, and predicting with a machine learning model on the Titanic dataset.
    """
    pass

@cli.command()
@click.option('--model-path', default='titanic_ml/models/model.pkl', help='Path to save the trained model')
@click.option('--data-path', type=click.Path(exists=True), help='Path to the dataset for training')
def train(model_path, data_path):
    """
    Train the model on the specified dataset and save it.

    Args:
        model_path (str): Path to save the trained model.
        data_path (str): Path to the dataset for training.
    """
    model = TitanicModel(model_path)
    model.train(data_path)

@cli.command()
@click.option('--model-path', default='titanic_ml/models/model.pkl', help='Path to save the trained model')
@click.option('--data-path', type=click.Path(exists=True), help='Path to the dataset for training')
def evaluate(model_path, data_path):
    """
    Evaluate the model's accuracy on the specified dataset.

    Args:
        model_path (str): Path to the trained model.
        data_path (str): Path to the dataset for evaluation.
    """
    model = TitanicModel(model_path)
    model.evaluate(data_path)

@cli.command()
@click.option('--model-path', default='titanic_ml/models/model.pkl', help='Path to save the trained model')
@click.option('--data-path', type=click.Path(exists=True), help='Path to the dataset for training')
def predict(model_path, data_path):
    """
    Make predictions using the trained model on a new dataset.

    Args:
        model_path (str): Path to the trained model.
        data_path (str): Path to the new dataset for prediction.
    """
    model = TitanicModel(model_path)
    model.predict(data_path)

if __name__ == '__main__':
    cli()
