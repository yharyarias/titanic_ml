import pandas as pd
import os
import pytest
from titanic_ml.titanic_model import TitanicModel

@pytest.fixture
def titanic_model():
    return TitanicModel()

def test_preprocess_data(titanic_model):
    data = pd.DataFrame({
        'Pclass': [1, 2, 3],
        'Sex': ['male', 'female', 'male'],
        'Age': [30, None, 25],
        'SibSp': [1, 0, 1],
        'Parch': [0, 1, 2],
        'Fare': [50, 25, None],
        'Embarked': ['C', None, 'S'],
        'Survived': [1, 0, 1]
    })
    X, y = titanic_model.preprocess_data(data)
    assert X.shape == (3, 7)
    assert y.shape == (3,)

def test_train_and_evaluate(titanic_model):
    data_path = 'test_data.csv'
    data = pd.DataFrame({
        'Pclass': [1, 2, 3],
        'Sex': ['male', 'female', 'male'],
        'Age': [30, 25, 35],
        'SibSp': [1, 0, 1],
        'Parch': [0, 1, 2],
        'Fare': [50, 25, 30],
        'Embarked': ['C', 'S', 'Q'],
        'Survived': [1, 0, 1]
    })
    data.to_csv(data_path, index=False)

    titanic_model.train(data_path)
    model = titanic_model.get_model()
    assert model is not None

    titanic_model.evaluate(data_path)

    # Clean up
    os.remove(data_path)

def test_predict(titanic_model):
    data_path = 'test_data.csv'
    data = pd.DataFrame({
        'Pclass': [1, 2, 3],
        'Sex': ['male', 'female', 'male'],
        'Age': [30, 25, 35],
        'SibSp': [1, 0, 1],
        'Parch': [0, 1, 2],
        'Fare': [50, 25, 30],
        'Embarked': ['C', 'S', 'Q'],
        'Survived': [1, 0, 1]
    })

    data['PassengerId'] = range(1, len(data)+1)
    data.to_csv(data_path, index=False)

    titanic_model.train(data_path)

    # Test prediction
    predicted_file = 'titanic_ml/data/predicted_data.csv'
    titanic_model.predict(data_path)

    # Clean up
    os.remove(data_path)
    os.remove(predicted_file)

def test_load_model(titanic_model):
    # Check if model is not loaded initially
    assert titanic_model.model is None

    # Load the model
    titanic_model.load_model()

    # Check if the model is loaded
    assert titanic_model.model is not None

def test_get_model(titanic_model):
    # Check if model is not loaded initially
    assert titanic_model.get_model() is None

    # Load the model
    titanic_model.load_model()

    # Check if the model is loaded
    assert titanic_model.get_model() is not None

