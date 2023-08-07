# Titanic ML Package

Titanic ML is a Python package that provides a simple implementation of a machine learning model to predict survival on the Titanic dataset from Kaggle. The package includes functionalities to train the model, make predictions, and evaluate the model's accuracy.

## Installation

To install the Titanic ML package, you can use `pip`:

```bash
pip install titanic-ml
```

### Usage
#### 1. Train the Model
To train the model, you need to provide the path to the training dataset. By default, the trained model will be saved as `titanic_ml/models/model.pkl`. You can optionally specify a different path to save the model.

```bash
titanic-ml train --model-path /path/to/save/model.pkl --data-path /path/to/training/dataset.csv
```

#### 2. Make Predictions
To make predictions using the trained model, you need to provide the path to the dataset on which you want to make predictions. The predictions will be saved in the file `titanic_ml/data/predicted_data.csv` by default.

```bash
titanic-ml predict --model-path /path/to/saved/model.pkl --data-path /path/to/dataset.csv
```

#### 3. Evaluate the Model
To evaluate the model's accuracy on a dataset, you can use the following command:

```bash
titanic-ml evaluate --model-path /path/to/saved/model.pkl --data-path /path/to/dataset.csv
```

### Project Structure
The Titanic ML package has the following project structure:

```bash
titanic_ml_package/
├── titanic_ml/
│   ├── __init__.py
│   ├── cli.py
│   ├── titanic_model.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── titanic.csv
│   │   └── predicted_data.csv
│   └── models/
│       ├── __init__.py
│       └── model.pkl
├── tests/
│   ├── __init__.py
│   └── test_titanic_model.py
├── setup.py
└── README.md
```

* `titanic_ml/`: Package directory containing the source code.
* `titanic_ml/cli.py`: CLI script defining the command-line interface.
* `titanic_ml/titanic_model.py`: Module with the TitanicModel class for training, predicting, and evaluating the model.
* `titanic_ml/data/`: Directory containing the Titanic dataset (titanic.csv) and the predicted data (predicted_data.csv).
* `titanic_ml/models/`: Directory where the trained model will be saved as model.pkl.
* `tests/`: Directory containing unit tests for the titanic_model module.
* `setup.py`: Script for packaging the Titanic ML package for distribution.
* `README.md`: This file, providing documentation on how to use the Titanic ML package.


### Testing and Coverage
The Titanic ML package includes unit tests for the `titanic_model` module. To run the tests and generate a coverage report, use the following command:

```bash
pytest --cov=titanic_ml --cov-report html
```

The coverage report will be available in the `htmlcov/` directory.

### Dependencies
* Python 3.x
* pandas
* scikit-learn
* joblib
* click
* pytest
* pytest-cov
* coverage

### License

For more information on how to use the Titanic ML package, you can refer to the package documentation or contact the authors.

* Note: Please make sure to replace `/path/to/save/model.pkl`, `/path/to/training/dataset.csv`, and other placeholder paths in the usage examples with the actual paths on your system.
