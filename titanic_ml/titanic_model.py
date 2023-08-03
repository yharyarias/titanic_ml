import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from sklearn.preprocessing import LabelEncoder

class TitanicModel:
    def __init__(self, model_path='titanic_ml/models/model.pkl'):
        """
        Initialize the TitanicModel.

        Args:
            model_path (str): Path to save or load the trained model.
        """
        self.model_path = model_path
        self.model = None
        self.label_encoder = LabelEncoder()

    def preprocess_data(self, data):
        """
        Preprocess the data for training or prediction.

        Args:
            data (pd.DataFrame): Input data as a DataFrame.

        Returns:
            tuple: Tuple containing the preprocessed features (X) and labels (y).
        """
        # Data preprocessing code
        data['Age'].fillna(data['Age'].median(), inplace=True)
        data['Fare'].fillna(data['Fare'].median(), inplace=True)
        data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

        data['Sex'] = self.label_encoder.fit_transform(data['Sex'])
        data['Embarked'] = self.label_encoder.fit_transform(data['Embarked'])

        X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
        y = data['Survived']

        return X, y

    def train(self, data_path='titanic_ml/data/titanic.csv'):
        """
        Train the model on the specified dataset.

        Args:
            data_path (str): Path to the dataset for training.
        """
        data = pd.read_csv(data_path)
        X, y = self.preprocess_data(data)

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(X_train, y_train)

        joblib.dump(self.model, self.model_path)

    def predict(self, data_path='titanic_ml/data/titanic.csv', predicted_file='titanic_ml/data/predicted_data.csv'):
        """
        Make predictions using the trained model on a new dataset.

        Args:
            data_path (str): Path to the new dataset for prediction.
        """
        data = pd.read_csv(data_path)
        X, _ = self.preprocess_data(data)

        self.load_model()
        predictions = self.model.predict(X)

        data['Predicted_Survived'] = predictions
        data.to_csv(predicted_file, index=False)
        print(data[['PassengerId', 'Predicted_Survived']])
        print(f"Predictions saved to {predicted_file}")

    def evaluate(self, data_path='titanic_ml/data/titanic.csv'):
        """
        Evaluate the model's accuracy on the specified dataset.

        Args:
            data_path (str): Path to the dataset for evaluation.
        """
        # Load the model before evaluation
        self.load_model()

        data = pd.read_csv(data_path)
        X, y = self.preprocess_data(data)

        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred)

        print(f"Model Accuracy: {accuracy}")

    def load_model(self):
        """Load the trained model from the specified path."""
        if not self.model:
            self.model = joblib.load(self.model_path)

    def get_model(self):
        """Get the trained model."""
        return self.model
