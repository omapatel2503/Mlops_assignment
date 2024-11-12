from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import mlflow
import mlflow.sklearn


class IrisDataProcessor:
    def __init__(self):
        self.iris = load_iris()
        self.data = pd.DataFrame(data=np.c_[self.iris['data'], self.iris['target']],
                                 columns=self.iris['feature_names'] + ['target'])

    def prepare_data(self):
        # Split data into training and testing sets
        X = self.data.drop('target', axis=1)
        y = self.data['target']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        # Feature scaling using StandardScaler
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_feature_stats(self):
        # Calculate basic statistics for each feature
        stats = self.data.describe()
        return stats


def evaluate_model(model, X, y):
    # Implement cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracy = cross_val_score(model, X, y, cv=cv, scoring='accuracy').mean()
    precision = cross_val_score(model, X, y, cv=cv, scoring='precision_weighted').mean()
    recall = cross_val_score(model, X, y, cv=cv, scoring='recall_weighted').mean()

    return accuracy, precision, recall

mlflow.set_tracking_uri("file:./mlruns")  # Ensure this is the directory for logging
experiment_name = "Iris_ClassificationExperiment"
mlflow.set_experiment(experiment_name)  # Create or set the experiment

processor = IrisDataProcessor()
X_train, X_test, y_train, y_test = processor.prepare_data()
feature_stats = processor.get_feature_stats()

with mlflow.start_run(run_name="RandomForest Classifier On iris data"):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=20)
    rf_accuracy, rf_precision, rf_recall = evaluate_model(rf_model, X_train, y_train)

    print(f'Random Forest Accuracy: {rf_accuracy}')
    print(f'Random Forest Precision: {rf_precision}')
    print(f'Random Forest Recall: {rf_recall}')

    # Log metrics and model in MLflow
    mlflow.log_metric("Accuracy", rf_accuracy)
    mlflow.log_metric("Precision", rf_precision)
    mlflow.log_metric("Recall", rf_recall)
    mlflow.sklearn.log_model(rf_model, "RandomForest Model")

# Train and log Linear Regression Model
with mlflow.start_run(run_name="Logistic regression on iris data"):
    lr_model = LogisticRegression(max_iter=1000)
    lr_accuracy, lr_precision, lr_recall = evaluate_model(lr_model, X_train, y_train)

    

    print(f'Logistic Accuracy: {lr_accuracy}')
    print(f'Logistic Precision: {lr_precision}')
    print(f'Logistic Recall: {lr_recall}')
    

    # Log metrics and model in MLflow
    mlflow.log_metric("Accuracy", lr_accuracy)
    mlflow.log_metric("Precision", lr_precision)
    mlflow.log_metric("Recall", lr_recall)
    mlflow.sklearn.log_model(lr_model, "Logistic regression model")



# Train Logistic Regression
# logreg_model = LogisticRegression(max_iter=1000)
# logreg_accuracy, logreg_precision, logreg_recall = evaluate_model(logreg_model, X_train, y_train)

# # Train Random Forest
# rf_model = RandomForestClassifier()
# rf_accuracy, rf_precision, rf_recall = evaluate_model(rf_model, X_train, y_train)

# print("Logistic Regression Metrics:")
# print(f"Accuracy: {logreg_accuracy:.4f}")
# print(f"Precision: {logreg_precision:.4f}")
# print(f"Recall: {logreg_recall:.4f}")

# print("\nRandom Forest Metrics:")
# print(f"Accuracy: {rf_accuracy:.4f}")
# print(f"Precision: {rf_precision:.4f}")
# print(f"Recall: {rf_recall:.4f}")
