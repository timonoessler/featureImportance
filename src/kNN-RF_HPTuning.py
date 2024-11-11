# Benötigte Bibliotheken importieren
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import SimpleImputer
import numpy as np
import joblib

def load_and_prepare_data():
    # Load and prepare data
    print("Loading and preparing data")
    data = pd.read_csv("data/merged.csv")

    # Extract target column and encode categorical variables
    label_encoder = LabelEncoder()
    data['escalated'] = label_encoder.fit_transform(data['escalated'])

    # Convert categorical variables into numerical variables
    categorical_cols = ['actioncreatororganisation', 'actioncreatordepartment', 
                        'last_actioneditordepartment', 'last_actioneditororganisation']
    data = pd.get_dummies(data, columns=categorical_cols)

    # Define features and target
    X = data.drop(columns=['casenumber', 'escalated', '68', '64'])
    y = data['escalated']

    # Split data into training and test sets
    print("Splitting data into training and test sets")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Imputation of missing values
    print("Imputing missing values")
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Standardization of features
    print("Standardizing features")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X.columns

def calculate_class_weights(y_train):
    # Calculate class weights for the imbalance
    print("Calculating class weights")
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    return class_weight_dict

def train_knn(X_train, y_train, X_test, y_test):
    # k-Nearest Neighbors (kNN)
    print("Training kNN model")

    # Model and hyperparameters for kNN
    knn = KNeighborsClassifier()
    knn_params = {'n_neighbors': [3, 5, 7],
                  'weights': ['uniform', 'distance']}

    # Grid Search for kNN
    print("Hyperparameter tuning for kNN")
    knn_grid = GridSearchCV(knn, param_grid=knn_params, scoring='f1', cv=3)
    knn_grid.fit(X_train, y_train)

    # Display best hyperparameters
    print("Best Hyperparameters for kNN:", knn_grid.best_params_)

    # Save the best model
    best_knn = knn_grid.best_estimator_
    joblib.dump(best_knn, "kNN_model.pkl")

    # Predictions and results for kNN
    y_pred_knn = best_knn.predict(X_test)
    print("kNN Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_knn))
    print("kNN Classification Report:")
    print(classification_report(y_test, y_pred_knn))

def train_random_forest(X_train, y_train, X_test, y_test, class_weight_dict, feature_names):
    # Random Forest
    print("Training Random Forest model")

    # Model and hyperparameters for Random Forest
    rf = RandomForestClassifier(random_state=42, class_weight=class_weight_dict)
    rf_params = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}

    # Grid Search for Random Forest
    rf_grid = GridSearchCV(rf, param_grid=rf_params, scoring='f1', cv=3)
    rf_grid.fit(X_train, y_train)

    # Display best hyperparameters
    print("Best Hyperparameters for Random Forest:", rf_grid.best_params_)

    # Save the best model
    best_rf = rf_grid.best_estimator_
    joblib.dump(best_rf, "RandomForest_model.pkl")

    # Predictions and results for Random Forest
    y_pred_rf = best_rf.predict(X_test)
    print("Random Forest Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_rf))
    print("Random Forest Classification Report:")
    print(classification_report(y_test, y_pred_rf))

    # Feature Importances for Random Forest
    feature_importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': best_rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    print("\nFeature Importances for Random Forest:")
    print(feature_importances)

def main():
    # Load and prepare data
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data()

    # Calculate class weights for the imbalance
    class_weight_dict = calculate_class_weights(y_train)

    # k-Nearest Neighbors trainieren und bewerten
    #train_knn(X_train, y_train, X_test, y_test)

    # Train and evaluate Random Forest
    train_random_forest(X_train, y_train, X_test, y_test, class_weight_dict, feature_names)

if __name__ == "__main__":
    main()
