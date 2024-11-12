import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import numpy as np
import joblib

def load_and_prepare_data():
    # Load and prepare data
    print("Loading and preparing data")
    data = pd.read_csv("data/very_small_df.csv")

    # Extract target column and encode categorical variables
    label_encoder = LabelEncoder()
    data['escalated'] = label_encoder.fit_transform(data['escalated'])

    # Convert categorical variables into numerical variables
    categorical_cols = ['actioncreatororganisation', 'actioncreatordepartment', 
                        'last_actioneditordepartment', 'last_actioneditororganisation']
    data = pd.get_dummies(data, columns=categorical_cols)

    # Define features and target
    X = data.drop(columns=['casenumber', 'escalated', '68', '64', '66', '400','401'
                           ,'402','403','404','405','406','407','408','409','410','411','412'])
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

def save_feature_importances_to_file(feature_importances, output_path="feature_importances.txt"):
    # Feature-Importances in eine Textdatei schreiben
    with open(output_path, "w") as file:
        for _, row in feature_importances.iterrows():
            file.write(f"{row['Feature']}: {row['Importance']}\n")
    print(f"Feature importances saved to {output_path}")

def train_random_forest(X_train, y_train, X_test, y_test, feature_names):
    # Train Random Forest model with fixed hyperparameters
    print("Training Random Forest model with fixed hyperparameters")

    # Initializing model with fixed hyperparameters
    rf = RandomForestClassifier(
        bootstrap=True,
        ccp_alpha=0.0,
        class_weight={0: 0.5355892970321585, 1: 7.524583817266744},
        criterion='gini',
        max_depth=None,
        max_features='sqrt',
        max_leaf_nodes=None,
        max_samples=None,
        min_impurity_decrease=0.0,
        min_samples_leaf=1,
        min_samples_split=2,
        min_weight_fraction_leaf=0.0,
        n_estimators=100,
        n_jobs=None,
        oob_score=False,
        random_state=42,
        verbose=0,
        warm_start=False
    )

    # Fit model
    rf.fit(X_train, y_train)

    # Save Random Forest model
    joblib.dump(rf, "RandomForest_model.pkl")
    print("Random Forest model saved as 'RandomForest_model.pkl'")

    # Predictions and results for Random Forest
    y_pred_rf = rf.predict(X_test)
    print("Random Forest Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_rf))
    print("Random Forest Classification Report:")
    print(classification_report(y_test, y_pred_rf))

    # Feature Importances für Random Forest
    feature_importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    print("\nFeature Importances for Random Forest:")
    print(feature_importances)
    
    # Save feature importances to a text file
    save_feature_importances_to_file(feature_importances)

def main():
    # Load and prepare data
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data()

    # Train and evaluate Random Forest
    train_random_forest(X_train, y_train, X_test, y_test, feature_names)

if __name__ == "__main__":
    main()
