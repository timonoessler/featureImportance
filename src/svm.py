# Programm to train the Support Vector Machine
# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import SimpleImputer
import numpy as np
import joblib

def load_and_prepare_data():
    # Load and prepare data
    print("Loading and preparing data: small_merged.csv")
    data = pd.read_csv("data/small_merged.csv")

    # Encode target column and transform categorical variables to numerical
    label_encoder = LabelEncoder()
    data['escalated'] = label_encoder.fit_transform(data['escalated'])

    # Convert categorical columns to numerical variables
    categorical_cols = ['actioncreatororganisation', 'actioncreatordepartment', 
                        'last_actioneditordepartment', 'last_actioneditororganisation']
    data = pd.get_dummies(data, columns=categorical_cols)

    # Define features and target variable
    X = data.drop(columns=['casenumber', 'escalated', '68', '64', '66', '400','401',
                           '402','403','404','405','406','407','408','409','410','411','412'])
    y = data['escalated']

    # Split data into training and test sets
    print("Splitting data into training and test sets")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Impute missing values
    print("Imputing missing values")
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Standardize features
    print("Standardizing features")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X.columns

def calculate_class_weights(y_train):
    # Calculate class weights for imbalanced data
    print("Calculating class weights")
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    return class_weight_dict

def save_feature_importances_to_file(feature_importances, output_path="SVM_feature_importances.txt"):
    # Save feature importances to a text file
    with open(output_path, "w") as file:
        for _, row in feature_importances.iterrows():
            file.write(f"{row['Feature']}: {row['Importance']}\n")
    print(f"Feature importances saved to {output_path}")

def train_svm(X_train, y_train, X_test, y_test, class_weight_dict, feature_names):
    # Support Vector Machine (SVM) Model
    print("Training SVM model")

    # Initialize SVM model with linear kernel and fixed hyperparameters
    svm = SVC(kernel='linear', class_weight=class_weight_dict)

    # Train the model
    svm.fit(X_train, y_train)

    # Save the model
    joblib.dump(svm, "SVM_model.pkl")
    print("SVM model saved as 'SVM_model.pkl'")

    # Predictions and results for SVM
    y_pred_svm = svm.predict(X_test)
    print("SVM Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_svm))
    print("SVM Classification Report:")
    print(classification_report(y_test, y_pred_svm))

    # Feature Importances using coefficients for linear SVM
    feature_importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': np.abs(svm.coef_[0])
    }).sort_values(by='Importance', ascending=False)

    print("\nFeature Importances for SVM:")
    print(feature_importances)

    # Save Feature Importances to a file
    save_feature_importances_to_file(feature_importances)

def main():
    # Load and prepare data
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data()

    # Calculate class weights for imbalanced data
    class_weight_dict = calculate_class_weights(y_train)

    # Train and evaluate SVM
    train_svm(X_train, y_train, X_test, y_test, class_weight_dict, feature_names)

if __name__ == "__main__":
    main()
