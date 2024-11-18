# Erstellt mit der Unterstützung von GitHub Copilot am 01.11.2024
# Programm to train the k-Nearest Neighbor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
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

    # Define features and target, drop columns 64 and 68 which is escalated feature
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

def train_knn(X_train, y_train, X_test, y_test):
    # Train kNN model with fixed hyperparameters
    print("Training kNN model with fixed hyperparameters")

    # Initialize model with fixed hyperparameters
    knn = KNeighborsClassifier(
        algorithm='auto',
        leaf_size=20,
        metric='minkowski',
        metric_params=None,
        n_jobs=None,
        n_neighbors=3,
        p=2,
        weights='distance'
    )

    # Fit model
    knn.fit(X_train, y_train)

    # Save kNN model
    joblib.dump(knn, "02_kNN_model.pkl")
    print("kNN model saved as 'kNN_model.pkl'")

    # Evaluate model
    y_pred_knn = knn.predict(X_test)
    print("kNN Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_knn))
    print("kNN Classification Report:")
    print(classification_report(y_test, y_pred_knn))

if __name__ == "__main__":
    # Load and prepare data
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data()
    
    train_knn(X_train, y_train, X_test, y_test)
