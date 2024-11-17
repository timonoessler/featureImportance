# This predictor can be used with every trained model in sis repo.
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def load_model():
    # Gespeichertes kNN-Modell laden
    knn_model = joblib.load("../SVM_model.pkl")
    return knn_model

def laod_data():
    # Laden der neuen Daten
    new_data = pd.read_csv("data/new_data.csv")
    return new_data

def preprocess_data(new_data):
    # Annahme: Die gleichen Vorverarbeitungsschritte wie beim Training
    categorical_cols = ['actioncreatororganisation', 'actioncreatordepartment', 
                        'last_actioneditordepartment', 'last_actioneditororganisation']
    new_data = pd.get_dummies(new_data, columns=categorical_cols)

    # Fehlende Werte auffüllen
    imputer = SimpleImputer(strategy='mean')
    new_data = imputer.fit_transform(new_data)

    # Standardisierung
    scaler = StandardScaler()
    new_data = scaler.fit_transform(new_data)

    return new_data

def predict_with(model, new_data):
    # Daten vorverarbeiten
    preprocessed_data = preprocess_data(new_data)

    # Vorhersage mit geladenem Modell
    predictions = model.predict(preprocessed_data)
    return predictions

# Beispiel für die Nutzung
if __name__ == "__main__":
    # Modell laden
    model = load_model()

    # Neue Daten laden
    new_data = laod_data()
    
    # Vorhersage durchführen
    predictions = predict_with(model, new_data)

    # Zeilenweise Vorhersagen anzeigen
    for i, prediction in enumerate(predictions, 1):
        print(f"Vorhersage für Zeile {i}: {prediction}")
