# Benötigte Bibliotheken importieren
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd
import shap

# Daten laden
data = pd.read_csv("data/very_small_df.csv")

# Zielspalte extrahieren und kategorische Variablen kodieren
label_encoder = LabelEncoder()
data['escalated'] = label_encoder.fit_transform(data['escalated'])

# Kategorische Variablen in numerische Variablen umwandeln
categorical_cols = ['actioncreatororganisation', 'actioncreatordepartment', 
                    'last_actioneditordepartment', 'last_actioneditororganisation']
data = pd.get_dummies(data, columns=categorical_cols)

# Features und Ziel festlegen
X = data.drop(columns=['casenumber', 'escalated', '68', '64', '66', '400','401',
                           '402','403','404','405','406','407','408','409','410','411','412'])
y = data['escalated']

# Daten in Trainings- und Testsets aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Imputation der fehlenden Werte
print("Imputing missing values")
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Standardisierung der Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Berechnung der Klassen-Gewichte für das Ungleichgewicht
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# MLPClassifier erstellen und trainieren
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)
mlp.fit(X_train, y_train)

# Modell evaluieren
y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# SHAP Feature Importance mit predict anstatt predict_proba
explainer = shap.KernelExplainer(mlp.predict, X_train[:100])
shap_values = explainer.shap_values(X_test[:100])

# Berechnen der mittleren absoluten SHAP-Werte für alle Features
feature_importance = np.mean(np.abs(shap_values), axis=0)
features = X.columns

# Anzeige der Feature-Importance
feature_importance_text = "\n".join([f"{feature}: {importance}" for feature, importance in zip(features, feature_importance)])
print("Feature Importance:")
print(feature_importance_text)

# Speichern als Textdatei
with open("feature_importance.txt", "w") as file:
    file.write(feature_importance_text)
