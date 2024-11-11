# Quelle: https://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/
# Benötigte Bibliotheken importieren
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd

# Daten laden
data = pd.read_csv("data/merged.csv")

# Zielspalte extrahieren und kategorische Variablen kodieren
label_encoder = LabelEncoder()
data['escalated'] = label_encoder.fit_transform(data['escalated'])

# Kategorische Variablen in numerische Variablen umwandeln
categorical_cols = ['actioncreatororganisation', 'actioncreatordepartment', 
                    'last_actioneditordepartment', 'last_actioneditororganisation']
data = pd.get_dummies(data, columns=categorical_cols)

# Features und Ziel festlegen
X = data.drop(columns=['casenumber', 'escalated', '68', '64', '66', '400','401'
                           ,'402','403','404','405','406','407','408','409','410','411','412'])
y = data['escalated']

# Daten in Trainings- und Testsets aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Imputation of missing values
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

# Funktion zur Erstellung des Modells
def create_model(optimizer='adam', init='glorot_uniform'):
    model = Sequential()
    model.add(Dense(64, input_shape=(X_train.shape[1],), kernel_initializer=init, activation='relu'))
    model.add(Dense(32, kernel_initializer=init, activation='relu'))
    model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# KerasClassifier mit dem Modell-Wrapper
model = KerasClassifier(model=create_model, verbose=0)

# Grid Search-Parameter definieren
optimizers = ['adam', 'rmsprop']
inits = ['glorot_uniform', 'normal', 'uniform']
epochs = [50, 100]
batches = [10, 20]

# Parameter für GridSearchCV
param_grid = dict(
    model__optimizer=optimizers,
    model__init=inits,
    epochs=epochs,
    batch_size=batches
)

# Grid Search einrichten
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='f1', cv=3)
grid_result = grid.fit(X_train, y_train, **{'class_weight': class_weight_dict})

# Ergebnisse anzeigen
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
