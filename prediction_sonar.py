import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Chargement des données
# On utilise header=None car le CSV n'a pas de noms de colonnes
df = pd.read_csv('sonar.all-data.csv', header=None)

# 2. Séparation des caractéristiques (X) et de la cible (y)
X = df.iloc[:, 0:60].values  # Les 60 colonnes de fréquences
y = df.iloc[:, 60].values    # La dernière colonne (M ou R)

# 3. Encodage des étiquettes (M/R -> 0/1)
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# 4. Division en données d'entraînement (80%) et de test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. ho any le entrainnement model iny
model = RandomForestClassifier(n_estimators=100, random_state=42)
print("Entraînement en cours...")
model.fit(X_train, y_train)

# 6. Évaluation
predictions = model.predict(X_test)
score = accuracy_score(y_test, predictions)

print("-" * 30)
print(f"Précision du modèle : {score * 100:.2f} %")
print("-" * 30)
print("Rapport détaillé :")
print(classification_report(y_test, predictions, target_names=encoder.classes_))