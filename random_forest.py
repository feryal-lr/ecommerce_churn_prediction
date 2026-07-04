import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("1. Chargement du dataset clean...")
df = pd.read_csv('clean_dataset.csv')

# Séparation des caractéristiques (X) et de la cible (y)
# On reste sur notre configuration équitable sans la récence !
X = df.drop(columns=['churn', 'recence'])
y = df['churn']

print("2. Division des données (80% Entraînement / 20% Test)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("3. Entraînement du modèle Random Forest...")
modele_rf = RandomForestClassifier(
    n_estimators=100,     # Nombre d'arbres dans la forêt
    max_depth=5,          # On limite la profondeur pour éviter le surapprentissage
    random_state=42
)
modele_rf.fit(X_train, y_train)

print("4. Évaluation du modèle sur le jeu de test...")
predictions = modele_rf.predict(X_test)

# Calcul des performances
precision_globale = accuracy_score(y_test, predictions)
print(f"\nPrécision globale du modèle (Accuracy) : {precision_globale * 100:.2f}%")

print("\nRapport de classification détaillé :")
print(classification_report(y_test, predictions))

print("Matrice de confusion :")
print(confusion_matrix(y_test, predictions))


'''Précision globale du modèle (Accuracy) : 77.00%

Rapport de classification détaillé :
              precision    recall  f1-score   support

           0       0.70      0.91      0.79        95
           1       0.88      0.65      0.75       105

    accuracy                           0.77       200
   macro avg       0.79      0.78      0.77       200
weighted avg       0.80      0.77      0.77       200

Matrice de confusion :
[[86  9]
 [37 68]]'''