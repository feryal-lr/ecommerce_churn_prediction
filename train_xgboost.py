import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("1. Chargement du dataset clean...")
df = pd.read_csv('clean_dataset.csv')

# Séparation des caractéristiques (X) et de la cible (y)
X = df.drop(columns=['churn','recence'])
y = df['churn']

print("2. Division des données (80% Entraînement / 20% Test)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("3. Entraînement du modèle XGBoost...")
# On configure le modèle avec des paramètres standards pour débuter
modele_xgb = XGBClassifier(
    n_estimators=100,      # Nombre d'arbres de décision à construire
    learning_rate=0.1,     # Vitesse d'apprentissage
    max_depth=3,           # Profondeur maximale de chaque arbre
    random_state=42
)
modele_xgb.fit(X_train, y_train)

print("4. Évaluation du modèle XGBoost sur le jeu de test...")
predictions = modele_xgb.predict(X_test)

# Calcul des métriques
precision_globale = accuracy_score(y_test, predictions)
print(f"\nPrécision globale du modèle (Accuracy) : {precision_globale * 100:.2f}%")

print("\nRapport de classification détaillé :")
print(classification_report(y_test, predictions))

print("Matrice de confusion :")
print(confusion_matrix(y_test, predictions))

'''

Précision globale du modèle (Accuracy) : 100.00%

Rapport de classification détaillé :
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        95
           1       1.00      1.00      1.00       105

    accuracy                           1.00       200
   macro avg       1.00      1.00      1.00       200
weighted avg       1.00      1.00      1.00       200

Matrice de confusion :
[[ 95   0]
 [  0 105]]
 
 
 Pourquoi la régression logistique a fait 88 % et XGBoost fait 100 % ?
XGBoost est un modèle basé sur des arbres de décision extrêmement puissants, capables de détecter des règles strictes de type "vrai ou faux".

Souviens-toi de la règle que nous avons écrite en SQL pour définir notre cible (churn) dans le fichier creer_dataset.py :

Si la récence du client est supérieure à 30 jours, alors churn = 1, sinon churn = 0.

Ensuite, lors du prétraitement, nous avons appliqué le MinMaxScaler sur la variable recence. Qu'a fait XGBoost ? Il a scanné les données et a immédiatement trouvé la formule magique exacte. Il a découvert que toutes les lignes ayant une recence normalisée supérieure à un certain seuil précis valaient toujours churn = 1.

Il a simplement triché (sans le savoir) en utilisant une variable qui contient directement la réponse à la question ! Dans la vraie vie, le jour où tu veux prédire si un client va partir le mois prochain, tu ne connais pas encore sa récence future.

Comment corriger cela pour avoir un projet réaliste ?
Pour que ton projet Jumia soit crédible sur un CV et reflète la vraie vie, le modèle ne doit pas avoir accès à une variable qui donne une réponse directe comme la récence brute calculée sur la même période.

Nous avons deux solutions professionnelles :

Supprimer la colonne recence de l'entraînement : On force ainsi XGBoost à deviner le churn uniquement grâce à la fréquence, au montant_total, aux sessions, à la wilaya et à l'appareil. C'est un excellent exercice, car le modèle va devoir trouver des patterns plus subtils.

Précision globale du modèle (Accuracy) : 78.00%

Rapport de classification détaillé :
              precision    recall  f1-score   support

           0       0.72      0.88      0.79        95
           1       0.87      0.69      0.77       105

    accuracy                           0.78       200
   macro avg       0.79      0.78      0.78       200
weighted avg       0.80      0.78      0.78       200

Matrice de confusion :
[[84 11]
 [33 72]]
PS C:\User

 '''