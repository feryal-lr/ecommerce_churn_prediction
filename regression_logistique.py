import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

print("1. Chargement du dataset clean...")
# Nous chargeons le fichier que tu as partagé
df = pd.read_csv('clean_dataset.csv')

# Séparation des caractéristiques (X) et de la cible (y)
X = df.drop(columns=['churn','recence'])
y = df['churn']

print("2. Division des données (80% Entraînement / 20% Test)...")
# On sépare les données de manière stratifiée pour garder l'équilibre 52/48 dans les deux blocs
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("3. Entraînement du modèle (Régression Logistique)...")
modele = LogisticRegression()
modele.fit(X_train, y_train) # Le modèle ajuste ses coefficients mathématiques ici

print("4. Évaluation du modèle sur le jeu de test caché...")
# Le modèle tente de deviner le churn pour les 20% de clients qu'il n'a jamais vus
predictions = modele.predict(X_test)

'''# Calcul des métriques de performance
precision_globale = accuracy_score(y_test, predictions)
print(f"\nPrécision globale du modèle (Accuracy) : {precision_globale * 100:.2f}%")

print("\nRapport de classification détaillé :")
print(classification_report(y_test, predictions))

print("Matrice de confusion :")
print(confusion_matrix(y_test, predictions))'''

# ==========================================
# SAUVEGARDE DU MODÈLE UNIQUEMENT
# ==========================================
print("3. Sérialisation et Sauvegarde du modèle...")
joblib.dump(modele, 'modele_logistic_jumia.joblib')

print("🎉 Succès ! Le fichier 'modele_logistic_jumia.joblib' a été créé !")




'''1. La Matrice de Confusion (Comprendre les erreurs)
La matrice te donne le décompte exact des prédictions sur tes 200 clients de test :

88 clients qui allaient rester ont été bien prédits (Vrais Négatifs).

89 clients qui allaient partir ont été bien détectés (Vrais Positifs).

7 clients fidèles ont été signalés à tort comme "sur le départ" (Faux Positifs).

16 clients qui ont réellement abandonné l'application ont été ratés par le modèle (Faux Négatifs).

2. Précision vs Recall (Le dilemme du Churn)(dans le rapport de classification)
Regardons la ligne 1 (les clients qui partent) :

La Précision (0.93 / 93%)
Ce que ça veut dire : Quand le modèle pointe du doigt un client en disant "Attention, il va partir", il a raison dans 93 % des cas.

Impact Business : C'est super pour Jumia. Si l'équipe marketing décide d'envoyer un bon de réduction de 500 DZD à ces clients pour les retenir, elle sait qu'elle ne va pas gaspiller d'argent sur des clients qui étaient de toute façon fidèles.

Le Recall / Rappel (0.85 / 85%)
Ce que ça veut dire : Sur l'ensemble des clients qui ont réellement quitté la plateforme, le modèle a réussi à en capturer 85 %. Les 15 % restants (nos 16 Faux Négatifs) sont partis sans qu'on ne les voie venir.

Impact Business : C'est le point à améliorer. En entreprise, rater un client qui part coûte souvent plus cher que de donner un coupon à tort.

3. Le F1-Score (0.89 / 89%)
C'est la moyenne harmonique entre la précision et le rappel. À 89 %, cela confirme que le modèle trouve un excellent équilibre entre "ne pas se tromper" et "capturer un maximum de départs".






'''