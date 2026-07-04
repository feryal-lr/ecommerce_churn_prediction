import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

print("1. Chargement du dataset...")
df = pd.read_csv('final_dataset.csv')

# On met de côté l'id_utilisateur (non prédictif) et la cible (churn) pour ne pas les normaliser
id_utilisateurs = df['id_utilisateur']
cible = df['churn']
X_brut = df.drop(columns=['id_utilisateur', 'churn'])

print("2. Encodage des variables catégorielles (wilaya, appareil)...")
# get_dummies transforme le texte en colonnes de 0 et 1
X_encode = pd.get_dummies(X_brut, columns=['wilaya', 'appareil'], dtype=int)

print("3. Normalisation des variables numériques...")
# On identifie les colonnes numériques à ramener entre 0 et 1
colonnes_a_normaliser = ['recence', 'frequence', 'montant_total', 'total_sessions']

scaler = MinMaxScaler()
# On applique la normalisation uniquement sur ces colonnes
X_encode[colonnes_a_normaliser] = scaler.fit_transform(X_encode[colonnes_a_normaliser])

print("4. Recomposition du dataset final...")
# On réassemble le tout dans un seul DataFrame propre
df_final = X_encode.copy()
df_final['churn'] = cible

#SERIALISATION DU SCALER 

joblib.dump(scaler, 'scaler_jumia.joblib')
print("✅ Le scaler officiel du prétraitement a été sauvegardé avec succès !")

'''
# Sauvegarde du fichier prêt pour l'entraînement
df_final.to_csv('clean_dataset.csv', index=False)

print("\nAperçu du dataset final normalisé et encodé :")
print(df_final.head(3))

print("\nDescription statistique des colonnes normalisées (min=0, max=1) :")
print(df_final[colonnes_a_normaliser].describe().loc[['min', 'max']])'''