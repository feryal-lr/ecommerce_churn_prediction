import pandas as pd

print("--- CHARGEMENT DU DATASET TRAITÉ ---")
df_traite = pd.read_csv('clean_dataset.csv')

print("\n1. COHÉRENCE DE L'ÉQUILIBRE (TARGET CHURN)")
comptage_churn = df_traite['churn'].value_counts()
pourcentage_churn = df_traite['churn'].value_counts(normalize=True) * 100

for categorie, nombre in comptage_churn.items():
    statut = "Perdu (Churn)" if categorie == 1 else "Actif (Fidèle)"
    print(f"   - Clients {statut} : {nombre} ({pourcentage_churn[categorie]:.2f}%)")

print("\n2. COHÉRENCE DES CORRÉLATIONS (AVANT vs APRÈS)")
# On calcule la corrélation sur le nouveau dataset
matrice_correlation = df_traite.corr()

print("Nouveaux coefficients de corrélation par rapport au Churn :")
print(matrice_correlation['churn'].sort_values(ascending=False).head(5))

'''1. COHÉRENCE DE L'ÉQUILIBRE (TARGET CHURN)
   - Clients Perdu (Churn) : 523 (52.30%)
   - Clients Actif (Fidèle) : 477 (47.70%)

2. COHÉRENCE DES CORRÉLATIONS (AVANT vs APRÈS)
Nouveaux coefficients de corrélation par rapport au Churn :
churn           1.000000
recence         0.719409
appareil_iOS    0.045459
wilaya_Blida    0.035779
wilaya_Alger    0.015891'''