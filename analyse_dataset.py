import pandas as pd

print("--- CHARGEMENT DU DATASET ---")
df = pd.read_csv('final_dataset.csv')

print("\n1. VÉRIFICATION DE L'ÉQUILIBRE DES DONNÉES (TARGET CHURN)")
# On compte le nombre de clients dans chaque catégorie
comptage_churn = df['churn'].value_counts()
pourcentage_churn = df['churn'].value_counts(normalize=True) * 100

for categorie, nombre in comptage_churn.items():
    statut = "Perdu (Churn)" if categorie == 1 else "Actif (Fidèle)"
    print(f"   - Clients {statut} : {nombre} ({pourcentage_churn[categorie]:.2f}%)")

print("\n2. ANALYSE DES CORRÉLATIONS NUMÉRIQUES")
# On sélectionne uniquement les colonnes numériques pour calculer la corrélation
colonnes_numeriques = ['recence', 'frequence', 'montant_total', 'total_sessions', 'churn']
matrice_correlation = df[colonnes_numeriques].corr()

# On regarde spécifiquement comment chaque variable est liée à la colonne 'churn'
print("Coefficient de corrélation par rapport au Churn :")
print(matrice_correlation['churn'].sort_values(ascending=False))

'''
1. VÉRIFICATION DE L'ÉQUILIBRE DES DONNÉES (TARGET CHURN)
   - Clients Perdu (Churn) : 523 (52.30%)
   - Clients Actif (Fidèle) : 477 (47.70%)

2. ANALYSE DES CORRÉLATIONS NUMÉRIQUES
Coefficient de corrélation par rapport au Churn :
churn             1.000000
recence           0.719409
total_sessions   -0.169836
montant_total    -0.406184
frequence        -0.690966
Name: churn, dtype: float64'''