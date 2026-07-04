import sqlite3
import pandas as pd

# 1. Connexion à notre base de données
connexion = sqlite3.connect('jumia_donnes.db')

# 2. Écriture de la grande requête SQL d'ingénierie des données
# On utilise la date du '2026-06-30' comme notre présent (Date de fin de simulation)
requete_features = """
SELECT 
    u.id_utilisateur,
    u.wilaya,
    u.appareil,
    
    -- 1. Calcul de la Récence (jours depuis la dernière commande)
    -- Si le client n'a jamais commandé, on met une valeur par défaut élevée (365 jours)
    COALESCE(JULIANDAY('2026-06-30') - JULIANDAY(MAX(c.date_commande)), 365) AS recence,
    
    -- 2. Calcul de la Fréquence (Nombre total de commandes livrées ou au total)
    COUNT(DISTINCT c.id_commande) AS frequence,
    
    -- 3. Calcul du Montant Total dépensé (en DZD)
    COALESCE(SUM(c.montant_total), 0) AS montant_total,
    
    -- 4. Calcul du nombre total de sessions (connexions)
    COUNT(DISTINCT s.id_session) AS total_sessions,
    
    -- 5. CRÉATION DE LA CIBLE (TARGET) : CHURN
    -- Si la dernière commande date de plus de 30 jours, Churn = 1, sinon 0
    CASE 
        WHEN COALESCE(JULIANDAY('2026-06-30') - JULIANDAY(MAX(c.date_commande)), 365) > 30 THEN 1
        ELSE 0
    END AS churn

FROM utilisateurs u
LEFT JOIN commandes c ON u.id_utilisateur = c.id_utilisateur
LEFT JOIN sessions s ON u.id_utilisateur = s.id_utilisateur
GROUP BY u.id_utilisateur;
"""

print("Exécution de la requête SQL d'ingénierie des caractéristiques...")
df_dataset = pd.read_sql_query(requete_features, connexion)

# 3. Sauvegarde de ce dataset propre dans un nouveau fichier CSV prêt pour le ML
df_dataset.to_csv('final_dataset.csv', index=False)

print("\n--- APERÇU DU DATASET PRÊT POUR LE MACHINE LEARNING ---")
print(df_dataset.head())

print(f"\nTaille du dataset généré : {df_dataset.shape[0]} lignes et {df_dataset.shape[1]} colonnes.")

connexion.close()
print("\nSuccès ! Le fichier 'dataset_pret_pour_ml.csv' a été créé.")