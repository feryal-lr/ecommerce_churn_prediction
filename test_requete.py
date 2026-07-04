import sqlite3
import pandas as pd

# 1. Connexion à la base de données que nous avons créée à l'étape précédente
connexion = sqlite3.connect('jumia_donnes.db')

# 2. Écriture de notre première requête SQL en texte
# Cette requête dit : "Sélectionne toutes les colonnes de la table utilisateurs, mais limite le résultat à 5 lignes"
requete_sql = "SELECT * FROM utilisateurs LIMIT 5;"

# 3. Utilisation de Pandas pour exécuter la requête et mettre le résultat au propre
print("Exécution de la requête SQL...")
resultat = pd.read_sql_query(requete_sql, connexion)

# 4. Affichage du résultat dans le terminal
print("\n--- LES 5 PREMIERS UTILISATEURS EN BASE DE DONNÉES ---")
print(resultat)

# 5. Fermeture de la connexion
connexion.close()