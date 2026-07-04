import sqlite3
import pandas as pd

print("Connexion à la base de données SQLite (création si elle n'existe pas)...")
# Cela crée un fichier 'jumia_donnes.db' qui va contenir nos tables SQL
connexion = sqlite3.connect('jumia_donnes.db')

print("Chargement des fichiers CSV dans les tables SQL...")
# On lit les CSV avec pandas
df_utilisateurs = pd.read_csv('utilisateurs.csv')
df_commandes = pd.read_csv('commandes.csv')
df_sessions = pd.read_csv('sessions.csv')

# On les pousse dans la base de données SQL sous forme de tables
df_utilisateurs.to_sql('utilisateurs', connexion, if_exists='replace', index=False)
df_commandes.to_sql('commandes', connexion, if_exists='replace', index=False)
df_sessions.to_sql('sessions', connexion, if_exists='replace', index=False)

print("Vérification des tables créées...")
curseur = connexion.cursor()
curseur.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = curseur.fetchall()

print("\nTables présentes dans votre base de données SQL :")
for table in tables:
    print(f"- {table[0]}")

connexion.close()
print("\nInstallation réussie ! Votre base de données SQL locale est prête.")