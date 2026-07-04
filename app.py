from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# =====================================================================
# ZONE 1 : INITIALISATION DE L'API
# =====================================================================
# Ici, on crée l'objet principal de notre application web.
app = FastAPI(
    title="Jumia Churn API",
    description="Interface de production pour prédire l'attrition des clients - Jumia Algérie",
    version="1.0"
)

# =====================================================================
# ZONE 2 : CHARGEMENT DES ARTIFACTS SÉRIALISÉS
# =====================================================================
# Au démarrage du serveur, on charge une bonne fois pour toutes nos fichiers .joblib
modele = joblib.load('modele_logistic_jumia.joblib')
scaler = joblib.load('scaler_jumia.joblib')

# =====================================================================
# ZONE 3 : LE SCHEMA DE DONNÉES (PYDANTIC)
# =====================================================================
# On définit exactement à quoi doivent ressembler les données d'un client.
# C'est la barrière de sécurité de notre modèle.
class SchemaClient(BaseModel):
    frequence: float
    montant_total: float
    total_sessions: float
    wilaya_Alger: int = 0
    wilaya_Bejaea: int = 0
    wilaya_Blida: int = 0
    wilaya_Constantine: int = 0
    wilaya_Ghardaia: int = 0
    wilaya_Oran: int = 0
    wilaya_Setif: int = 0
    wilaya_Tlemcen: int = 0
    appareil_Android: int = 0
    appareil_Web: int = 0
    appareil_iOS: int = 0

# =====================================================================
# ZONE 4 : LES ROUTES (ENDPOINTS)
# =====================================================================

# Route 1 : Accueil (Pour vérifier que l'API tourne)
@app.get("/")
def accueil():
    return {"status": "Online", "message": "API Jumia Churn opérationnelle. Rendez-vous sur /docs"}

# Route 2 : La prédiction (Méthode POST car on reçoit des données)
@app.post("/predict")
def predire_churn(client: SchemaClient):
    donnees_dict = client.model_dump()
    df_client = pd.DataFrame([donnees_dict])
    
    # 1. On recrée temporairement les variables numériques dans l'ordre exact du prétraitement
    # On ajoute 'recence' à 0 pour satisfaire le scaler
    df_client['recence'] = 0.0
    
    colonnes_num = ['recence', 'frequence', 'montant_total', 'total_sessions']
    
    # 2. On applique le scaler UNIQUEMENT sur ces colonnes numériques
    # (Si ton scaler du prétraitement n'avait été entraîné que sur les données numériques)
    try:
        df_client[colonnes_num] = scaler.transform(df_client[colonnes_num])
    except ValueError:
        # Si ton scaler avait été entraîné sur TOUTES les colonnes (y compris Wilayas/Appareils),
        # on lui donne le DataFrame complet avec 'recence' insérée au bon endroit.
        # Pour être sûr, on va juste s'assurer que 'recence' est présente :
        df_client = df_client[['recence'] + [col for col in df_client.columns if col != 'recence']]
        df_client = pd.DataFrame(scaler.transform(df_client), columns=df_client.columns)

    # 3. On supprime définitivement 'recence' avant de donner les données au modèle de classification
    df_client = df_client.drop(columns=['recence'])
    
    # 4. Prédiction
    prediction = int(modele.predict(df_client)[0])
    probabilite = float(modele.predict_proba(df_client)[0][1])
    
    return {
        "churn_prediction": prediction,
        "probability": round(probabilite, 4),
        "decision_business": "⚠️ ALERTE : Risque de départ élevé !" if prediction == 1 else "✅ Client fidèle et stable."
    }

'''Explications détaillées des composants :
client.model_dump() : C'est la fonction qui transforme instantanément la requête reçue du web en un dictionnaire Python standard.

modele.predict_proba() : Au lieu de donner juste un verdict sec (0 ou 1), cette fonction magique nous renvoie la probabilité exacte (ex: 0.842 soit 84.2 % de chances de partir). C'est ce pourcentage qui est précieux pour l'équipe business pour prioriser les clients les plus en danger.


APRES EXEC DE LA CMD uvicorn app:app --reload I got this :
INFO:     Will watch for changes in these directories: ['C:\\Users\\ferya\\OneDrive\\Desktop\\AI projects\\jumia_churn_project']
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [10324] using StatReload
INFO:     Started server process [14900]
INFO:     Waiting for application startup.
INFO:     Application startup complete.


VERSION INITIALE DE PREDICT AVANT DAJOUTER RECENCE :

@app.post("/predict")
def predire_churn(client: SchemaClient):
    # a. Transformer les données JSON reçues en dictionnaire puis en DataFrame Pandas
    donnees_dict = client.model_dump()
    df_client = pd.DataFrame([donnees_dict])
    
    # b. Normaliser uniquement les colonnes numériques avec notre scaler chargé
    colonnes_num = ['frequence', 'montant_total', 'total_sessions']
    df_client[colonnes_num] = scaler.transform(df_client[colonnes_num])
    
    # c. Soumettre les données traitées au modèle mathématique
    prediction = int(modele.predict(df_client)[0])
    probabilite = float(modele.predict_proba(df_client)[0][1])
    
    # d. Renvoyer le diagnostic au format JSON
    return {
        "churn_prediction": prediction,
        "probability": round(probabilite, 4),
        "decision_business": "⚠️ ALERTE : Risque de départ élevé !" if prediction == 1 else "✅ Client fidèle et stable."
    }

LA VERSION APRES AVOIR AJOUTE RECENCE :

@app.post("/predict")
def predire_churn(client: SchemaClient):
    donnees_dict = client.model_dump()
    df_client = pd.DataFrame([donnees_dict])
    
    # 1. On recrée temporairement les variables numériques dans l'ordre exact du prétraitement
    # On ajoute 'recence' à 0 pour satisfaire le scaler
    df_client['recence'] = 0.0
    
    colonnes_num = ['recence', 'frequence', 'montant_total', 'total_sessions']
    
    # 2. On applique le scaler UNIQUEMENT sur ces colonnes numériques
    # (Si ton scaler du prétraitement n'avait été entraîné que sur les données numériques)
    try:
        df_client[colonnes_num] = scaler.transform(df_client[colonnes_num])
    except ValueError:
        # Si ton scaler avait été entraîné sur TOUTES les colonnes (y compris Wilayas/Appareils),
        # on lui donne le DataFrame complet avec 'recence' insérée au bon endroit.
        # Pour être sûr, on va juste s'assurer que 'recence' est présente :
        df_client = df_client[['recence'] + [col for col in df_client.columns if col != 'recence']]
        df_client = pd.DataFrame(scaler.transform(df_client), columns=df_client.columns)

    # 3. On supprime définitivement 'recence' avant de donner les données au modèle de classification
    df_client = df_client.drop(columns=['recence'])
    
    # 4. Prédiction
    prediction = int(modele.predict(df_client)[0])
    probabilite = float(modele.predict_proba(df_client)[0][1])
    
    return {
        "churn_prediction": prediction,
        "probability": round(probabilite, 4),
        "decision_business": "⚠️ ALERTE : Risque de départ élevé !" if prediction == 1 else "✅ Client fidèle et stable."
    }
''' 