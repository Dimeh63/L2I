import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob
import tweepy
from typing import List
from sklearn.pipeline import make_pipeline
import joblib

# Configuration de l'API Twitter (remplacer par vos propres clés)
API_KEY = os.getenv('TWITTER_API_KEY', 'votre_clé_api')
API_SECRET = os.getenv('TWITTER_API_SECRET', 'votre_secret_api')
ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN', 'votre_token_accès')
ACCESS_TOKEN_SECRET = os.getenv('TWITTER_ACCESS_TOKEN_SECRET', 'votre_secret_token_accès')

# Initialisation de l'API Twitter
def init_twitter_api() -> tweepy.API:
    """
    Authentifie à l'API Twitter et retourne une instance de l'API.
    """
    auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    return api

# Obtenir les tendances Twitter en temps réel
def obtenir_tendances(api: tweepy.API) -> List[str]:
    """
    Récupère les tendances Twitter mondiales et retourne une liste de noms de tendances.
    """
    tendances = api.get_place_trends(id=1)  # id=1 pour le monde entier
    noms_tendances = [tendance['name'] for tendance in tendances[0]['trends']]
    return noms_tendances

# Détecter les campagnes de désinformation
def detecter_campagnes_desinformation(tendances: List[str], clf: make_pipeline) -> List[str]:
    """
    Détecte les campagnes de désinformation dans les tendances fournies et retourne une liste de tendances suspectes.
    """
    campagnes_desinformation = []
    for tendance in tendances:
        predicted = clf.predict([tendance])
        sentiment = TextBlob(tendance).sentiment
        
        if predicted[0] == 1 and abs(sentiment.polarity) > 0.5:
            campagnes_desinformation.append(tendance)
    
    return campagnes_desinformation

if __name__ == "__main__":
    api = init_twitter_api()
    tendances = obtenir_tendances(api)

    # Simule le chargement d'un modèle et d'un vectoriseur pré-entraînés
    # Dans un scénario réel, remplacez cela par le chargement de votre modèle/vectoriseur sauvegardé
    # Exemple : clf = joblib.load('chemin_vers_votre_model_et_vectoriseur.pkl')
    clf = make_pipeline(TfidfVectorizer(stop_words='english'), MultinomialNB())

    # Supposons ici que clf est déjà entraîné avec des données pertinentes à votre cas d'usage
    campagnes_desinformation = detecter_campagnes_desinformation(tendances, clf)
    if campagnes_desinformation:
        for campagne in campagnes_desinformation:
            print(f"Campagne de désinformation détectée dans la tendance : {campagne}")
    else:
        print("Aucune campagne de désinformation détectée.")
