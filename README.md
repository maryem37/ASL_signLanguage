Aperçu du projet

Le projet se compose de quatre étapes principales :

Collecte des images :

Capturer des images via webcam, classées par signe (A-Z et 0-9).

Création du dataset :

Détection des landmarks de la main avec MediaPipe Hands.

Extraction des coordonnées (x, y) → vecteurs de features pour chaque image.

Entraînement du modèle :

Un Random Forest Classifier est entraîné sur les vecteurs de landmarks.

Le modèle apprend à associer chaque configuration de main à un signe ASL.

Reconnaissance en temps réel :

La webcam capture la vidéo, MediaPipe détecte les landmarks, et le modèle prédit le signe en temps réel.

Les prédictions sont affichées avec une bounding box autour de la main.

Description de model.p

model.p est le fichier sauvegardé du modèle Random Forest.

Il contient :

L’objet RandomForestClassifier entraîné sur les landmarks extraits.

Les informations nécessaires pour effectuer la prédiction sur de nouvelles images en temps réel.

Format : Pickle (.p) → permet de charger le modèle directement sans réentraîner à chaque fois.

Côté code :

import pickle

# Charger le modèle entraîné
with open('model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

# Utilisation pour prédiction
prediction = model.predict(nouveaux_landmarks)


En résumé, model.p est le cœur du système de reconnaissance : il contient le Random Forest déjà entraîné, prêt à prédire les signes ASL en temps réel.

Utilisation

Collecte des images → collect_images.py

Création du dataset → create_dataset.py

Entraînement du modèle → train_classifier.py

Reconnaissance en temps réel → inference_classifier.py

Environnement et requirements

Python 3.x

OpenCV

MediaPipe

scikit-learn

NumPy & Pickle

Développement testé localement, Jupyter Notebook utilisé pour expérimenter et charger certains modèles.
