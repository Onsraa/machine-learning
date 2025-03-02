# Machine Learning - Projet

Un framework complet d'apprentissage automatique en Rust avec une interface utilisateur graphique basée sur Bevy. Ce projet a été développé dans le cadre du cours "Machine Learning" et permet de tester différents algorithmes d'apprentissage sur des cas d'études simples ainsi que sur des tâches de classification d'images.

## Table des matières

- [Présentation du projet](#présentation-du-projet)
- [Fonctionnalités](#fonctionnalités)
- [Algorithmes implémentés](#algorithmes-implémentés)
- [Structure du projet](#structure-du-projet)
- [Prérequis et installation](#prérequis-et-installation)
- [Guide d'utilisation](#guide-dutilisation)
  - [Cas de tests](#cas-de-tests)
  - [Classification d'images de jeux](#classification-dimages-de-jeux)
- [Développement et contribution](#développement-et-contribution)

## Présentation du projet

Ce projet implémente une bibliothèque d'algorithmes de machine learning en Rust, avec une interface graphique permettant de visualiser les résultats et d'interagir avec les modèles. Il offre deux fonctionnalités principales :

1. **Cas de tests** : Plusieurs jeux de données synthétiques pour comparer et évaluer différents algorithmes de machine learning.
2. **Classification d'images de jeux** : Un système d'apprentissage pour classifier des images de jeux vidéo.

Conformément au syllabus du cours, le projet permet d'implémenter, de comparer et d'analyser les performances des modèles sur des problèmes de classification et de régression.

## Fonctionnalités

- Interface graphique interactive basée sur Bevy et egui
- Visualisation 3D des données et des frontières de décision
- Entraînement en temps réel avec retour visuel
- Graphiques de progression de l'apprentissage
- Sauvegarde et chargement de modèles entraînés
- Classification et régression sur divers jeux de données synthétiques
- Classification d'images de jeux vidéo

## Algorithmes implémentés

- **Modèles linéaires**
  - Régression linéaire
  - Classification linéaire

- **Perceptron Multi-Couches (MLP)**
  - Fonctions d'activation : Tanh, ReLU, Sigmoïde, Linéaire
  - Support pour plusieurs couches cachées
  - Algorithme de rétropropagation du gradient

- **Réseaux à Fonctions de Base Radiale (RBF)**
  - Sélection de centres par K-means
  - Ajustement automatique des paramètres
  - Support pour la classification et la régression

- **Machines à Vecteurs de Support (SVM)**
  - Algorithme SMO (Sequential Minimal Optimization)
  - Noyaux : linéaire, polynomial, RBF
  - Classification binaire

## Structure du projet

```
machine-learning/
├── src/
│   ├── algorithms/         # Implémentations des algorithmes de ML
│   │   ├── learning_model.rs   # Trait commun pour tous les modèles
│   │   ├── linear_classifier.rs
│   │   ├── linear_regression.rs
│   │   ├── mlp.rs          # Perceptron multicouches
│   │   ├── rbf.rs          # Réseaux à fonctions de base radiale
│   │   └── svm.rs          # Machines à vecteurs de support
│   ├── data/               # Gestion des données et jeux de tests
│   ├── ui/                 # Interface utilisateur avec egui
│   ├── systems/            # Systèmes Bevy
│   ├── plugins/            # Plugins Bevy
│   ├── resources/          # Ressources globales de l'application
│   └── main.rs             # Point d'entrée de l'application
├── Cargo.toml              # Dépendances et configuration du projet
└── README.md               # Ce fichier
```

## Prérequis et installation

### Prérequis

- [Rust](https://www.rust-lang.org/tools/install) (version 1.75.0 ou supérieure)
- Dépendances de développement pour Bevy:
  - Sur Linux: `sudo apt install libudev-dev libasound2-dev`
  - Sur macOS: Xcode et ses outils de ligne de commande
  - Sur Windows: Pas de dépendances supplémentaires

### Installation

1. Clonez ce dépôt :
   ```bash
   git clone https://github.com/votre-nom/machine-learning.git
   cd machine-learning
   ```

2. Construisez le projet :
   ```bash
   cargo build --release
   ```

3. Exécutez l'application :
   ```bash
   cargo run --release
   ```

## Guide d'utilisation

### Menu principal

<img width="562" alt="Capture d’écran 2025-03-02 à 11 57 22" src="https://github.com/user-attachments/assets/bc639f90-ed0f-43f7-a904-d66793063c21" />

Le menu principal offre deux options:
- **Cas de Tests** : Pour explorer et comparer différents algorithmes sur des jeux de données synthétiques
- **Classification de Jeux** : Pour entraîner et tester un modèle de classification d'images de jeux vidéo

### Cas de tests

<img width="1512" alt="Capture d’écran 2025-03-02 à 12 15 38" src="https://github.com/user-attachments/assets/5fe75f48-b64c-4899-b0ab-77da058fd333" />

Cette section vous permet de visualiser et d'expérimenter avec différents algorithmes sur des jeux de données prédéfinis.

#### Sélection du jeu de données

1. Utilisez la fenêtre "Test Case Selector" pour choisir un jeu de données:
   - Linear Simple: Classification binaire simple
   - Linear Multiple: Classification binaire avec plus de points
   - XOR: Problème XOR (non linéairement séparable)
   - Cross: Problème en forme de croix
   - Multi Linear 3 Classes: Classification à 3 classes
   - Linear Simple 2d/3d: Régression linéaire en 2D/3D
   - Non Linear Simple 2d/3d: Régression non linéaire

#### Sélection du modèle

1. Utilisez la fenêtre "Model Selector" pour choisir un algorithme:
   - Linear Classifier/Regression: Pour les problèmes linéairement séparables
   - MLP: Pour les problèmes plus complexes, avec configuration des couches cachées
   - RBF: Alternative aux MLP, particulièrement efficace pour certains types de données
   - SVM: Pour la classification binaire avec marge maximale

#### Configuration du modèle

Chaque type de modèle dispose de sa propre fenêtre de configuration:
- MLP Configuration: Nombre de couches, nombre de neurones, fonction d'activation
- RBF Configuration: Nombre de centres, gamma, utilisation de K-means
- SVM Configuration: Type de noyau, paramètres C et gamma

#### Entraînement

1. Dans la fenêtre "Training Control", ajustez les hyperparamètres:
   - Learning Rate: Taux d'apprentissage 
   - Train Ratio: Proportion des données utilisées pour l'entraînement
   - Batch Size: Taille des lots pour l'entraînement

2. Cliquez sur "Start Training" pour lancer l'entraînement
   - Observez la progression dans le graphique d'apprentissage
   - Les métriques d'entraînement et de test sont affichées

3. Cliquez sur "Stop Training" pour arrêter l'entraînement à tout moment

#### Gestion des modèles

<img width="1512" alt="Capture d’écran 2025-03-02 à 12 17 03" src="https://github.com/user-attachments/assets/bcbe5ee7-b96d-4da9-9fbc-adba70689270" />
<img width="624" alt="Capture d’écran 2025-03-02 à 12 17 30" src="https://github.com/user-attachments/assets/da3ba13a-9f21-4c14-bd10-6bc2bafbd109" />

La fenêtre "Model Manager" permet de:
- Sauvegarder le modèle actuel
- Charger un modèle préalablement sauvegardé
- Supprimer des modèles existants

### Classification d'images de jeux

<img width="1512" alt="Capture d’écran 2025-03-02 à 12 36 26" src="https://github.com/user-attachments/assets/5e6e7789-ce53-44f5-acaa-a2d26eb90232" />

Cette section vous permet d'entraîner un modèle pour classifier des images de jeux vidéo.

#### Préparation des données

1. Organisez vos images dans des sous-dossiers par catégorie:
   ```
   dataset/
   ├── fps/         # Images de jeux FPS
   │   ├── image1.jpg
   │   └── ...
   ├── moba/        # Images de jeux MOBA
   │   ├── image1.jpg
   │   └── ...
   └── rts/         # Images de jeux RTS
       ├── image1.jpg
       └── ...
   ```

2. Dans l'interface, entrez le chemin du dossier contenant vos catégories et cliquez sur "Load Dataset"

#### Configuration du modèle MLP

1. Ajustez la configuration du MLP:
   - Nombre de couches cachées
   - Nombre de neurones par couche
   - Fonction d'activation

2. Ajustez les hyperparamètres:
   - Learning Rate
   - Batch Size
   - Train Ratio

#### Entraînement du modèle

1. Cliquez sur "Start" pour démarrer l'entraînement
2. Observez les courbes d'apprentissage pour suivre la progression
3. Cliquez sur "Stop" pour arrêter l'entraînement
4. Sauvegardez le modèle une fois satisfait des résultats

#### Classification d'images

1. Entrez le chemin d'une image à classifier dans "Image path"
2. Cliquez sur "Classify image"
3. Le résultat s'affiche avec les scores pour chaque catégorie

### Idées d'améliorations

- Ajouter de nouveaux algorithmes
- Améliorer la visualisation des frontières de décision
- Ajouter le support pour d'autres types de données (audio, texte)
- Optimiser les performances de calcul avec du compute shader

---

Projet développé dans le cadre du cours de Machine Learning - ESGI 2024-2025.
