# MulliVC - Multilingual Voice Conversion System

Un système de conversion vocale multilingue qui préserve le contenu original tout en convertissant le timbre vers celui d'un autre locuteur, fonctionnant sans données appariées.

## 🏗️ Architecture

Le système MulliVC se compose de plusieurs composants clés :

- **Content Encoder** : Encode le contenu linguistique (WavLM/ContentVec)
- **Timbre Encoder** : Encode le timbre global du locuteur
- **Fine-grained Timbre Conformer** : Capture les détails fins du timbre
- **Mel Decoder** : Génère les mél-spectrogrammes synthétiques
- **Discriminateur (GAN)** : Améliore la qualité du spectrogramme généré

## 📦 Installation

```bash
# Cloner le repository
git clone <repository-url>
cd MulliVC

# Installer les dépendances
pip install -r requirements.txt

# Tester l'installation
python test_system.py
```

## 🚀 Utilisation

### Entraînement

```bash
# Entraînement standard
python train.py --config configs/mullivc_config.yaml

# Reprendre depuis un checkpoint
python train.py --config configs/mullivc_config.yaml --resume checkpoints/checkpoint_epoch_10.pt
```

### Inférence

```bash
# Conversion vocale standard
python inference.py --checkpoint checkpoints/best_model.pt --source_audio path/to/source.wav --target_speaker_audio path/to/target.wav --output converted.wav

# Conversion cross-linguale
python inference.py --checkpoint checkpoints/best_model.pt --source_audio path/to/source.wav --target_speaker_audio path/to/target.wav --output converted.wav --source_language en --target_language fongbe
```

### Démonstration

```bash
# Démonstration complète
python demo.py --config configs/mullivc_config.yaml --checkpoint checkpoints/best_model.pt

# Analyse du modèle
python demo.py --config configs/mullivc_config.yaml --analyze

# Benchmark de performance
python demo.py --config configs/mullivc_config.yaml --benchmark
```

### Évaluation

```bash
# Évaluation standard
python evaluation/evaluate.py --config configs/mullivc_config.yaml --checkpoint checkpoints/best_model.pt --output_dir evaluation_results

# Évaluation cross-linguale
python evaluation/evaluate.py --config configs/mullivc_config.yaml --checkpoint checkpoints/best_model.pt --cross_lingual --output_dir evaluation_results
```

## 📊 Données

Le système utilise :
- **LibriTTS** : Corpus anglais avec plusieurs locuteurs
- **Fongbe Speech** : Corpus Fongbe avec locuteurs masculins et féminins

Les datasets sont automatiquement téléchargés depuis HuggingFace lors du premier lancement.

## 🧪 Évaluation

### Métriques objectives
- **WER/CER** : Évaluation de l'intelligibilité avec Whisper
- **SIM** : Similarité du locuteur avec ECAPA-TDNN
- **Qualité audio** : Métriques spectrales et temporelles

### Métriques subjectives
- **MOS** : Évaluation subjective de la qualité
- **Similarité du locuteur** : Évaluation humaine
- **Préservation du contenu** : Évaluation de l'intelligibilité

## 🔧 Configuration

Le fichier `configs/mullivc_config.yaml` contient tous les paramètres du système :

- **Données** : Chemins, paramètres audio, batch size
- **Modèle** : Architecture des composants
- **Entraînement** : Hyperparamètres, pertes, optimiseurs
- **Chemins** : Dossiers de sortie

## 📁 Structure du projet

```
MulliVC/
├── models/                 # Modèles du système
│   ├── content_encoder.py  # Encodeur de contenu
│   ├── timbre_encoder.py   # Encodeur de timbre
│   ├── fine_grained_conformer.py  # Conformer fine-grained
│   ├── mel_decoder.py      # Décodeur mél
│   ├── discriminator.py    # Discriminateur GAN
│   ├── losses.py          # Fonctions de perte
│   └── mullivc.py         # Modèle principal
├── utils/                  # Utilitaires
│   ├── audio_utils.py     # Traitement audio
│   ├── data_utils.py      # Chargement des données
│   └── model_utils.py     # Utilitaires modèles
├── evaluation/           # Évaluation
│   ├── metrics.py         # Métriques d'évaluation
│   └── evaluate.py        # Script d'évaluation
├── configs/               # Configurations
│   ├── mullivc_config.yaml
│   └── wandb_config.yaml
├── train.py              # Script d'entraînement
├── inference.py          # Script d'inférence
├── demo.py               # Script de démonstration
├── test_system.py        # Tests du système
└── requirements.txt      # Dépendances
```

## 🎯 Fonctionnalités

### Conversion monolingue
- Préservation du contenu linguistique
- Conversion du timbre vers un autre locuteur
- Qualité audio élevée

### Conversion cross-linguale
- Conversion entre langues différentes
- Préservation du contenu sémantique
- Adaptation du timbre à la langue cible

### Entraînement en 3 étapes
1. **Étape 1** : Entraînement standard (monolingue)
2. **Étape 2** : Conversion croisée simulée
3. **Étape 3** : Cohérence cyclique

## 🚀 Démarrage rapide

```bash
# 1. Tester l'installation
python test_system.py

# 2. Démonstration avec données synthétiques
python demo.py --config configs/mullivc_config.yaml

# 3. Entraînement (nécessite des données)
python train.py --config configs/mullivc_config.yaml

# 4. Inférence
python inference.py --checkpoint checkpoints/best_model.pt --source_audio source.wav --target_speaker_audio target.wav --output converted.wav
```

## 📈 Monitoring

Le système supporte Weights & Biases pour le monitoring :

```bash
# Activer W&B dans la configuration
wandb:
  enabled: true
  project: "mullivc"
```

## 🤝 Contribution

1. Fork le repository
2. Créer une branche feature
3. Commiter les changements
4. Pousser vers la branche
5. Ouvrir une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.
