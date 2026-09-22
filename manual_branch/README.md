# Branche manuelle indépendante

Cette branche produit un embedding vocal `float32` de forme `[192]` et de norme L2 proche de 1. Elle fonctionne indépendamment d'ECAPA : aucun fichier `.npy` ECAPA n'est requis.

## Organisation des WAV

Les données peuvent être regroupées par source d'enregistrement puis par session de locuteur : `manual_branch/audio/Tel-Caroline/nabil2/*.wav`. Le générateur retire les chiffres finaux du dossier session (`nabil2` devient le label `nabil`) et conserve le nom de la source pour l'analyse. Les WAV doivent être mono, 16 kHz, non vides et finis.

## Générer les CSV

Depuis la racine :

```bash
source .venv/bin/activate
python -m manual_branch.prepare_data
```

Le script parcourt les sous-dossiers récursivement, crée les trois CSV au format `wav_path,label` et sépare chaque locuteur en environ 70 % entraînement, 15 % validation et 15 % test. Les suffixes de session sont normalisés (`andrea3`, `andrea2` → `andrea`).

## Entraînement

Le CNN utilise une perte contrastive supervisée : les embeddings du même locuteur sont rapprochés et ceux de locuteurs différents éloignés. Aucun classifieur n'est conservé dans l'encodeur.

```bash
python -m manual_branch.train --train-csv manual_branch/data/train.csv --validation-csv manual_branch/data/validation.csv --root manual_branch --epochs 50 --batch-size 16
```

Le meilleur modèle est enregistré dans `manual_branch/models/best.pt`. Chaque batch doit contenir au moins deux fichiers d'un même locuteur.

## Classifieur manuel

Une fois l'encodeur entraîné, entraînez séparément le classifieur de décision :

```bash
python -m manual_branch.train_decision --epochs 50 --batch-size 32
python -m manual_branch.evaluate_decision
```

Le classifieur reçoit un embedding `[192]` et retourne une classe de locuteur. Il est sauvegardé dans `models/manual_decision.pt`. Cette première tête est entraînée uniquement sur les embeddings manuels. Lorsque les embeddings ECAPA seront disponibles, la même architecture et les mêmes CSV pourront être utilisés pour entraîner et évaluer une tête ECAPA séparément, sans concaténer les deux embeddings.

## Inférence

```python
from manual_branch import encode_wav
embedding = encode_wav("manual_branch/audio/rasim/rasim-001.wav")
print(embedding.shape)  # torch.Size([192])
```

ECAPA peut traiter le même WAV en parallèle. Le modèle de décision séparé recevra ensuite l'embedding manuel, l'embedding ECAPA, ou les deux.

## Tests

```bash
python -m pytest -q manual_branch/test_pipeline.py
python -m manual_branch.test_pipeline --wav manual_branch/audio/rasim/rasim-001.wav
```

Le dossier `ecapa_embeddings/` n'est plus utilisé par cette branche indépendante.

## Test avec un microphone

Installez `sounddevice`, puis listez les périphériques si nécessaire :

```bash
python -m pip install sounddevice
python -m manual_branch.live_predict_manual --list-devices
```

Enregistrez ensuite quatre secondes et obtenez la décision manuelle :

```bash
python -m manual_branch.live_predict_manual --seconds 4
```

Le microphone est capturé en mono 16 kHz et passe par le même prétraitement que les WAV du dataset. Une confiance élevée n'est interprétable que si le signal est audible et enregistré dans des conditions proches du corpus.
