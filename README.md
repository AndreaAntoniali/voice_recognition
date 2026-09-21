# voice_recognition

Reconnaissance de locuteur avec un petit CNN : chaque segment audio de 3 s est converti en
spectrogramme de Mel (dB) puis classé parmi les locuteurs connus.

## Installation

Le projet utilise [uv](https://docs.astral.sh/uv/) et Python ≥ 3.13.

```bash
uv sync
```

Toutes les commandes ci-dessous se lancent depuis la racine du projet avec
`uv run python <script>.py` (ou `.venv/bin/python <script>.py`).

## Organisation des données

Les fichiers audio sont attendus sous `audio_files/` (ignoré par git) :

```
audio_files/
├── audio_files/<nom>_processed/<nom>/*.wav          # dataset d'origine
└── audios-processed-*/audios-processed/
    ├── Audacity/16000Hz/sample{1,2,3}_*/<nom>-<k>.zip   # source "audacity"
    └── Python/sample{1,2,3}_*/<Nom>-<k>.zip             # source "python"
```

- Le **label** est le locuteur, déduit du nom du fichier/zip (mis en minuscules).
- Un **sample** (`sample1`, `sample2`, `sample3`) est une session d'enregistrement qui
  contient tous les locuteurs. Rasim n'a un zip 16 kHz que pour le sample 1.
- Les zips sont lus **directement en Python** (module `zipfile`), sans extraction.
- Format attendu : WAV PCM 16 bits. Les segments dont la durée s'écarte de 3 s (restes de
  découpage) sont ignorés.

## Scripts

| Fichier | Rôle |
|---|---|
| `preprocess.py` | WAV / zips → spectrogrammes de Mel, split, normalisation → `data/*.pt` |
| `train.py` | Entraînement (early stopping) et évaluation sur les splits de `data/` |
| `cross_validate.py` | Validation croisée « leave-one-sample-out » sur les zips + figure |
| `model.py` | Architecture `SpeakerCNN` (3 blocs conv + pooling adaptatif + linéaire) |
| `dataset.py` | `SpeakerDataset` ; `spec_augment` (SpecAugment) y est disponible mais désactivé dans `train.py` (`augment=False`) |
| `main.py` | Reste du modèle de projet `uv init`, non utilisé |

### 1. Prétraitement : `preprocess.py`

```bash
python preprocess.py
```

Écrit dans `data/` :

- `train_data.pt` : train du dataset d'origine (80 %) + zips Audacity des samples 1 et 2 ;
- `test_data.pt` : test du dataset d'origine (20 %, stratifié, seed fixe) ;
- `test_extra_data.pt` : zips Audacity du sample 3 (`HELD_OUT_SAMPLE`), une autre session.

Chaque fichier contient `X`, `y`, `source` (origine de chaque segment), `label_map`, les
statistiques de normalisation (calculées sur le train uniquement) et les paramètres du
spectrogramme (16 kHz, 64 bandes de Mel, fenêtre 25 ms, hop 10 ms).

### 2. Entraînement : `train.py`

```bash
python train.py              # dataset d'origine + zips (samples 1 et 2)
python train.py --no-extra   # baseline : dataset d'origine seul
```

Évalue ensuite le meilleur checkpoint sur le test d'origine et sur le sample 3
(accuracy, matrice de confusion, classification report). Checkpoints dans `models/`
(`best_model.pt`, ou `best_model_original.pt` avec `--no-extra`).

L'early stopping se base sur un split de validation tiré au hasard dans le train : les
segments d'un même enregistrement sont voisins, donc cette `val_acc` est optimiste et
ne sert qu'à choisir l'epoch. Seuls les tests font foi.

### 3. Validation croisée : `cross_validate.py`

```bash
python cross_validate.py                   # zips Audacity (défaut)
python cross_validate.py --source python   # zips du dossier Python/
```

Pour chaque sample, entraîne sur les deux autres et teste sur celui-là (3 folds),
sur 4 locuteurs (rasim exclu). Le dataset d'origine n'est pas utilisé. Pour chaque
fold : accuracy, matrice de confusion et classification report ; puis un récapitulatif
(moyenne, écart-type) et une figure `models/cv_<source>_confusion_matrices.png`
(les 3 matrices côte à côte). La normalisation est recalculée sur le train de chaque fold.

| Fold | Train | Test |
|---|---|---|
| A | sample2 + sample3 | sample1 |
| B | sample1 + sample3 | sample2 |
| C | sample1 + sample2 | sample3 |

## Résultats et limites

Accuracy moyenne en leave-one-sample-out (un run par fold) : environ 93 % (Audacity) et
98,6 % (Python).

**Attention : ces scores sont probablement surestimés.** Un classifieur linéaire qui ne
voit que les frames les plus silencieuses de chaque segment retrouve le locuteur sur un
sample inconnu bien au-dessus du hasard (70–93 % selon le fold, hasard = 25 %, avec une
exception à 32 % pour Audacity/sample3). Chaque locuteur semble donc avoir des conditions
d'enregistrement stables (pièce, micro, gain) que le CNN peut exploiter à la place de la
voix. Le contrôle avec labels mélangés retombe bien au hasard, et il n'y a ni doublon ni
recoupement audio entre samples : le problème vient de l'environnement, pas d'une fuite de
données du test vers le train.

## Configuration

Les constantes sont en tête de chaque script : paramètres du spectrogramme et
`HELD_OUT_SAMPLE` dans `preprocess.py` ; taille de batch, epochs, patience, learning rate
dans `train.py` ; `EXCLUDED` et `SAMPLES` dans `cross_validate.py`.
