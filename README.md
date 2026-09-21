# voice_recognition

Reconnaissance de locuteur avec un petit CNN : chaque segment audio de 3 s est converti en
spectrogramme de Mel (dB) puis classé parmi les locuteurs connus. Le projet permet
d'entraîner le modèle, de l'évaluer par validation croisée et de prédire le locuteur d'un
nouvel enregistrement brut.

## Installation

Le projet utilise [uv](https://docs.astral.sh/uv/) et Python ≥ 3.13.

```bash
uv sync
```

Les commandes ci-dessous se lancent depuis la racine du projet avec
`uv run python <script>.py` (ou `python <script>.py` dans l'environnement activé).

Décoder des `.m4a` (`audio_processing.py`, `predict.py`) demande **ffmpeg et ffprobe**
installés sur le système (`sudo apt install ffmpeg`). Les `.wav` fonctionnent sans.

## Organisation des données

Les fichiers audio sont attendus sous `audio_files/` (ignoré par git) :

```
audio_files/
├── audio_files/<nom>_processed/<nom>/*.wav              # dataset d'origine
└── audios-processed-*/audios-processed/
    ├── Audacity/16000Hz/sample{1,2,3}_*/<nom>-<k>.zip   # source "audacity"
    └── Python/sample{1,2,3}_*/<Nom>-<k>.zip             # source "python"
```

- Le **label** est le locuteur, déduit du nom du fichier ou du zip (mis en minuscules).
- Un **sample** (`sample1`, `sample2`, `sample3`) est une session d'enregistrement qui
  contient tous les locuteurs. Rasim n'a un zip 16 kHz que pour le sample 1.
- Les zips sont lus **directement en Python** (module `zipfile`), sans extraction.
- Format attendu : WAV PCM 16 bits. Les segments dont la durée s'écarte de 3 s (restes de
  découpage) sont ignorés.
- La source **« python »** est produite par `audio_processing.py` (silences tronqués,
  segments de 3 s, sonie normalisée à −23 LUFS) ; `predict.py` applique le même traitement
  à un nouvel audio.

## Scripts

| Fichier | Rôle |
|---|---|
| `preprocess.py` | WAV / zips → spectrogrammes de Mel, split, normalisation → `data/*.pt` |
| `train.py` | Entraînement (early stopping) : modèle sur `data/*.pt`, ou modèle final sur une source de zips |
| `cross_validate.py` | Validation croisée « leave-one-sample-out » sur les zips + figure |
| `audio_processing.py` | Enregistrement brut (.m4a, .wav) → segments de 3 s normalisés |
| `predict.py` | Prédit le locuteur d'un enregistrement avec un modèle entraîné |
| `model.py` | Architecture `SpeakerCNN` (3 blocs conv + pooling adaptatif + linéaire) |
| `dataset.py` | `SpeakerDataset` et `spec_augment` (SpecAugment, actif à l'entraînement) |

### 1. Prétraitement : `preprocess.py`

```bash
python preprocess.py
```

Écrit dans `data/` (utilisé par `python train.py` sans `--source`) :

- `train_data.pt` : train du dataset d'origine (80 %) + zips Audacity des samples 1 et 2 ;
- `test_data.pt` : test du dataset d'origine (20 %, stratifié, seed fixe) ;
- `test_extra_data.pt` : zips Audacity du sample 3 (`HELD_OUT_SAMPLE`), une autre session.

Chaque fichier contient `X`, `y`, `source` (origine de chaque segment), `label_map`, les
statistiques de normalisation (calculées sur le train uniquement) et les paramètres du
spectrogramme (16 kHz, 64 bandes de Mel, fenêtre 25 ms, hop 10 ms).

### 2. Entraînement : `train.py`

```bash
python train.py                  # dataset d'origine + zips Audacity (samples 1 et 2)
python train.py --no-extra       # baseline : dataset d'origine seul
python train.py --source python  # modèle final : tous les samples de la source Python
```

| Commande | Données | Évaluation | Checkpoint |
|---|---|---|---|
| `train.py` | `data/train_data.pt` | test d'origine + sample 3 | `models/best_model.pt` |
| `train.py --no-extra` | segments d'origine de `data/train_data.pt` | idem | `models/best_model_original.pt` |
| `train.py --source python` | les 3 samples Python, 4 classes (rasim exclu) | **aucune** | `models/best_model_python.pt` |

Les checkpoints embarquent le `label_map` et les statistiques de normalisation (`mean`,
`std`) utilisées à l'entraînement.

L'early stopping se base sur un split de validation tiré au hasard dans le train : les
segments d'un même enregistrement sont voisins, donc la `val_acc` est optimiste et ne
sert qu'à choisir l'epoch. Avec `--source python`, tous les samples servent à
l'entraînement et il n'y a pas de jeu de test : l'estimation honnête est celle de
`cross_validate.py`.

### 3. Validation croisée : `cross_validate.py`

```bash
python cross_validate.py                   # zips Audacity (défaut)
python cross_validate.py --source python   # zips du dossier Python/
```

Pour chaque sample, entraîne sur les deux autres et teste sur celui-là (3 folds), sur
4 locuteurs (rasim exclu). Le dataset d'origine n'est pas utilisé. Affiche pour chaque fold
l'accuracy, la matrice de confusion et le classification report, puis un récapitulatif
(moyenne, écart-type), et sauvegarde `models/cv_<source>_confusion_matrices.png` (les
3 matrices côte à côte). La normalisation est recalculée sur le train de chaque fold.

| Fold | Train | Test |
|---|---|---|
| A | sample2 + sample3 | sample1 |
| B | sample1 + sample3 | sample2 |
| C | sample1 + sample2 | sample3 |

Les checkpoints des folds sont écrits dans `models/cv_<source>_<sample>.pt`.

### 4. Découpage d'enregistrements bruts : `audio_processing.py`

```bash
python audio_processing.py
```

Découpe tous les `.m4a` de `./audios_raw/traitement/` et écrit les segments WAV dans
`./audios_processed/python/<nom>/`. Pipeline : mono 16 kHz → troncature des silences
(> 500 ms ramenés à 200 ms, comme Audacity) → tranches de 3 s → normalisation LUFS à
−23 par tranche. La dernière tranche incomplète (< 3 s) est abandonnée. Les fonctions
`charger_audio()` et `decouper_en_segments()` sont réutilisées par `predict.py`.

### 5. Prédire le locuteur d'un audio : `predict.py`

```bash
python predict.py enregistrement.m4a                                   # models/best_model_python.pt
python predict.py enregistrement.wav --model models/cv_python_sample2.pt
```

Le fichier brut suffit : le script applique `audio_processing.py`, calcule les mêmes
spectrogrammes et la même normalisation qu'à l'entraînement, classe chaque segment de 3 s
et affiche la prédiction de chaque segment, la probabilité moyenne par locuteur et le
locuteur retenu.

- Il faut **au moins 3 s** exploitables après troncature des silences. Plus l'audio est
  long, plus la prédiction moyenne est fiable.
- Le modèle par défaut est `models/best_model_python.pt` (`python train.py --source python`).
  S'il manque, le script le dit.
- Les anciens checkpoints sans `mean` / `std` (créés avant l'ajout de ces champs) sont
  acceptés, mais `predict.py` reprend alors les statistiques de `data/train_data.pt`,
  correctes seulement pour `best_model.pt`. Ré-entraînez les autres pour les stocker.
- Choisir un modèle cohérent avec l'audio : `best_model.pt` (Audacity + données d'origine)
  se trompe sur des audios traités par `audio_processing.py`.
- Un locuteur absent des classes du modèle (par exemple rasim avec `best_model_python.pt`)
  est nécessairement classé comme un autre.

## Résultats et limites

Accuracy moyenne en leave-one-sample-out (un run par fold, SpecAugment actif) :

| Source | sample1 | sample2 | sample3 | Moyenne |
|---|---|---|---|---|
| Audacity | 93,8 % | 91,7 % | 92,9 % | 92,8 % |
| Python | 98,9 % | 98,0 % | 99,0 % | 98,6 % |

**Ces scores sont probablement surestimés.** Un classifieur linéaire qui ne voit que les
frames les plus silencieuses de chaque segment retrouve le locuteur sur un sample inconnu
bien au-dessus du hasard (70–93 % selon le fold, hasard = 25 %, avec une exception à 32 %
pour Audacity/sample3). Chaque locuteur semble donc avoir des conditions d'enregistrement
stables (pièce, micro, gain) que le CNN peut exploiter à la place de la voix. En revanche,
il n'y a ni doublon ni recoupement audio entre samples, et un contrôle avec labels mélangés
retombe au hasard : le problème vient de l'environnement, pas d'une fuite du test vers le
train. Sur un audio enregistré dans un environnement nouveau pour un locuteur, le modèle
peut donc être nettement moins bon.

## Configuration

Les constantes sont en tête de chaque script : paramètres du spectrogramme, `ZIP_GLOBS`,
`HELD_OUT_SAMPLE` et `FINAL_EXCLUDED_CLASSES` dans `preprocess.py` ; taille de batch,
epochs, patience et learning rate dans `train.py` ; `SAMPLES` dans `cross_validate.py`.
