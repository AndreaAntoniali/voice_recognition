"""Prétraitement des enregistrements bruts (.m4a, .wav...) en segments de 3 s.

Pipeline : mono 16 kHz -> troncature des silences (comme Audacity) -> découpage en
tranches de 3 s -> normalisation LUFS (-23) par tranche. C'est le pipeline qui a produit
les segments de la source "python" ; `predict.py` le réutilise à l'identique pour
classer un nouvel enregistrement avec le modèle entraîné.

En script (`python audio_processing.py`), écrit les segments en WAV pour tout un dossier.
Décoder un `.m4a` nécessite ffmpeg installé sur le système.
"""

import os
import glob
import numpy as np
import pyloudnorm as pyln
from pydub import AudioSegment
from pydub.silence import detect_silence

SAMPLE_RATE = 16000
DUREE_TRANCHE_MS = 3000


def normaliser_lufs_array(samples_float, sample_rate, cible_lufs=-23.0):
    """
    Normalise directement un tableau numpy float32 1D en mono.
    Retourne le tableau inchangé si la sonie est indéfinie (segment silencieux).
    """
    meter = pyln.Meter(sample_rate)
    loudness = meter.integrated_loudness(samples_float)
    # Gestion des segments très courts ou silencieux
    if np.isinf(loudness):
        return samples_float
    return pyln.normalize.loudness(samples_float, loudness, cible_lufs)

def tronquer_silences(audio, threshold=-40, min_silence_len=500, target_silence_len=200):
    """
    Reproduit le comportement de Truncate Silence d'Audacity :
    réduit les silences de > 500ms à exactement 200ms au lieu de découper/recoller brut.

    Args:
        audio: `AudioSegment` mono.
        threshold: seuil de silence en dBFS.
        min_silence_len: durée minimale (ms) d'un silence à tronquer.
        target_silence_len: durée (ms) conservée pour chaque silence tronqué.
    """
    silences = detect_silence(audio, min_silence_len=min_silence_len, silence_thresh=threshold)
    if not silences:
        return audio

    resultat = AudioSegment.empty()
    dernier_fin = 0

    for debut, fin in silences:
        # Conserve la partie sonore précédente
        resultat += audio[dernier_fin:debut]
        # Ajoute la durée de silence tronquée (comme Audacity à 0.2s)
        resultat += audio[debut:debut + target_silence_len]
        dernier_fin = fin

    resultat += audio[dernier_fin:]
    return resultat

def charger_audio(fichier_entree):
    """Charge un fichier audio (format déduit de l'extension) en mono 16 kHz.

    Raises:
        pydub.exceptions.CouldntDecodeError: si le format n'est pas décodable (ffmpeg
            manquant pour `.m4a`, par exemple).
    """
    extension = os.path.splitext(fichier_entree)[1].lstrip(".").lower() or None
    audio = AudioSegment.from_file(fichier_entree, format=extension)
    audio = audio.set_channels(1)  # Assure le Mono AVANT toute extraction
    return audio.set_frame_rate(SAMPLE_RATE)


def decouper_en_segments(audio):
    """Tronque les silences, découpe en tranches de 3 s et normalise chaque tranche.

    La dernière tranche incomplète (< 3 s) est abandonnée, comme dans le pipeline
    d'origine.

    Args:
        audio: `AudioSegment` mono 16 kHz (voir `charger_audio`).

    Returns:
        Liste de tableaux numpy `int16` de 48000 échantillons (3 s à 16 kHz).
    """
    audio_sans_silence = tronquer_silences(
        audio,
        threshold=-40,
        min_silence_len=500,
        target_silence_len=200
    )

    segments = []
    total_ms = len(audio_sans_silence)
    for i in range(0, total_ms, DUREE_TRANCHE_MS):
        segment = audio_sans_silence[i:i + DUREE_TRANCHE_MS]

        if len(segment) == DUREE_TRANCHE_MS:
            # Conversion propre Mono en numpy float32
            samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
            samples /= 32768.0

            # Normalisation LUFS individuelle par tranche de 3s (Conforme Audacity)
            samples_norm = normaliser_lufs_array(samples, segment.frame_rate, cible_lufs=-23.0)

            segments.append((np.clip(samples_norm, -1.0, 1.0) * 32767).astype(np.int16))
    return segments


def traiter_audio(fichier_entree, dossier_sortie, locuteur_id):
    """Découpe un enregistrement et écrit chaque segment en WAV dans `dossier_sortie`
    (`<locuteur_id>_sample_001.wav`, ...)."""
    os.makedirs(dossier_sortie, exist_ok=True)

    segments = decouper_en_segments(charger_audio(fichier_entree))

    for index_segment, samples_int16 in enumerate(segments, start=1):
        segment_normalise = AudioSegment(
            samples_int16.tobytes(),
            frame_rate=SAMPLE_RATE,
            sample_width=2,
            channels=1
        )
        nom_fichier = f"{locuteur_id}_sample_{index_segment:03d}.wav"
        chemin_export = os.path.join(dossier_sortie, nom_fichier)
        segment_normalise.export(chemin_export, format="wav")

    print(f"Traitement terminé pour {locuteur_id} : {len(segments)} fichiers générés.")


# Traitement de tout un dossier
if __name__ == "__main__":
    dossier_entree = "./audios_raw/traitement"
    dossier_racine_sortie = "./audios_processed/python"

    # Récupère tous les fichiers .m4a du dossier d'entrée
    fichiers_m4a = glob.glob(os.path.join(dossier_entree, "*.m4a"))

    for chemin_fichier in fichiers_m4a:
        # Extrait le nom du fichier sans extension pour servir d'ID locuteur
        nom_fichier = os.path.basename(chemin_fichier)
        locuteur_id = os.path.splitext(nom_fichier)[0]

        # Définition du dossier de sortie propre à ce locuteur/fichier
        dossier_sortie_locuteur = os.path.join(dossier_racine_sortie, locuteur_id)

        print(f"Traitement de : {nom_fichier}...")
        traiter_audio(chemin_fichier, dossier_sortie_locuteur, locuteur_id)