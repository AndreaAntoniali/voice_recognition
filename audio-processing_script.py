import os
import glob
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
from pydub import AudioSegment
from pydub.silence import detect_silence

def normaliser_lufs_array(samples_float, sample_rate, cible_lufs=-23.0):
    """
    Normalise directement un tableau numpy float32 1D en mono.
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

def traiter_audio(fichier_entree, dossier_sortie, locuteur_id):
    os.makedirs(dossier_sortie, exist_ok=True)

    # 1. Chargement et uniformisation (Mono + 16000 Hz)
    audio = AudioSegment.from_file(fichier_entree, format="m4a")
    audio = audio.set_channels(1)        # Assure le Mono AVANT toute extraction
    audio = audio.set_frame_rate(16000)  # Assure les 16000 Hz

    # 2. Troncature des silences (Conforme Audacity)
    audio_sans_silence = tronquer_silences(
        audio, 
        threshold=-40, 
        min_silence_len=500, 
        target_silence_len=200
    )

    # 3. Découpage en tranches de 3 secondes ET Normalisation LUFS par tranche
    duree_tranche = 3000
    total_ms = len(audio_sans_silence)
    index_segment = 1

    for i in range(0, total_ms, duree_tranche):
        segment = audio_sans_silence[i:i + duree_tranche]

        if len(segment) == duree_tranche:
            # Conversion propre Mono en numpy float32
            samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
            samples /= 32768.0

            # Normalisation LUFS individuelle par tranche de 3s (Conforme Audacity)
            samples_norm = normaliser_lufs_array(samples, segment.frame_rate, cible_lufs=-23.0)

            # Conversion d'export
            samples_int16 = (np.clip(samples_norm, -1.0, 1.0) * 32767).astype(np.int16)
            
            segment_normalise = AudioSegment(
                samples_int16.tobytes(),
                frame_rate=segment.frame_rate,
                sample_width=2,
                channels=1
            )

            nom_fichier = f"{locuteur_id}_sample_{index_segment:03d}.wav"
            chemin_export = os.path.join(dossier_sortie, nom_fichier)
            segment_normalise.export(chemin_export, format="wav")
            index_segment += 1

    print(f"Traitement terminé pour {locuteur_id} : {index_segment - 1} fichiers générés.")


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