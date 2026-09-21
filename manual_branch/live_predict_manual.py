"""Enregistre le microphone puis prédit le locuteur avec la branche manuelle."""
import argparse
import tempfile
import numpy as np
import soundfile as sf

try:
    import sounddevice as sd
except OSError as exc:
    sd = None
    _SOUNDDEVICE_ERROR = exc

from manual_branch.config import BRANCH_ROOT
from manual_branch.predict_manual import predict


def record_and_predict(seconds=4.0, sample_rate=16000, device=None, min_rms=0.01):
    if sd is None:
        raise RuntimeError(
            "PortAudio est absent. Installez-le avec "
            "sudo apt-get install libportaudio2 portaudio19-dev, puis relancez la commande."
        ) from _SOUNDDEVICE_ERROR
    if seconds <= 0:
        raise ValueError("La durée doit être positive.")
    print(f"Parlez pendant {seconds:.1f} secondes...", flush=True)
    recording = sd.rec(int(round(seconds * sample_rate)), samplerate=sample_rate, channels=1, dtype="float32", device=device)
    sd.wait()
    recording = np.asarray(recording[:, 0])
    rms = float(np.sqrt(np.mean(recording ** 2)))
    peak = float(np.max(np.abs(recording)))
    print(f"Niveau enregistré : RMS={rms:.5f}, crête={peak:.5f}", flush=True)
    if not np.isfinite(recording).all() or not np.any(recording):
        raise ValueError("Enregistrement vide ou invalide. Vérifiez le microphone.")
    if rms < min_rms:
        return "inconnu (signal trop silencieux)", 0.0, {}
    with tempfile.NamedTemporaryFile(suffix=".wav") as temporary:
        sf.write(temporary.name, recording, sample_rate, subtype="FLOAT")
        return predict(temporary.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--min-rms", type=float, default=0.01, help="Seuil de silence, défaut 0.01")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()
    if sd is None:
        raise SystemExit(
            "Erreur : PortAudio est absent. Lancez : "
            "sudo apt-get update && sudo apt-get install libportaudio2 portaudio19-dev"
        )
    if args.list_devices:
        print(sd.query_devices())
        return
    label, confidence, probabilities = record_and_predict(args.seconds, device=args.device, min_rms=args.min_rms)
    print(f"Locuteur prédit par la branche manuelle : {label}")
    print(f"Confiance : {confidence:.4f}")
    for name, value in sorted(probabilities.items(), key=lambda item: item[1], reverse=True):
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
