"""HTDemucs v4 „ft-vocals" nach ONNX — ADR 0009, Roadmap R3.4a.

Lädt **ein** Modell aus der Familie `htdemucs_ft` (das für Gesang,
`04573f0d`), exportiert es mit der STFT im Modell (Mixxx-Fork von Demucs)
und prüft die ONNX-Datei gegen das PyTorch-Original:

- **Original** heißt: dasselbe Modell mit der STFT von PyTorch
  (`onnx_exportable = False`), so wie Demucs es selbst rechnet.
- Verglichen wird der Abstand der Ausgaben als Signal-Rausch-Abstand in dB
  (SNR, Original als Bezug), je Quelle. Gefordert: mindestens 40 dB — die
  Abweichung ist dann hundertmal leiser als das Signal und verschiebt eine
  SDR-Messung gegen die Wahrheit um weniger als 0,1 dB (E141).
- **Zwei Proben.** „musik": ein Ausschnitt ohne Gesang — prüft Schlagzeug,
  Bass und den Rest. „gesang": eine künstliche Singstimme (Stimmimpulse mit
  Vibrato durch Vokalformanten, eine kleine Melodie) über derselben Musik —
  prüft die Gesangsspur. Sie zählt nur, wenn das Original dort wirklich
  Gesang findet (Pegel und Ähnlichkeit zur Stimme); sonst wäre der Vergleich
  einer stillen Spur kein Beleg.

Aufruf (Python-Umgebung unter `.modellbau/venv`, siehe `modellbau/README.md`):

    python modellbau/htdemucs_ft_vocals.py <gewichte.th> <zielordner> <musik.wav|mp3>
"""

import fractions
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from scipy.signal import lfilter
from torch.nn import functional as F

from demucs.htdemucs import HTDemucs
from demucs.states import load_model

QUELLEN = ["drums", "bass", "other", "vocals"]
# Was die Gewichtsdatei außer Tensoren enthält: die Modellklasse, Brüche
# und Zahlen von NumPy (unter ihrem alten Namen `numpy.core`).
ERLAUBTE_KLASSEN = [
    HTDemucs,
    fractions.Fraction,
    (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
    np.dtype,
    type(np.dtype(np.float64)),
    type(np.dtype(np.float32)),
    type(np.dtype(np.int64)),
]
SNR_MINDESTENS_DB = 40.0
# Die Gesangsprobe zählt nur, wenn das Original die Stimme findet.
GESANG_PEGEL_MINDESTENS = 0.01
GESANG_AEHNLICHKEIT_MINDESTENS = 0.8


def snr_db(bezug: np.ndarray, probe: np.ndarray) -> float:
    fehler = np.sum((bezug - probe) ** 2)
    signal = np.sum(bezug**2)
    if fehler == 0:
        return float("inf")
    return float(10 * np.log10(signal / fehler))


def sha256(pfad: Path) -> str:
    h = hashlib.sha256()
    with open(pfad, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def musik(pfad: str, laenge: int, rate: int) -> np.ndarray:
    """Ein Ausschnitt aus dem ersten Drittel, (2, laenge) float32."""
    ton, sr = sf.read(pfad, dtype="float32", always_2d=True)
    if sr != rate:
        raise SystemExit(f"{pfad}: {sr} Hz, erwartet {rate} Hz")
    ton = ton.T
    if ton.shape[0] == 1:
        ton = np.repeat(ton, 2, axis=0)
    start = min(ton.shape[1] // 3, max(0, ton.shape[1] - laenge))
    stueck = ton[:2, start : start + laenge]
    return np.pad(stueck, ((0, 0), (0, laenge - stueck.shape[1])))


def formant(x: np.ndarray, frequenz: float, breite: float, rate: int) -> np.ndarray:
    r = np.exp(-np.pi * breite / rate)
    winkel = 2 * np.pi * frequenz / rate
    return lfilter([1 - r], [1, -2 * r * np.cos(winkel), r * r], x)


def singstimme(laenge: int, rate: int) -> np.ndarray:
    """Acht Töne mit Vibrato, drei Vokale (a, i, o) im Wechsel, Spitze 0,25."""
    t = np.arange(laenge) / rate
    noten = [220, 247, 262, 294, 330, 294, 262, 247]
    f0 = np.repeat(noten, laenge // len(noten) + 1)[:laenge].astype(float)
    f0 = f0 * (1 + 0.012 * np.sin(2 * np.pi * 5.5 * t))
    phase = np.cumsum(f0) / rate
    puls = lfilter([1], [1, -0.97], (np.diff(np.floor(phase), prepend=0) > 0).astype(float))
    vokale = [(700, 1220, 2600), (300, 2300, 3000), (500, 900, 2500)]
    stimme = np.zeros(laenge)
    stueck = laenge // 6
    for i in range(6):
        a, b = i * stueck, min(laenge, (i + 1) * stueck)
        f1, f2, f3 = vokale[i % 3]
        p = puls[a:b]
        stimme[a:b] = formant(p, f1, 80, rate) + 0.6 * formant(p, f2, 100, rate) + 0.3 * formant(p, f3, 150, rate)
    einsatz = np.minimum(1, (t % (laenge / rate / 6)) * 20)
    stimme = stimme * einsatz
    return (stimme / np.max(np.abs(stimme)) * 0.25).astype(np.float32)


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    gewichte, ziel = Path(sys.argv[1]), Path(sys.argv[2])
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    musik_pfad = sys.argv[3]
    ziel.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    # Nur die Klassen, die die Datei braucht — kein `weights_only=False`,
    # das beliebigen Code aus der Datei ausführen ließe.
    with torch.serialization.safe_globals(ERLAUBTE_KLASSEN):
        modell = load_model(gewichte)
    if not isinstance(modell, HTDemucs):
        raise SystemExit(f"Kein HTDemucs: {type(modell).__name__}")
    modell.eval()
    rate = modell.samplerate
    laenge = int(modell.segment * rate)
    if list(modell.sources) != QUELLEN:
        raise SystemExit(f"Unerwartete Quellen: {modell.sources}")

    onnx_pfad = ziel / "htdemucs_ft_vocals.onnx"
    # Im Inferenzmodus nähme PyTorch für die Attention einen eigenen schnellen
    # Weg (`_native_multi_head_attention`), den ONNX nicht kennt; dieselbe
    # Rechnung über die einzelnen Operationen ist exportierbar.
    torch.backends.mha.set_fastpath_enabled(False)
    modell.onnx_exportable = True
    beispiel = F.pad(torch.randn(1, 2, laenge // 2), (0, laenge - laenge // 2))
    with torch.no_grad():
        torch.onnx.export(
            modell,
            (beispiel,),
            str(onnx_pfad),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamo=False,
        )

    sitzung = ort.InferenceSession(str(onnx_pfad), providers=["CPUExecutionProvider"])
    instrumental = musik(musik_pfad, laenge, rate)
    stimme = singstimme(laenge, rate)
    proben = {
        "musik": instrumental,
        "gesang": (0.6 * instrumental + np.stack([stimme, stimme])).astype(np.float32),
    }
    pegel = lambda a: float(np.sqrt(np.mean(a**2)))  # noqa: E731
    aehnlich = lambda a: float(np.corrcoef(a[0], stimme)[0, 1])  # noqa: E731
    ergebnis = {}
    for name, probe in proben.items():
        x = torch.from_numpy(probe[None])
        modell.onnx_exportable = False
        with torch.no_grad():
            original = modell(x).numpy()[0]
        aus_onnx = sitzung.run(["output"], {"input": probe[None]})[0][0]
        werte = {q: round(snr_db(original[i], aus_onnx[i]), 2) for i, q in enumerate(QUELLEN)}
        werte["pegel_gesang_original"] = round(pegel(original[3]), 5)
        if name == "gesang":
            werte["aehnlichkeit_original"] = round(aehnlich(original[3]), 3)
            werte["aehnlichkeit_onnx"] = round(aehnlich(aus_onnx[3]), 3)
        ergebnis[name] = werte

    g = ergebnis["gesang"]
    gesang_gefunden = (
        g["pegel_gesang_original"] >= GESANG_PEGEL_MINDESTENS
        and g["aehnlichkeit_original"] >= GESANG_AEHNLICHKEIT_MINDESTENS
        and g["aehnlichkeit_onnx"] >= GESANG_AEHNLICHKEIT_MINDESTENS
    )
    nah_am_original = g["vocals"] >= SNR_MINDESTENS_DB and all(
        ergebnis["musik"][q] >= SNR_MINDESTENS_DB for q in ["drums", "bass", "other"]
    )
    bericht = {
        "modell": "htdemucs_ft (vocals, 04573f0d)",
        "gewichte_sha256": sha256(gewichte),
        "onnx": onnx_pfad.name,
        "onnx_bytes": onnx_pfad.stat().st_size,
        "onnx_sha256": sha256(onnx_pfad),
        "samplerate": rate,
        "eingabe": [1, 2, laenge],
        "ausgabe": [1, len(QUELLEN), 2, laenge],
        "quellen": QUELLEN,
        "opset": 17,
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "snr_db_gegen_original": ergebnis,
        "mindestens_db": SNR_MINDESTENS_DB,
        "gesang_gefunden": gesang_gefunden,
        "bestanden": gesang_gefunden and nah_am_original,
    }
    (ziel / "htdemucs_ft_vocals.json").write_text(json.dumps(bericht, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(bericht, indent=2, ensure_ascii=False))
    if not gesang_gefunden:
        raise SystemExit("Die Gesangsprobe enthält für das Original keinen Gesang — der Vergleich belegt nichts.")
    if not nah_am_original:
        raise SystemExit("Abweichung zu groß — ONNX-Datei nicht verwenden.")


if __name__ == "__main__":
    main()
