"""Gierige CTC-Erkennung mit einer ONNX-Datei — Funktionsprobe beim Modellbau (W.14c1).

Die Prüfung in `wav2vec2_ctc.py` vergleicht Zahlen mit dem Original. Diese
Probe zeigt zusätzlich, dass die Datei Sprache wirklich erkennt: Sie nimmt
das wahrscheinlichste Zeichen je Bild, fasst Wiederholungen zusammen, lässt
das Leerzeichen der CTC (`[PAD]` bzw. `<pad>`) weg und macht aus `|` ein
Leerzeichen — wie `Wav2Vec2Processor.decode`.

    python modellbau/wav2vec2_erkennen.py <modell.onnx> <vocab.json> <ton.wav>
"""

import json
import sys

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample_poly

RATE = 16_000


def main() -> None:
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    modell, vokabular_pfad, ton_pfad = sys.argv[1:4]
    vokabular = json.loads(open(vokabular_pfad, encoding="utf-8").read())
    zeichen = {i: z for z, i in vokabular.items()}
    # jonatasgrosman nennt es `<pad>` (Index 0), CLASSLA `[PAD]` (Index 36).
    leer = vokabular["[PAD]"] if "[PAD]" in vokabular else vokabular["<pad>"]

    ton, sr = sf.read(ton_pfad, dtype="float32", always_2d=True)
    ton = ton.mean(axis=1)
    if sr != RATE:
        ton = resample_poly(ton, RATE, sr).astype(np.float32)
    x = ((ton - ton.mean()) / np.sqrt(ton.var() + 1e-7)).astype(np.float32)

    sitzung = ort.InferenceSession(modell, providers=["CPUExecutionProvider"])
    logits = sitzung.run(["logits"], {"input_values": x[None]})[0][0]
    wege = logits.argmax(-1)
    text = []
    vorher = None
    for i in wege:
        if i != vorher and i != leer:
            # Eine Ausgabe ohne Zeichen (Serbisch hat zwei) bleibt leer.
            text.append(zeichen.get(int(i), ""))
        vorher = i
    satz = "".join(text).replace("|", " ")
    for sonder in ("[UNK]", "<unk>", "<s>", "</s>"):
        satz = satz.replace(sonder, "")
    print(json.dumps({"sekunden": round(len(x) / RATE, 2), "text": " ".join(satz.split())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
