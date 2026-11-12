"""wav2vec2 für die Ausrichtung nach ONNX — ADR 0009, Roadmap R3.5a, R3.6c, W.14c1.

Exportiert das CTC-Modell mit variabler Länge und prüft die ONNX-Datei gegen
das PyTorch-Original. Deutsch, Englisch und Spanisch sind wav2vec2-large-
XLSR-53 von jonatasgrosman (Apache-2.0); Serbisch ist XLS-R 300M von CLASSLA,
trainiert auf Južne vesti (CC BY-SA 4.0, Namensnennung, Frage 55).

- **Eingabe** `input_values`: float32 `[1, n]`, Mono, 16 kHz, **schon
  normiert** (Mittel 0, Streuung 1 — `do_normalize` des Feature-Extractors;
  das macht der Kern vor dem Modell).
- **Ausgabe** `logits`: float32 `[1, bilder, zeichen]`, ein Bild je 20 ms.
- **Geprüft** an zwei Proben unterschiedlicher Länge (die Länge ist
  variabel): der Abstand der Logits als SNR (Original als Bezug, mindestens
  40 dB) und das wahrscheinlichste Zeichen je Bild (mindestens 99,9 % gleich).
  Gewichte werden ohne ausführbaren Code geladen (Safetensors bzw.
  `weights_only`).

    python modellbau/wav2vec2_ctc.py <gewichte-ordner> <zielordner> <de|en|es|sr> <probe.wav>
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import Wav2Vec2ForCTC

RATE = 16_000
SNR_MINDESTENS_DB = 40.0
GLEICH_MINDESTENS = 0.999

# Je Sprache: Herkunft, Dateistamm und Lizenz der Gewichte.
QUELLEN = {
    "de": ("jonatasgrosman/wav2vec2-large-xlsr-53-german", "wav2vec2_xlsr53_de", "Apache-2.0", None),
    "en": ("jonatasgrosman/wav2vec2-large-xlsr-53-english", "wav2vec2_xlsr53_en", "Apache-2.0", None),
    "es": ("jonatasgrosman/wav2vec2-large-xlsr-53-spanish", "wav2vec2_xlsr53_es", "Apache-2.0", None),
    "sr": (
        "classla/wav2vec2-xls-r-juznevesti-sr",
        "wav2vec2_xlsr300m_sr",
        "CC BY-SA 4.0",
        "CLASSLA (classla/wav2vec2-xls-r-juznevesti-sr), trainiert auf dem Korpus Južne vesti; "
        "nach ONNX umgewandelt, sonst unverändert",
    ),
}


def sha256(pfad: Path) -> str:
    h = hashlib.sha256()
    with open(pfad, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def normiert(x: np.ndarray) -> np.ndarray:
    return ((x - x.mean()) / np.sqrt(x.var() + 1e-7)).astype(np.float32)


def main() -> None:
    if len(sys.argv) < 5:
        raise SystemExit(__doc__)
    quelle, ziel, sprache, probe = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
    if sprache not in QUELLEN:
        raise SystemExit("Sprache de, en, es oder sr")
    herkunft, stamm, lizenz, namensnennung = QUELLEN[sprache]
    ziel.mkdir(parents=True, exist_ok=True)

    modell = Wav2Vec2ForCTC.from_pretrained(quelle, use_safetensors=(quelle / "model.safetensors").exists())
    modell.eval()

    ton, sr = sf.read(probe, dtype="float32", always_2d=True)
    ton = resample_poly(ton.mean(axis=1), RATE, sr).astype(np.float32)
    proben = {"kurz": normiert(ton[: 5 * RATE]), "lang": normiert(ton[: 20 * RATE])}

    name = f"{stamm}.onnx"
    onnx_pfad = ziel / name
    beispiel = torch.from_numpy(proben["kurz"][None])
    with torch.no_grad():
        torch.onnx.export(
            modell,
            (beispiel,),
            str(onnx_pfad),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input_values"],
            output_names=["logits"],
            dynamic_axes={"input_values": {1: "samples"}, "logits": {1: "frames"}},
            dynamo=False,
        )

    sitzung = ort.InferenceSession(str(onnx_pfad), providers=["CPUExecutionProvider"])
    ergebnis = {}
    ausgaben = None
    for probe_name, x in proben.items():
        with torch.no_grad():
            original = modell(torch.from_numpy(x[None])).logits.numpy()[0]
        aus_onnx = sitzung.run(["logits"], {"input_values": x[None]})[0][0]
        ausgaben = int(aus_onnx.shape[-1])
        fehler = np.sum((original - aus_onnx) ** 2)
        snr = float(10 * np.log10(np.sum(original**2) / fehler)) if fehler > 0 else float("inf")
        gleich = float(np.mean(original.argmax(-1) == aus_onnx.argmax(-1)))
        ergebnis[probe_name] = {
            "sekunden": round(len(x) / RATE, 2),
            "bilder": int(original.shape[0]),
            "snr_db": round(snr, 2),
            "gleiches_zeichen": round(gleich, 5),
        }

    vokabular = json.loads((quelle / "vocab.json").read_text(encoding="utf-8"))
    shutil.copy(quelle / "vocab.json", ziel / f"{stamm}.vocab.json")
    bericht = {
        "modell": herkunft,
        "lizenz": lizenz,
        **({"namensnennung": namensnennung} if namensnennung else {}),
        "gewichte_sha256": sha256(next(p for p in [quelle / "model.safetensors", quelle / "pytorch_model.bin"] if p.exists())),
        "onnx": name,
        "onnx_bytes": onnx_pfad.stat().st_size,
        "onnx_sha256": sha256(onnx_pfad),
        "samplerate": RATE,
        "eingabe": "input_values [1, n], normiert (Mittel 0, Streuung 1)",
        "ausgabe": f"logits [1, bilder, {ausgaben}]",
        "zeichen": len(vokabular),
        # Serbisch: Das Modell hat 50 Ausgaben, das Vokabular nennt 48 — die
        # zwei ohne Zeichen gehören zu keinem Buchstaben (W.14c2).
        "ausgaben": ausgaben,
        "opset": 17,
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "gegen_original": ergebnis,
        "bestanden": all(
            e["snr_db"] >= SNR_MINDESTENS_DB and e["gleiches_zeichen"] >= GLEICH_MINDESTENS for e in ergebnis.values()
        ),
    }
    (ziel / f"{stamm}.json").write_text(
        json.dumps(bericht, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(bericht, indent=2, ensure_ascii=False))
    if not bericht["bestanden"]:
        raise SystemExit("Abweichung zu groß — ONNX-Datei nicht verwenden.")


if __name__ == "__main__":
    main()
