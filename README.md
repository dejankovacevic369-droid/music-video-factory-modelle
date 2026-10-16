# Music Video Factory · Modelle

Dieses Repository enthält Modelldateien, die die Desktop-App *Music Video
Factory* auf Wunsch lädt, um lokal zu rechnen. Code der App gibt es hier
nicht. Die Dateien liegen als Release-Dateien bei, damit sie ohne Anmeldung
abrufbar sind.

Für jede Datei ist hier festgehalten:

- woher sie stammt und unter welcher Lizenz sie steht,
- mit welchem Skript sie erzeugt wurde,
- wie sie gegen das Original geprüft wurde,
- ihre Prüfsumme (SHA-256). Die App lädt nur Dateien, deren Prüfsumme sie
  kennt.

**Lizenzen:**

- Für die Skripte und Texte dieses Repositorys gilt [`LICENSE`](LICENSE)
  (MIT).
- Jedes Modell behält die Lizenz seiner Urheber. Sie steht in der Tabelle des
  jeweiligen Modells und in [`LIZENZEN/`](LIZENZEN/).

## HTDemucs v4 „ft-vocals“ (ONNX)

| | |
|---|---|
| Zweck | Gesang aus einem Song freistellen |
| Release | [`htdemucs-ft-vocals-1`](../../releases/tag/htdemucs-ft-vocals-1) |
| Datei | `htdemucs_ft_vocals.onnx`, 304.413.278 Bytes |
| SHA-256 | `194726def610fce4842e9bbd930fb6937d4cd1fcb2eaa75689457558a104801f` |
| Eingabe | `input`: float32 `[1, 2, 343980]`, Stereo, 44,1 kHz (7,8 s) |
| Ausgabe | `output`: float32 `[1, 4, 2, 343980]`, Quellen `drums, bass, other, vocals`. Verwendet wird nur `vocals`; das Instrumental ergibt sich als Mix minus Gesang |
| Lizenz | MIT, siehe [`LIZENZEN/demucs-MIT.txt`](LIZENZEN/demucs-MIT.txt) |

**Herkunft**

- Die Gewichte sind das Gesangsmodell der Familie `htdemucs_ft` von Demucs
  (Meta, Alexandre Défossez): Datei `04573f0d-f3cf25b2.th` von
  `dl.fbaipublicfiles.com/demucs/hybrid_transformer/`, SHA-256
  `f3cf25b222c4eed7cd49dd8b2c9597d50c18bd154090f7b919cfa5f93cf22c49`.
- Exportiert wurde mit dem ONNX-fähigen Fork von Mixxx
  ([`mixxxdj/demucs`](https://github.com/mixxxdj/demucs), Commit
  `d788c1a06876ced89b11d6531f771e5e40204d48`). Dort liegt die STFT im Modell.
- Werkzeuge: `torch` 2.8.0, Opset 17, geprüft mit `onnxruntime` 1.31.0.
- Das Skript dazu ist
  [`htdemucs_ft_vocals/htdemucs_ft_vocals.py`](htdemucs_ft_vocals/htdemucs_ft_vocals.py).

**Prüfung gegen das Original**

Gerechnet wurde mit demselben Modell in PyTorch, dort mit der STFT von
PyTorch. Gemessen wurde der Abstand der Ausgaben (SNR, Original als Bezug).
Gefordert waren mindestens 40 dB. Die Proben:

- ein Musikausschnitt ohne Gesang,
- eine künstliche Singstimme über derselben Musik. Diese Probe zählt nur,
  wenn das Original die Stimme wirklich als Gesang erkennt.

| Probe | Gesang | Schlagzeug | Bass | Rest | Ähnlichkeit zur Stimme (Original / ONNX) |
|---|---|---|---|---|---|
| Musik | (kein Gesang) | 69,6 dB | 73,3 dB | 75,8 dB | — |
| Singstimme über Musik | **86,8 dB** | 69,0 dB | 74,4 dB | 77,7 dB | 0,95 / 0,95 |

Der vollständige Bericht steht in
[`htdemucs_ft_vocals/htdemucs_ft_vocals.json`](htdemucs_ft_vocals/htdemucs_ft_vocals.json).

## wav2vec2-large-XLSR-53, Deutsch, Englisch und Spanisch (ONNX)

| | Deutsch | Englisch | Spanisch |
|---|---|---|---|
| Zweck | Liedtext auf die Gesangsspur ausrichten (CTC) | dasselbe | dasselbe |
| Release | [`wav2vec2-xlsr53-de-1`](../../releases/tag/wav2vec2-xlsr53-de-1) | [`wav2vec2-xlsr53-en-1`](../../releases/tag/wav2vec2-xlsr53-en-1) | [`wav2vec2-xlsr53-es-1`](../../releases/tag/wav2vec2-xlsr53-es-1) |
| Datei | `wav2vec2_xlsr53_de.onnx`, 1.262.310.086 Bytes | `wav2vec2_xlsr53_en.onnx`, 1.262.289.586 Bytes | `wav2vec2_xlsr53_es.onnx`, 1.262.322.386 Bytes |
| SHA-256 | `18bb0679d4948b3779ce8a091a7c70469c36f674910377f143bb55d9fdc5ce1f` | `deaef716ef06bc7bed6a3048f7f185a7d4ae00de257c6dd370026a93af66d86f` | `b7969bb73fdf241b12f9496f7a716243c6ed71ba83076fa0a29c7a068d2a8fc6` |
| Zeichen | 38 (`wav2vec2_xlsr53_de.vocab.json`) | 33 (`wav2vec2_xlsr53_en.vocab.json`) | 41 (`wav2vec2_xlsr53_es.vocab.json`) |
| Lizenz | Apache-2.0, siehe [`LIZENZEN/apache-2.0.txt`](LIZENZEN/apache-2.0.txt) | dasselbe | dasselbe |

**Eingabe:** `input_values` als float32 `[1, n]`, Mono, 16 kHz. Der Ton muss
vorher normiert sein, also Mittel 0 und Streuung 1. **Ausgabe:** `logits`
als float32 `[1, Bilder, Zeichen]`, ein Bild je 20 ms. Die Länge ist variabel.

**Herkunft:**

- Die Gewichte stammen von Jonatas Grosman, auf Hugging Face fest auf einen
  Commit genagelt:
  - Deutsch: `jonatasgrosman/wav2vec2-large-xlsr-53-german` @ `4b8a029`,
    Datei `pytorch_model.bin`, SHA-256
    `bff6d75ab89d8ca9cd103df9beb9c10f547501cf5a34aeabea1c8d736c1b81cb`.
  - Englisch: `jonatasgrosman/wav2vec2-large-xlsr-53-english` @ `569a623`,
    Datei `model.safetensors`, SHA-256
    `6144f8464c6aaa220dd57c5a2ad4039b5710dcf8ee6e67057675f76597c19875`.
  - Spanisch: `jonatasgrosman/wav2vec2-large-xlsr-53-spanish` @ `96d7e9b`,
    Datei `pytorch_model.bin`, SHA-256
    `41c110e55d2eac8c79486ad87dbe8f9527ed034fe087a6adf03c891eeba914c1`.
- Sie bauen auf XLSR-53 von Facebook auf, ebenfalls Apache-2.0.
- Exportiert mit `transformers` 4.57.1 und `torch` 2.8.0, Opset 17, über das
  Skript [`wav2vec2_xlsr53/wav2vec2_ctc.py`](wav2vec2_xlsr53/wav2vec2_ctc.py).
- Die Gewichte werden ohne ausführbaren Code geladen (Safetensors bzw.
  `weights_only`).

**Prüfung gegen das Original:** PyTorch und ONNX liefen an zwei Proben
unterschiedlicher Länge (5 s und 20 s). Gefordert waren mindestens 40 dB
Abstand der Logits und 99,9 % gleiches wahrscheinlichstes Zeichen je Bild.

| | 5 s | 20 s | gleiches Zeichen |
|---|---|---|---|
| Deutsch | 89,1 dB | 94,1 dB | 100 % |
| Englisch | 84,2 dB | 94,1 dB | 100 % |
| Spanisch | 89,7 dB | 91,2 dB | 100 % |

Die vollständigen Berichte stehen in `wav2vec2_xlsr53/*.json`.

## Whisper large-v3-turbo, quantisiert (ggml)

| | |
|---|---|
| Zweck | Liedtext aus der Gesangsspur erkennen, wenn kein Liedtext vorliegt. Das Ergebnis ist ein Entwurf; die Zeiten kommen danach aus der Ausrichtung mit wav2vec2 |
| Release | [`whisper-large-v3-turbo-q5_0-1`](../../releases/tag/whisper-large-v3-turbo-q5_0-1) |
| Datei | `ggml-large-v3-turbo-q5_0.bin`, 574.041.195 Bytes |
| SHA-256 | `394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2` |
| Format | ggml, gelesen von [whisper.cpp](https://github.com/ggml-org/whisper.cpp). Eingabe: Mono, 16 kHz |
| Lizenz | MIT, siehe [`LIZENZEN/whisper-MIT.txt`](LIZENZEN/whisper-MIT.txt) |

**Herkunft:**

- Die Datei ist **unverändert** gespiegelt. Sie stammt von den Autoren von
  whisper.cpp auf Hugging Face, fest auf einen Commit genagelt:
  `ggerganov/whisper.cpp` @ `5359861c739e955e79d9a303bcbc70fb988958b1`.
  Die Prüfsumme dort (Git LFS) ist dieselbe wie oben.
- Die Gewichte sind `whisper-large-v3-turbo` von OpenAI (MIT). whisper.cpp
  hat sie ins ggml-Format umgewandelt und auf 5 Bit quantisiert (`q5_0`),
  ebenfalls unter MIT.
- Gespiegelt wird, damit die App nur von einem Ort lädt, dessen Inhalt hier
  beschrieben und geprüft ist.

**Prüfung:** Erkannt wurden zwei synthetische Sprachproben (Windows-
Sprachausgabe, je vier Zeilen), ohne die Sprache vorzugeben. Gerechnet hat
whisper.cpp über `whisper-rs` 0.16.0, nur auf dem Prozessor mit 16 Threads.

| | Ton | Rechenzeit | erkannte Sprache | Wortfehlerrate |
|---|---|---|---|---|
| Deutsch | 10,7 s | 11,9 s | `de` | 0,00 |
| Englisch | 11,9 s | 11,9 s | `en` | 0,10 |

Die Fehler im Englischen: „harbor“ statt „harbour“, „Rhein“ statt „rain“,
„girls“ statt „gulls“. Der Bericht steht in
[`whisper/ggml-large-v3-turbo-q5_0.json`](whisper/ggml-large-v3-turbo-q5_0.json).
