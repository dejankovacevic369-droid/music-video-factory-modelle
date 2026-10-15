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
