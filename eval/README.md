Eval tools — MIDI-level metrics

This folder contains a lightweight MIDI evaluation script.

Files:
- `compute_midi_metrics.py`: CLI script that compares reference MIDI files and generated MIDI files (matched by basename). Computes onset precision/recall/F1, pitch accuracy on matched onsets, note density, and durations.

Quick start:
1. Ensure dependencies installed:

pip install pretty_midi numpy

2. Render or place reference MIDIs in `refs/` and generated MIDIs in `gens/` (matching basenames).

3. Run:

python eval/compute_midi_metrics.py --refs refs --gens gens --out eval_results.csv

Notes and next steps:
- This is a minimal starting point. I can extend it to: onset/pitch F1 from `mir_eval`, chord/tonality agreement (music21), LM perplexity (requires a symbolic LM), and audio-level metrics (FAD + openl3) if you want.
- If your generated data are in `.npz` format (measures), we can add a helper to convert `.npz` → `.mid` using the repo's `Token2Midi` or an existing renderer.
