"""
compute_midi_metrics.py

Quick MIDI-level evaluation script.

Usage:
  python eval/compute_midi_metrics.py --refs path/to/ref_midis --gens path/to/gen_midis --out results.csv
  or
  python eval/compute_midi_metrics.py --auto-dir path/to/parent_of_generated_samples

Notes:
- Matches files by basename (ref: a.mid, gen: a.mid) or by pairing <base>_GT.mid with <base>.mid under --auto-dir.
- Computes: onset precision/recall/f1, pitch accuracy (on matched onsets), note density (notes/sec), duration.
- Outputs CSV and a summary CSV (same dirname): <out> and <out>.summary.csv
- Requires: pretty_midi, numpy
"""

import argparse
import os
import csv
import numpy as np
import pretty_midi


def load_notes(midi_path):
    pm = pretty_midi.PrettyMIDI(midi_path)
    notes = []  # tuples (onset, pitch)
    for inst in pm.instruments:
        for n in inst.notes:
            notes.append((n.start, n.pitch))
    if len(notes) == 0:
        return np.array([], dtype=float), np.array([], dtype=int), 0.0
    notes = sorted(notes, key=lambda x: x[0])
    onsets = np.array([n[0] for n in notes], dtype=float)
    pitches = np.array([n[1] for n in notes], dtype=int)
    duration = pm.get_end_time()
    return onsets, pitches, duration


def match_onsets(ref_onsets, est_onsets, tol=0.05):
    """Greedy matching: for each est, find nearest unmatched ref within tol."""
    if len(ref_onsets) == 0 or len(est_onsets) == 0:
        return 0, []

    ref_used = np.zeros(len(ref_onsets), dtype=bool)
    matches = []  # list of (ref_idx, est_idx)
    for ei, e in enumerate(est_onsets):
        diffs = np.abs(ref_onsets - e)
        cand_idx = np.where((diffs <= tol) & (~ref_used))[0]
        if cand_idx.size > 0:
            best = cand_idx[np.argmin(diffs[cand_idx])]
            ref_used[best] = True
            matches.append((best, ei))
    return len(matches), matches


def onset_metrics(ref_onsets, est_onsets, tol=0.05):
    matches, _ = match_onsets(ref_onsets, est_onsets, tol=tol)
    p = matches / len(est_onsets) if len(est_onsets) > 0 else 0.0
    r = matches / len(ref_onsets) if len(ref_onsets) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1, matches


def pitch_accuracy(ref_onsets, ref_pitches, est_onsets, est_pitches, tol=0.05):
    matches, pairs = match_onsets(ref_onsets, est_onsets, tol=tol)
    if matches == 0:
        return 0.0, 0
    correct = 0
    for ri, ei in pairs:
        if est_pitches[ei] == ref_pitches[ri]:
            correct += 1
    acc = correct / matches
    return acc, matches


def eval_pair(ref_path, gen_path, tol=0.05):
    ref_onsets, ref_pitches, ref_dur = load_notes(ref_path)
    gen_onsets, gen_pitches, gen_dur = load_notes(gen_path)

    p, r, f1, matched = onset_metrics(ref_onsets, gen_onsets, tol=tol)
    pitch_acc, matched_count = pitch_accuracy(ref_onsets, ref_pitches, gen_onsets, gen_pitches, tol=tol)

    note_density_ref = len(ref_onsets) / ref_dur if ref_dur > 0 else 0.0
    note_density_gen = len(gen_onsets) / gen_dur if gen_dur > 0 else 0.0

    return {
        "ref_file": os.path.basename(ref_path),
        "gen_file": os.path.basename(gen_path),
        "ref_notes": len(ref_onsets),
        "gen_notes": len(gen_onsets),
        "ref_dur": ref_dur,
        "gen_dur": gen_dur,
        "onset_p": p,
        "onset_r": r,
        "onset_f1": f1,
        "pitch_acc": pitch_acc,
        "matched": matched_count,
        "note_density_ref": note_density_ref,
        "note_density_gen": note_density_gen,
    }


def find_pairs(ref_dir, gen_dir, match_gt_suffix=False):
    # If match_gt_suffix is True, look for files like NAME_GT.mid and NAME.mid in the same folder
    if match_gt_suffix:
        files = os.listdir(ref_dir)
        pairs = []
        for f in files:
            if f.endswith("_GT.mid"):
                base_no_ext = os.path.splitext(f)[0]
                base = base_no_ext[:-3]  # strip trailing '_GT'
                ref_path = os.path.join(ref_dir, f)
                gen_name = base + ".mid"
                gen_path = os.path.join(gen_dir, gen_name)
                if os.path.exists(gen_path):
                    pairs.append((ref_path, gen_path))
        return pairs

    refs = {os.path.splitext(f)[0]: os.path.join(ref_dir, f) for f in os.listdir(ref_dir) if f.endswith(".mid")}
    gens = {os.path.splitext(f)[0]: os.path.join(gen_dir, f) for f in os.listdir(gen_dir) if f.endswith(".mid")}
    common = sorted(set(refs.keys()) & set(gens.keys()))
    pairs = [(refs[k], gens[k]) for k in common]
    return pairs


def find_pairs_auto(root_dir, gt_suffix="_GT"):
    """Recursively scan `root_dir` and pair files where one file is named <base>_GT.mid and the other <base>.mid.

    Returns list of (ref_path, gen_path).
    """
    refs = {}
    gens = {}
    for dpath, _, files in os.walk(root_dir):
        for f in files:
            if not f.lower().endswith(".mid"):
                continue
            full = os.path.join(dpath, f)
            if f.endswith(gt_suffix + ".mid"):
                base = f[: -len(gt_suffix + ".mid")]
                refs[base] = full
            else:
                base = os.path.splitext(f)[0]
                if base.endswith(gt_suffix):
                    base = base[: -len(gt_suffix)]
                gens[base] = full

    common = sorted(set(refs.keys()) & set(gens.keys()))
    pairs = [(refs[k], gens[k]) for k in common]
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refs", required=False, help="dir of reference MIDI files")
    parser.add_argument("--gens", required=False, help="dir of generated MIDI files")
    parser.add_argument(
        "--auto-dir", required=False, help="root dir to recursively scan for paired *_GT.mid and .mid files"
    )
    parser.add_argument("--gt-suffix", default="_GT", help="suffix used to mark GT files (default: _GT)")
    parser.add_argument("--out", default="eval_results.csv")
    parser.add_argument("--match-gt-suffix", action="store_true", help="match NAME_GT.mid with NAME.mid in same folder")
    parser.add_argument("--tol", type=float, default=0.05, help="onset tolerance in seconds")
    args = parser.parse_args()

    # Auto-scan mode: if --auto-dir provided, find pairs under that root
    if args.auto_dir:
        pairs = find_pairs_auto(args.auto_dir, gt_suffix=args.gt_suffix)
    else:
        if not args.refs or not args.gens:
            print("Either --auto-dir or both --refs and --gens must be provided.")
            return
        pairs = find_pairs(args.refs, args.gens, match_gt_suffix=args.match_gt_suffix)

    if len(pairs) == 0:
        print("No matching MIDI filenames found.")
        return

    # If using --auto-dir and user didn't supply an explicit --out, set
    # out to out/<last_dir>/eval_results.csv for convenience.
    if args.auto_dir and (args.out is None or args.out == "eval_results.csv"):
        last_dir = os.path.basename(os.path.normpath(args.auto_dir))
        out_dir = os.path.join("out", last_dir)
        os.makedirs(out_dir, exist_ok=True)
        args.out = os.path.join(out_dir, "eval_results.csv")

    rows = []
    for ref_path, gen_path in pairs:
        print(f"Evaluating {os.path.basename(ref_path)}")
        r = eval_pair(ref_path, gen_path, tol=args.tol)
        rows.append(r)

    # write CSV
    keys = [
        "ref_file",
        "gen_file",
        "ref_notes",
        "gen_notes",
        "ref_dur",
        "gen_dur",
        "onset_p",
        "onset_r",
        "onset_f1",
        "pitch_acc",
        "matched",
        "note_density_ref",
        "note_density_gen",
    ]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})

    # print averages
    numeric_keys = [
        "ref_notes",
        "gen_notes",
        "ref_dur",
        "gen_dur",
        "onset_p",
        "onset_r",
        "onset_f1",
        "pitch_acc",
        "matched",
        "note_density_ref",
        "note_density_gen",
    ]

    summary = {}
    for k in numeric_keys:
        vals = np.array([float(row.get(k, 0.0)) for row in rows], dtype=float)
        summary[k] = {
            "count": int(vals.size),
            "mean": float(np.mean(vals)) if vals.size > 0 else None,
            "std": float(np.std(vals, ddof=0)) if vals.size > 0 else None,
            "var": float(np.var(vals, ddof=0)) if vals.size > 0 else None,
            "median": float(np.median(vals)) if vals.size > 0 else None,
            "min": float(np.min(vals)) if vals.size > 0 else None,
            "max": float(np.max(vals)) if vals.size > 0 else None,
        }

    # also keep simple averages for quick print
    quick_avg = {k: summary[k]["mean"] for k in ["onset_p", "onset_r", "onset_f1", "pitch_acc"]}
    print("Averages:")
    print(quick_avg)

    # write a human-readable summary CSV next to the CSV (easier to open)
    try:
        summary_csv = args.out + ".summary.csv"
        with open(summary_csv, "w", newline="") as scf:
            swriter = csv.writer(scf)
            # header
            swriter.writerow(["field", "count", "mean", "std", "var", "median", "min", "max"])
            for k in numeric_keys:
                v = summary.get(k, {})
                swriter.writerow(
                    [
                        k,
                        v.get("count", ""),
                        v.get("mean", ""),
                        v.get("std", ""),
                        v.get("var", ""),
                        v.get("median", ""),
                        v.get("min", ""),
                        v.get("max", ""),
                    ]
                )

        print(f"Wrote summary CSV to {summary_csv}")
    except Exception as e:
        print("Failed writing summary:", e)


if __name__ == "__main__":
    main()
