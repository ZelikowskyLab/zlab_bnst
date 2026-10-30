"""Step 4a of the pipeline (see README.md): cuts out the window before each
resident attack, for the attack-approach t-SNE.

For every start of a resident "Attack" (the `behavior` column of the feature
export), takes the WINDOW_FRAMES frames just before it and saves:

  data/attack_bouts/
      bouts.csv         one row per bout: id, group, trial, attack start
                        frame/time, and flags for filtering
      bout_frames.csv   one row per frame of each window: the resident's nose
                        in the intruder's frame (the t-SNE input), both mice's
                        cleaned nose/center/tail positions, and the labels
      bout_paths.npz    paths [n_bouts, WINDOW_FRAMES, 2] (the same relative
                        positions as an array) + bout_ids, in bouts.csv order

The relative positions are read from data/features_export/. The cleaned
positions of both mice aren't in that export, so they are recomputed here
from the tracking CSVs with the same cleaning as the export
(extract_features.interpolate_all_points), and checked against the export's
relative positions.

bouts.csv also gives the length of the attack that follows each window
(attack_frames, attack_duration in seconds).

Flags in bouts.csv:
  intruder_aggression_at_start   intruder scored "White Mouse Aggression" on
                                 the attack's first frame (mutual fighting)
  intruder_aggression_fraction   fraction of the window with that label
  previous_attack_in_window      the window includes frames of an earlier
                                 attack (attacks in quick succession)
  frames_since_previous_attack   frames from the end of the previous attack
                                 in the same trial to this start (blank if none)

Run directly:

    python3 extract_attack_bouts.py              # 60-frame (2 s) windows
    python3 extract_attack_bouts.py --window 90
"""

import argparse

import numpy as np
import pandas as pd

from extract_features import interpolate_all_points, calculate_point_relative_to_vector, _trial_number
from pipeline_paths import GROUPS, ETHO_ROOT, file_slug, find_trial_pairs
from export_features_csv import EXTRA_EXPORT_GROUPS, OUT_DIR as FEATURES_DIR

BOUTS_DIR = ETHO_ROOT.parent / "attack_bouts"
WINDOW_FRAMES = 60   # 2 s at ~30 fps
FRAME_SECONDS = 0.033  # EthoVision sample interval

ATTACK = "Attack"
INTRUDER_AGGRESSION = "White Mouse Aggression"
REL_COLS = ["resident_nose_relative_to_intruder_x", "resident_nose_relative_to_intruder_y"]
POSITION_COLS = ["X nose", "Y nose", "X center", "Y center", "X tail", "Y tail"]


def load_cleaned_positions(s1_file):
    """Both mice's cleaned positions for one trial, one row per frame, using
    the same cleaning as the feature export. Columns: resident_x_nose, ...,
    intruder_y_tail."""
    s2_file = s1_file.replace("_subject1.csv", "_subject2.csv")
    out = {}
    for mouse, path in [("resident", s1_file), ("intruder", s2_file)]:
        df = pd.read_csv(path, header=35, na_values="-", low_memory=False).drop(0).reset_index(drop=True)
        interpolate_all_points([df])
        for col in POSITION_COLS:
            axis, point = col.split()
            out[f"{mouse}_{axis.lower()}_{point}"] = df[col].to_numpy(dtype=float)
    return pd.DataFrame(out)


def extract_trial_bouts(features, window):
    """Returns [(start_frame, attack_frames, frames_since_previous_attack)]
    for every attack start in one trial that has a full window before it."""
    is_attack = (features["behavior"] == ATTACK).to_numpy()
    starts = np.flatnonzero(is_attack & ~np.r_[False, is_attack[:-1]])
    ends = np.flatnonzero(is_attack & ~np.r_[is_attack[1:], False])   # last frame of each attack

    bouts = []
    for start, end in zip(starts, ends):
        if start < window:
            print(f"    skipped an attack at frame {start}: fewer than {window} frames before it")
            continue
        previous_ends = ends[ends < start]
        since_previous = int(start - previous_ends[-1] - 1) if len(previous_ends) else None
        bouts.append((int(start), int(end - start + 1), since_previous))
    return bouts


def main(window):
    bout_rows, frame_tables, paths = [], [], []

    for group_folder, group_label in GROUPS + EXTRA_EXPORT_GROUPS:
        s1_files, _ = find_trial_pairs(group_folder)
        group_count = 0

        for s1_file in s1_files:
            trial = file_slug(s1_file)
            features = pd.read_csv(FEATURES_DIR / group_folder / f"{trial}.csv", low_memory=False)
            if not (features["behavior"] == ATTACK).any():
                continue

            positions = load_cleaned_positions(s1_file)
            assert len(positions) == len(features), f"{trial}: tracking and feature export have different lengths"

            # the recomputed positions must reproduce the export's relative positions
            rel = calculate_point_relative_to_vector(
                positions[["resident_x_nose", "resident_y_nose"]].to_numpy(),
                positions[["intruder_x_nose", "intruder_y_nose"]].to_numpy(),
                positions[["intruder_x_center", "intruder_y_center"]].to_numpy(),
            )
            assert np.allclose(rel, features[REL_COLS].to_numpy(), atol=1e-6), \
                f"{trial}: recomputed positions don't match the feature export"

            for start, attack_frames, since_previous in extract_trial_bouts(features, window):
                bout_id = len(bout_rows)
                win = slice(start - window, start)
                win_features = features.iloc[win]
                intruder_aggr = (win_features["intruder_behavior"] == INTRUDER_AGGRESSION).to_numpy()

                bout_rows.append({
                    "bout_id": bout_id,
                    "group_folder": group_folder,
                    "group": group_label,
                    "trial": trial,
                    "trial_number": _trial_number(s1_file),
                    "attack_start_frame": start,
                    "attack_start_time": float(features["trial_time"].iloc[start]),
                    "attack_frames": attack_frames,
                    "attack_duration": round(attack_frames * FRAME_SECONDS, 3),
                    "window_frames": window,
                    "intruder_aggression_at_start": bool(features["intruder_behavior"].iloc[start] == INTRUDER_AGGRESSION),
                    "intruder_aggression_fraction": round(float(intruder_aggr.mean()), 3),
                    "previous_attack_in_window": bool((win_features["behavior"] == ATTACK).any()),
                    "frames_since_previous_attack": since_previous,
                })

                frames = pd.DataFrame({
                    "bout_id": bout_id,
                    "frame_in_window": np.arange(window),          # 0 = start of window, window-1 = last frame before the attack
                    "frame": np.arange(start - window, start),
                    "trial_time": win_features["trial_time"].to_numpy(),
                    "rel_x": win_features[REL_COLS[0]].to_numpy(),
                    "rel_y": win_features[REL_COLS[1]].to_numpy(),
                })
                frames = pd.concat([frames, positions.iloc[win].reset_index(drop=True)], axis=1)
                frames["behavior"] = win_features["behavior"].to_numpy()
                frames["intruder_behavior"] = win_features["intruder_behavior"].to_numpy()
                frame_tables.append(frames)
                paths.append(win_features[REL_COLS].to_numpy())
                group_count += 1

        print(f"{group_label:8s} ({group_folder}): {group_count} bouts")

    bouts = pd.DataFrame(bout_rows)
    bouts["frames_since_previous_attack"] = bouts["frames_since_previous_attack"].astype("Int64")
    bout_frames = pd.concat(frame_tables, ignore_index=True)
    paths = np.stack(paths)

    BOUTS_DIR.mkdir(exist_ok=True)
    bouts.to_csv(BOUTS_DIR / "bouts.csv", index=False)
    bout_frames.to_csv(BOUTS_DIR / "bout_frames.csv", index=False)
    np.savez(BOUTS_DIR / "bout_paths.npz", paths=paths, bout_ids=bouts["bout_id"].to_numpy())

    print(f"\nwrote {len(bouts)} bouts x {window} frames to {BOUTS_DIR}")
    print(f"  paths array: {paths.shape}")
    return bouts, bout_frames, paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--window", type=int, default=WINDOW_FRAMES,
                        help="frames before each attack start to keep (default 60 = 2 s)")
    main(parser.parse_args().window)
