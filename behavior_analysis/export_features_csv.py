"""Runs the full feature-extraction pipeline (extract_features.py, with the
fixed calculate_radial_velocity and center-relative nose/tail
interpolation) once per trial, and writes one CSV per trial to a new
export folder mirroring the group structure (Updated-C1/A, Updated-C1/C,
Updated-C2/B).

Trials are NOT cropped here -- every frame is written out, including the
stretch before the intruder is physically present (~2 min in, see
export_all_feature_plots.py's CROP_SECONDS), so that decision is left to
whatever reads this CSV rather than being baked into the export. Frames
before the intruder shows up will have its position held at a single
frozen value (fill_missing's leading-edge behavior), which reads as
near-zero intruder velocity/acceleration for that stretch -- filter on
trial_time downstream if that's not wanted for a given analysis.

Each CSV has one row per frame: frame index, trial_time, the resident's
annotated behavior label, the intruder's (White Mouse ...) label -- both
blank where unannotated; rows are NOT dropped, so this is suitable for an
all-frames analysis -- and the 33 named kinematic features. Where the
resident's scored behaviors overlap, the most recently started one wins
(see extract_features.extract_behavior_frames).

The 33 features are inherently per-trial (per resident-intruder pair), not
per individual mouse -- roughly half of them (the "intermouse" features)
only make sense as a pair, so one file per trial is the natural unit here,
not one file per mouse.

Run directly:

    python3 export_features_csv.py            # every group, incl. Opto
    python3 export_features_csv.py Opto       # only the listed group folders
"""

import pandas as pd

from extract_features import interpolate_all_points, extract_behavior_frames, extract_feature_set
from pipeline_paths import GROUPS, ETHO_ROOT, trial_label, file_slug, find_trial_pairs

OUT_DIR = ETHO_ROOT.parent / "features_export"

# per-group behavior file conventions: (header row, time column), matching
# extract_features.py main()'s own per-group parsing.
BEHAVIOR_CONFIG = {
    "Updated-C1/A": (0, "Time_Relative_sf", "Event_Type"),
    "Updated-C1/C": (0, "Time_Relative_sf", "Event_Type"),
    "Updated-C2/B": (35, "Trial time", "Event"),
    "Opto": (0, "Time_Relative_sf", "Event_Type"),
    "Updated-C2/C": (0, "Time_Relative_sf", "Event"),
}

# Exported alongside GROUPS but kept out of it, so the UMAP (which loads
# only GROUPS) doesn't pick these up.
EXTRA_EXPORT_GROUPS = [
    ("Opto", "Opto"),
]

FEATURE_NAMES = [
    "resident_nose_velocity", "resident_center_velocity", "resident_tail_velocity",
    "intruder_nose_velocity", "intruder_center_velocity", "intruder_tail_velocity",
    "resident_nose_acceleration", "resident_center_acceleration", "resident_tail_acceleration",
    "intruder_nose_acceleration", "intruder_center_acceleration", "intruder_tail_acceleration",
    "resident_body_angle", "intruder_body_angle",
    "intruder_nose_radial_velocity", "resident_nose_radial_velocity",
    "resident_nose_tail_distance", "intruder_nose_tail_distance",
    "intermouse_nose_nose_distance", "intermouse_nose_center_distance", "intermouse_nose_tail_distance",
    "intermouse_nose_nose_velocity", "intermouse_nose_center_velocity", "intermouse_nose_tail_velocity",
    "intermouse_nose_nose_acceleration", "intermouse_nose_center_acceleration", "intermouse_nose_tail_acceleration",
    "relative_head_angle",
    "intermouse_nose_nose_radial_velocity", "intermouse_nose_center_radial_velocity", "intermouse_nose_tail_radial_velocity",
    "resident_nose_relative_to_intruder_x", "resident_nose_relative_to_intruder_y",
]
assert len(FEATURE_NAMES) == 33


def build_combined_df(resident_df, intruder_df):
    cols = ["X nose", "Y nose", "X center", "Y center", "X tail", "Y tail"]
    return pd.concat(
        [resident_df[cols], intruder_df[cols].rename(columns=str.lower)],
        axis=1,
    )


def main(only_groups=None):
    total = 0
    for group_folder, group_label in GROUPS + EXTRA_EXPORT_GROUPS:
        if only_groups and group_folder not in only_groups:
            continue
        s1_files, behavior_files = find_trial_pairs(group_folder)
        header, time_col, event_col = BEHAVIOR_CONFIG[group_folder]

        out_subdir = OUT_DIR / group_folder
        out_subdir.mkdir(parents=True, exist_ok=True)

        for s1_file, behavior_file in zip(s1_files, behavior_files):
            s2_file = s1_file.replace("_subject1.csv", "_subject2.csv")
            label = trial_label(s1_file)
            slug = file_slug(s1_file)

            resident_df = pd.read_csv(s1_file, header=35, na_values="-", low_memory=False).drop(0).reset_index(drop=True)
            intruder_df = pd.read_csv(s2_file, header=35, na_values="-", low_memory=False).drop(0).reset_index(drop=True)

            interpolate_all_points([resident_df])
            interpolate_all_points([intruder_df])

            behavior_df = pd.read_csv(behavior_file, header=header).dropna(subset=["Behavior"])
            resident_frames, intruder_frames = extract_behavior_frames(
                [behavior_df["Behavior"].values], [behavior_df[event_col].values],
                [pd.to_numeric(behavior_df[time_col]).values], [resident_df],
            )

            combined_df = build_combined_df(resident_df, intruder_df)
            trial_time = pd.to_numeric(resident_df["Trial time"]).to_numpy()

            features = extract_feature_set([combined_df])[0][0]

            out_df = pd.DataFrame(features, columns=FEATURE_NAMES)
            out_df.insert(0, "frame", range(len(out_df)))
            out_df.insert(1, "trial_time", trial_time)
            out_df.insert(2, "behavior", resident_frames[0])
            out_df.insert(3, "intruder_behavior", intruder_frames[0])

            out_path = out_subdir / f"{slug}.csv"
            out_df.to_csv(out_path, index=False)
            total += 1
            print(f"wrote {out_path} ({len(out_df)} frames)")

    print(f"\nwrote {total} feature CSVs under {OUT_DIR}")


if __name__ == "__main__":
    import sys
    # optional group folders to (re)export, e.g. `python3 export_features_csv.py Opto`
    main(only_groups=sys.argv[1:])
