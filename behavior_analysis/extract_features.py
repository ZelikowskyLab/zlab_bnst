### Imports ###

import numpy as np
import pandas as pd
from natsort import natsorted
import glob
import re
import pywt
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler


### Portable Helpers ###

def _recover_numeric_columns(df, max_bad_fraction=0.05):
    """CSV round-tripping loses Excel's per-cell type info: a column that
    openpyxl parsed cleanly as numbers (tolerating a stray non-numeric cell,
    e.g. a units-row artifact like "s" left over from a multi-row header)
    can come back from pd.read_csv as strings (object dtype), because a CSV
    reader picks one dtype for the whole column and a single bad cell forces
    it to fall back to text. Recover those columns explicitly: a column is
    converted to numeric if at most a small fraction of its non-null values
    fail to parse (those become NaN), so genuinely text columns (e.g.
    "Behavior", which fails almost entirely) are left untouched."""
    for col in df.columns:
        if df[col].dtype == object:
            non_null = df[col].notna().sum()
            if non_null == 0:
                continue
            coerced = pd.to_numeric(df[col], errors="coerce")
            newly_failed = coerced.isna().sum() - df[col].isna().sum()
            if newly_failed / non_null <= max_bad_fraction:
                df[col] = coerced
    return df

def read_excel_cached(path, sheet_name=0, header=0, na_values=None):
    """Reads one Excel sheet, caching the parsed result as a CSV next to the
    source file. openpyxl parses .xlsx cell-by-cell and is slow on large
    sheets; pandas' CSV reader is much faster, so repeat runs (e.g. while
    debugging) skip re-parsing Excel entirely once the cache exists."""
    src = Path(path)
    cache_path = src.with_name(f"{src.stem}__sheet{sheet_name}.csv")

    if cache_path.exists():
        return _recover_numeric_columns(pd.read_csv(cache_path, low_memory=False))

    df = pd.read_excel(path, sheet_name=sheet_name, header=header, na_values=na_values)
    df.to_csv(cache_path, index=False)
    return df

def _trial_number(path):
    """Extracts the numeric trial id from an Ethovision/Observer export filename."""
    match = re.search(r"Trial\s*[_]?\s*(\d+)", Path(path).stem)
    if not match:
        raise ValueError(f"Could not parse a trial number from {path}")
    return int(match.group(1))

def align_by_trial(tracking_s1_files, behavior_files):
    """Tracking and behavior exports aren't always available for the same
    set of trials (a trial can be tracked before it's scored, or vice
    versa). Positionally zipping two independently-globbed lists would
    silently pair the wrong behavior with the wrong tracking trial whenever
    one side is missing a file. Instead, key both lists by trial number and
    keep only the trials present in both, so the returned lists are
    guaranteed to line up index-for-index."""
    tracking_by_trial = {_trial_number(f): f for f in tracking_s1_files}
    behavior_by_trial = {_trial_number(f): f for f in behavior_files}
    common_trials = natsorted(set(tracking_by_trial) & set(behavior_by_trial))
    missing = len(tracking_by_trial) - len(common_trials)
    if missing:
        print(f"Note: {missing} tracked trial(s) have no matching behavior file yet and will be skipped")
    return (
        [tracking_by_trial[t] for t in common_trials],
        [behavior_by_trial[t] for t in common_trials],
    )

# interpolate helper from https://docs.sleap.ai/latest/notebooks/Analysis_examples/#loading-the-data
def fill_missing(Y, kind="linear"):
    """Fills missing values independently along each dimension after the first."""
    # Store initial shape.
    initial_shape = Y.shape
    # Flatten after first dim.
    Y = Y.reshape((initial_shape[0], -1))
    # Interpolate along each slice.
    for i in range(Y.shape[-1]):
        y = Y[:, i]
        # Build interpolant.
        x = np.flatnonzero(~np.isnan(y))
        f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)
        # Fill missing
        xq = np.flatnonzero(np.isnan(y))
        y[xq] = f(xq)
        # Fill leading or trailing NaNs with the nearest non-NaN values
        mask = np.isnan(y)
        y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])
        # Save slice
        Y[:, i] = y
    # Restore to initial shape.
    Y = Y.reshape(initial_shape)

    return Y

def sequential_despike(xy, t, max_speed):
    """Rejects a tracked point when its implied speed from the last
    ACCEPTED point exceeds a physically-plausible max. Unlike a local-
    window filter (e.g. Hampel), comparing against the last trusted point
    -- rather than nearby frames, which may themselves be bad -- means a
    short run of consecutive false detections never accumulates enough
    influence to look "normal" to each other. Operates on paired (x, y)
    so a diagonal jump is judged as one displacement, not two independent
    per-axis ones. Rejected points are set to NaN and left for
    fill_missing to bridge, like any other tracking dropout.
    xy: [frames, 2] position array. t: [frames] timestamps (seconds).
    max_speed: implausible-speed threshold in position units / second."""
    xy = xy.copy()
    last_xy, last_t = None, None
    for i in range(xy.shape[0]):
        if np.isnan(xy[i, 0]) or np.isnan(xy[i, 1]):
            continue
        if last_xy is None:
            last_xy, last_t = xy[i], t[i]
            continue
        dt = t[i] - last_t
        speed = np.linalg.norm(xy[i] - last_xy) / dt if dt > 0 else np.inf
        if speed > max_speed:
            xy[i] = np.nan
        else:
            last_xy, last_t = xy[i], t[i]
    return xy

# velocity calculator helper from https://docs.sleap.ai/latest/notebooks/Analysis_examples/#more-advanced-visualizations
def smooth_diff(node_loc, win=25, poly=3, deriv=1):
    """
    node_loc is a [frames, 2] array

    win defines the window to smooth over

    poly defines the order of the polynomial
    to fit with

    """
    node_loc_vel = np.zeros_like(node_loc)

    for c in range(node_loc.shape[-1]):
        node_loc_vel[:, c] = savgol_filter(node_loc[:, c], win, poly, deriv=deriv)

    node_vel = np.linalg.norm(node_loc_vel,axis=1)

    return node_vel

def calculate_distance(node_loc1, node_loc2):
    """
    node_loc1 & 2 are a [frames, 2] arrays

    """
    min_frames = min(node_loc1.shape[0], node_loc2.shape[0])

    node_diff = node_loc1[:min_frames] - node_loc2[:min_frames]

    node_dist = np.linalg.norm(node_diff, axis=1)

    return node_dist

def calculate_angle(n1, c1, n2, c2):

    v1 = n1 - c1
    v2 = n2 - c2

    cos_theta = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))

    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    return np.degrees(theta)

def calculate_point_relative_to_vector(point, v_tip, v_base):

    """ Vector always points north """
    translated_point = point - v_base
    translated_tip = v_tip - v_base

    theta = np.arctan2(translated_tip[:, 1], translated_tip[:, 0])
    relative_theta = np.pi / 2 - theta

    relative_x = translated_point[:, 0] * np.cos(relative_theta) - translated_point[:, 1] * np.sin(relative_theta)
    relative_y = translated_point[:, 0] * np.sin(relative_theta) + translated_point[:, 1] * np.cos(relative_theta)

    return np.column_stack((relative_x, relative_y))

# edited from https://docs.sleap.ai/latest/notebooks/Analysis_examples/#more-advanced-visualizations
def calculate_relative_velocity(node_loc1, node_loc2, win=25, poly=3, deriv=1):

    distance = calculate_distance(node_loc1, node_loc2)

    vel_diff = np.zeros_like(distance)

    vel_diff = savgol_filter(distance, win, poly, deriv=deriv)

    return vel_diff

def calculate_radial_velocity(v_tip, v_base, win=25, poly=3, deriv=1):

    min_frames = min(v_tip.shape[0], v_base.shape[0])
    tip = v_tip[:min_frames]
    base = v_base[:min_frames]

    node_diff = tip - base

    node_dist = np.linalg.norm(node_diff, axis=1, keepdims=True)

    node_dist = np.where(node_dist == 0, 1e-8, node_dist)
    unit_vectors = node_diff / node_dist

    tip_vel = np.zeros_like(tip)
    base_vel = np.zeros_like(base)

    for c in range(2):
        tip_vel[:, c] = savgol_filter(tip[:, c], win, poly, deriv=deriv)
        base_vel[:, c] = savgol_filter(base[:, c], win, poly, deriv=deriv)

    rel_vel = tip_vel - base_vel

    radial_vel = np.sum(rel_vel * unit_vectors, axis=1)

    return radial_vel


### Interpolate Missing Data ###

# interpolate missing data - nose to center per each mouse
def relative_fill(target_despiked_xy, anchor_x_filled, anchor_y_filled):
    """Interpolates (target - anchor) instead of target's absolute position,
    then reconstructs target = anchor + filled_offset. anchor must already
    be a complete (despiked + filled) trajectory. Used for nose/tail, which
    are anchored to center: a straight-line interpolation of raw absolute
    position assumes the body held still during a gap, but center -- far
    more reliably tracked -- usually captures real motion through the same
    gap, so reconstructing via the offset lets nose/tail inherit it."""
    offset_x = target_despiked_xy[:, 0] - anchor_x_filled
    offset_y = target_despiked_xy[:, 1] - anchor_y_filled
    offset_x_filled = fill_missing(offset_x.copy())
    offset_y_filled = fill_missing(offset_y.copy())
    return anchor_x_filled + offset_x_filled, anchor_y_filled + offset_y_filled


def interpolate_all_points(etho_subject_dfs, max_speed=60.0):
    for etho_subject_df in etho_subject_dfs:

        t = pd.to_numeric(etho_subject_df["Trial time"]).to_numpy()

        # Center is cleaned first and independently -- it's the most
        # reliably tracked point, and nose/tail both anchor to its already-
        # complete trajectory below.
        center_xy = np.column_stack([
            pd.to_numeric(etho_subject_df["X center"]).to_numpy(),
            pd.to_numeric(etho_subject_df["Y center"]).to_numpy(),
        ])
        center_xy = sequential_despike(center_xy, t, max_speed=max_speed)
        center_x_filled = fill_missing(center_xy[:, 0].copy())
        center_y_filled = fill_missing(center_xy[:, 1].copy())
        etho_subject_df["X center"] = center_x_filled
        etho_subject_df["Y center"] = center_y_filled

        # Nose and tail are despiked on their own absolute position (same
        # outlier rejection as center), but gaps are filled relative to
        # center's already-complete trajectory via relative_fill, so a
        # reconstructed point inherits the body's real motion during a gap
        # instead of being interpolated as if the body held still.
        for x_col, y_col in [("X nose", "Y nose"), ("X tail", "Y tail")]:
            xy = np.column_stack([
                pd.to_numeric(etho_subject_df[x_col]).to_numpy(),
                pd.to_numeric(etho_subject_df[y_col]).to_numpy(),
            ])
            xy = sequential_despike(xy, t, max_speed=max_speed)
            x_filled, y_filled = relative_fill(xy, center_x_filled, center_y_filled)
            etho_subject_df[x_col] = x_filled
            etho_subject_df[y_col] = y_filled

    return etho_subject_dfs


### Associate Behavior Labels With Frames ###

# Behaviors scored for the intruder (white CD-1) rather than the resident.
# Both mice were scored in one event stream, so these are split out into
# their own per-frame column instead of competing with the resident's label.
INTRUDER_BEHAVIORS = {"White Mouse Aggression", "White Mouse Behavior of Interest"}

# Same behavior spelled differently between cohorts' scoring templates.
BEHAVIOR_NAME_ALIASES = {
    "Self-Grooming": "Self Grooming",
    "Tail-rattling": "Tail Rattling",
    "Aggressive Grooming Barbing": "Aggressive Grooming/Barbing",
}

def _label_change_points(behaviors, events, time_spans):
    """Walks one subject's start/stop event stream and returns the
    (timestamp, label) at every point the label changes. Stops are matched
    to starts by behavior name (not by row position), since scorers overlap
    behaviors. While several behaviors are active, the most recently started
    one is the label; when it stops, the next most recent still-active one
    resumes (e.g. Thigmotaxis continues after a nested sniff ends)."""
    active = [] # behavior names in start order; last = current label
    change_points = []
    for behavior, event, timestamp in zip(behaviors, events, time_spans):
        event = str(event).strip().lower()
        if event == "state start":
            if behavior in active:
                raise ValueError(f"'{behavior}' started at {timestamp} while already active")
            active.append(behavior)
        elif event == "state stop":
            if behavior not in active:
                raise ValueError(f"'{behavior}' stopped at {timestamp} without a matching start")
            active.remove(behavior)
        else:
            raise ValueError(f"Unexpected event type '{event}' for '{behavior}' at {timestamp}")
        change_points.append({"Timestamp": timestamp, "Behavior": active[-1] if active else np.nan})

    if active:
        raise ValueError(f"Behaviors never stopped: {active}")

    # several events can share a timestamp; only the state after the last applies
    return pd.DataFrame(change_points).drop_duplicates(subset="Timestamp", keep="last")

def extract_behavior_frames(behaviors_for_group, events_for_group, time_spans_for_group, etho_whole_subject_dfs):
    """Assigns every tracking frame a resident label and an intruder label
    from the scored start/stop events. Returns (resident_frames,
    intruder_frames): one per-frame label list per trial for each, NaN where
    nothing is scored."""

    resident_frames = []
    intruder_frames = []
    for behaviors, events, time_spans, etho_whole_subject_df in zip(behaviors_for_group, events_for_group, time_spans_for_group, etho_whole_subject_dfs):

        behaviors = pd.Series(behaviors).replace(BEHAVIOR_NAME_ALIASES).to_numpy()
        events = np.asarray(events)
        time_spans = pd.to_numeric(pd.Series(time_spans)).to_numpy()
        is_intruder = np.isin(behaviors, list(INTRUDER_BEHAVIORS))

        frames_df = pd.DataFrame({"Timestamp": (etho_whole_subject_df["Trial time"].values)})
        frames_df["Timestamp"] = pd.to_numeric(frames_df["Timestamp"], errors="coerce")

        for subject_mask, out in [(~is_intruder, resident_frames), (is_intruder, intruder_frames)]:
            if not subject_mask.any():
                out.append([np.nan] * len(frames_df))
                continue
            behaviors_df = _label_change_points(behaviors[subject_mask], events[subject_mask], time_spans[subject_mask])
            merged_df = pd.merge_asof(frames_df, behaviors_df, on="Timestamp", direction="backward")
            out.append(merged_df["Behavior"].tolist())

    return resident_frames, intruder_frames


### Extract Desired Features ###

def extract_feature_set(etho_dfs_for_group):

    etho_for_group = []
    num_frames_for_group = []

    for etho_df in etho_dfs_for_group:

        # Intramouse
        n_vel_resi = smooth_diff(etho_df[["X nose", "Y nose"]].values)
        c_vel_resi = smooth_diff(etho_df[["X center", "Y center"]].values)
        t_vel_resi = smooth_diff(etho_df[["X tail", "Y tail"]].values)
        n_vel_intr = smooth_diff(etho_df[["x nose", "y nose"]].values)
        c_vel_intr = smooth_diff(etho_df[["x center", "y center"]].values)
        t_vel_intr = smooth_diff(etho_df[["x tail", "y tail"]].values)

        n_acc_resi = smooth_diff(etho_df[["X nose", "Y nose"]].values, deriv=2)
        c_acc_resi = smooth_diff(etho_df[["X center", "Y center"]].values, deriv=2)
        t_acc_resi = smooth_diff(etho_df[["X tail", "Y tail"]].values, deriv=2)
        n_acc_intr = smooth_diff(etho_df[["x nose", "y nose"]].values, deriv=2)
        c_acc_intr = smooth_diff(etho_df[["x center", "y center"]].values, deriv=2)
        t_acc_intr = smooth_diff(etho_df[["x tail", "y tail"]].values, deriv=2)

        body_angl_resi = calculate_angle(etho_df[["X nose", "Y nose"]].values, etho_df[["X center", "Y center"]].values,
                                         etho_df[["X tail", "Y tail"]].values, etho_df[["X center", "Y center"]].values)
        body_angl_intr = calculate_angle(etho_df[["x nose", "y nose"]].values, etho_df[["x center", "y center"]].values,
                                         etho_df[["x tail", "y tail"]].values, etho_df[["x center", "y center"]].values)

        rad_vel_resi = calculate_radial_velocity(etho_df[["X nose", "Y nose"]].values, etho_df[["X center", "Y center"]].values)
        rad_vel_intr = calculate_radial_velocity(etho_df[["x nose", "y nose"]].values, etho_df[["x center", "y center"]].values)

        nt_dist_resi = calculate_distance(etho_df[["X nose", "Y nose"]].values, etho_df[["X tail", "Y tail"]].values)
        nt_dist_intr = calculate_distance(etho_df[["x nose", "y nose"]].values, etho_df[["x tail", "y tail"]].values)

        # Intermouse
        nn_dist = calculate_distance(etho_df[["X nose", "Y nose"]].values, etho_df[["x nose", "y nose"]].values)
        nc_dist = calculate_distance(etho_df[["X nose", "Y nose"]].values, etho_df[["x center", "y center"]].values)
        nt_dist = calculate_distance(etho_df[["X nose", "Y nose"]].values, etho_df[["x tail", "y tail"]].values)

        nn_vel = calculate_relative_velocity(etho_df[["X nose", "Y nose"]].values, etho_df[["x nose", "y nose"]].values)
        nc_vel = calculate_relative_velocity(etho_df[["X nose", "Y nose"]].values, etho_df[["x center", "y center"]].values)
        nt_vel = calculate_relative_velocity(etho_df[["X nose", "Y nose"]].values, etho_df[["x tail", "y tail"]].values)

        nn_acc = calculate_relative_velocity(etho_df[["X nose", "Y nose"]].values, etho_df[["x nose", "y nose"]].values, deriv=2)
        nc_acc = calculate_relative_velocity(etho_df[["X nose", "Y nose"]].values, etho_df[["x center", "y center"]].values, deriv=2)
        nt_acc = calculate_relative_velocity(etho_df[["X nose", "Y nose"]].values, etho_df[["x tail", "y tail"]].values, deriv=2)

        rel_head_angl = calculate_angle(etho_df[["X nose", "Y nose"]].values, etho_df[["X center", "Y center"]].values,
                                        etho_df[["x nose", "y nose"]].values, etho_df[["x center", "y center"]].values)

        nn_rad_vel = calculate_radial_velocity(etho_df[["X nose", "Y nose"]].values, etho_df[["x nose", "y nose"]].values)
        nc_rad_vel = calculate_radial_velocity(etho_df[["X nose", "Y nose"]].values, etho_df[["x center", "y center"]].values)
        nt_rad_vel = calculate_radial_velocity(etho_df[["X nose", "Y nose"]].values, etho_df[["x tail", "y tail"]].values)

        rel_pos = calculate_point_relative_to_vector(etho_df[["X nose", "Y nose"]].values, etho_df[["x nose", "y nose"]].values, etho_df[["x center", "y center"]].values)

        features = np.concatenate([[ n_vel_resi, c_vel_resi, t_vel_resi, n_vel_intr, c_vel_intr, t_vel_intr, n_acc_resi, c_acc_resi, t_acc_resi,
                                    n_acc_intr, c_acc_intr, t_acc_intr, body_angl_resi, body_angl_intr, rad_vel_intr, rad_vel_resi, nt_dist_resi,
                                    nt_dist_intr, nn_dist, nc_dist, nt_dist, nn_vel, nc_vel, nt_vel, nn_acc, nc_acc, nt_acc, rel_head_angl, nn_rad_vel,
                                    nc_rad_vel, nt_rad_vel, rel_pos[:, 0], rel_pos[:, 1]]]).T


        etho_for_group.append(features)
        num_frames_for_group.append(len(etho_df))

    return etho_for_group, num_frames_for_group


def main():

    group_names = ["Group A", "Group B", "Group C"] # blinded labels — groupD (opto) excluded for now

    from pipeline_paths import ETHO_ROOT   # here, not at the top: pipeline_paths imports this module

    ### List & Align Trial Files ###

    # Both behavior and tracking data are sorted into per-trial Blinded
    # Group subfolders (see Blinded_Behavior_fs_sic_c1andc2.csv); each group
    # here pulls from the blinded subfolder that corresponds to it.
    #
    # groupA/groupC are Cohort 1 (BALB/c): tracking + Blinded Group letter
    # from Updated-C1, behavior scored in Observer ("Event Logs" exports,
    # header=0, Time_Relative_sf).
    #
    # groupB is the Cohort 2 (CD-1) group, sourced ONLY from Updated-C2/B --
    # Cohort 2 used Ethovision for both tracking and scoring ("Manual
    # Scoring" exports, header=35, Trial time). Updated-C1/B is a *different*
    # NS/CD-1 subset run within the cohort-1 batch that is intentionally out
    # of scope for this pipeline (not part of the 3 groups in the original
    # study design) and is deliberately NOT pulled in here.
    #
    # groupD (opto) excluded for now -- not sorted into these folders yet.
    # each group folder holds tracking/ and scoring/ subfolders (see organize_group_folders.py)
    etho_groupA_s1_files_all = natsorted(glob.glob(f"{ETHO_ROOT}/Updated-C1/A/tracking/*_subject1.csv"))
    etho_groupC_s1_files_all = natsorted(glob.glob(f"{ETHO_ROOT}/Updated-C1/C/tracking/*_subject1.csv"))
    etho_groupB_s1_files_all = natsorted(glob.glob(f"{ETHO_ROOT}/Updated-C2/B/tracking/*_subject1.csv"))

    behavior_groupA_files_all = natsorted(glob.glob(f"{ETHO_ROOT}/Updated-C1/A/scoring/*.csv"))
    behavior_groupC_files_all = natsorted(glob.glob(f"{ETHO_ROOT}/Updated-C1/C/scoring/*.csv"))
    behavior_groupB_files_all = natsorted(glob.glob(f"{ETHO_ROOT}/Updated-C2/B/scoring/*.csv"))

    # A trial's tracking and behavior files aren't guaranteed to both be
    # present yet, so align by trial number rather than assuming the two
    # globs came back in the same order/length.
    etho_groupA_s1_files, behavior_groupA_files = align_by_trial(etho_groupA_s1_files_all, behavior_groupA_files_all)
    etho_groupC_s1_files, behavior_groupC_files = align_by_trial(etho_groupC_s1_files_all, behavior_groupC_files_all)
    etho_groupB_s1_files, behavior_groupB_files = align_by_trial(etho_groupB_s1_files_all, behavior_groupB_files_all)

    etho_groupA_s2_files = [f.replace("_subject1.csv", "_subject2.csv") for f in etho_groupA_s1_files]
    etho_groupC_s2_files = [f.replace("_subject1.csv", "_subject2.csv") for f in etho_groupC_s1_files]
    etho_groupB_s2_files = [f.replace("_subject1.csv", "_subject2.csv") for f in etho_groupB_s1_files]


    ### Parse Behavioral Annotations ###

    behavior_groupA_dfs = [pd.read_csv(f, header=0).dropna(subset=["Behavior"]) for f in behavior_groupA_files]
    behavior_groupC_dfs = [pd.read_csv(f, header=0).dropna(subset=["Behavior"]) for f in behavior_groupC_files]
    behavior_groupB_dfs = [pd.read_csv(f, header=35).dropna(subset=["Behavior"]) for f in behavior_groupB_files]


    ### Parse Ethovision Tracking Data ###

    etho_groupA_s1_dfs = [pd.read_csv(f, header=35, na_values='-', low_memory=False).drop(0).reset_index(drop=True) for f in etho_groupA_s1_files]
    etho_groupA_s2_dfs = [pd.read_csv(f, header=35, na_values='-', low_memory=False).drop(0).reset_index(drop=True) for f in etho_groupA_s2_files]
    etho_groupC_s1_dfs = [pd.read_csv(f, header=35, na_values='-', low_memory=False).drop(0).reset_index(drop=True) for f in etho_groupC_s1_files]
    etho_groupC_s2_dfs = [pd.read_csv(f, header=35, na_values='-', low_memory=False).drop(0).reset_index(drop=True) for f in etho_groupC_s2_files]
    etho_groupB_s1_dfs = [pd.read_csv(f, header=35, na_values='-', low_memory=False).drop(0).reset_index(drop=True) for f in etho_groupB_s1_files]
    etho_groupB_s2_dfs = [pd.read_csv(f, header=35, na_values='-', low_memory=False).drop(0).reset_index(drop=True) for f in etho_groupB_s2_files]


    ### Interpolate Missing Data ###

    etho_groupA_s1_dfs = interpolate_all_points(etho_groupA_s1_dfs)
    etho_groupA_s2_dfs = interpolate_all_points(etho_groupA_s2_dfs)
    etho_groupC_s1_dfs = interpolate_all_points(etho_groupC_s1_dfs)
    etho_groupC_s2_dfs = interpolate_all_points(etho_groupC_s2_dfs)
    etho_groupB_s1_dfs = interpolate_all_points(etho_groupB_s1_dfs)
    etho_groupB_s2_dfs = interpolate_all_points(etho_groupB_s2_dfs)


    ### Isolate Behavior and Timepoint Data ###

    # extract behaviors
    behaviors_groupA = []
    events_groupA = []
    time_spans_groupA = []
    for behavior_groupA_df in behavior_groupA_dfs:
        behaviors_groupA.append(behavior_groupA_df["Behavior"].values)
        events_groupA.append(behavior_groupA_df["Event_Type"].values)
        time_spans_groupA.append(behavior_groupA_df["Time_Relative_sf"].values)

    behaviors_groupC = []
    events_groupC = []
    time_spans_groupC = []
    for behavior_groupC_df in behavior_groupC_dfs:
        behaviors_groupC.append(behavior_groupC_df["Behavior"].values)
        events_groupC.append(behavior_groupC_df["Event_Type"].values)
        time_spans_groupC.append(behavior_groupC_df["Time_Relative_sf"].values)

    behaviors_groupB = []
    events_groupB = []
    time_spans_groupB = []
    for behavior_groupB_df in behavior_groupB_dfs:
        behaviors_groupB.append(behavior_groupB_df["Behavior"].values)
        events_groupB.append(behavior_groupB_df["Event"].values)
        time_spans_groupB.append(pd.to_numeric(behavior_groupB_df["Trial time"]).values)

    ### Isolate and Combine Position Data for Both Subjects ###

    etho_groupA_dfs = []
    for etho_groupA_s1_df, etho_groupA_s2_df in zip(etho_groupA_s1_dfs, etho_groupA_s2_dfs):

        etho_groupA_df = pd.concat(
            [etho_groupA_s1_df[["X nose", "Y nose", "X center", "Y center", "X tail", "Y tail"]],
             etho_groupA_s2_df[["X nose", "Y nose", "X center", "Y center", "X tail", "Y tail"]].rename(columns=str.lower)], # make lower for no duplicate names
            axis=1
            )
        etho_groupA_dfs.append(etho_groupA_df)

    etho_groupC_dfs = []
    for etho_groupC_s1_df, etho_groupC_s2_df in zip(etho_groupC_s1_dfs, etho_groupC_s2_dfs):

        etho_groupC_df = pd.concat(
            [etho_groupC_s1_df[["X nose", "Y nose", "X center", "Y center", "X tail", "Y tail"]],
             etho_groupC_s2_df[["X nose", "Y nose", "X center", "Y center", "X tail", "Y tail"]].rename(columns=str.lower)],
            axis=1
            )
        etho_groupC_dfs.append(etho_groupC_df)

    etho_groupB_dfs = []
    for etho_groupB_s1_df, etho_groupB_s2_df in zip(etho_groupB_s1_dfs, etho_groupB_s2_dfs):

        etho_groupB_df = pd.concat(
            [etho_groupB_s1_df[["X nose", "Y nose", "X center", "Y center", "X tail", "Y tail"]],
             etho_groupB_s2_df[["X nose", "Y nose", "X center", "Y center", "X tail", "Y tail"]].rename(columns=str.lower)],
            axis=1
            )
        etho_groupB_dfs.append(etho_groupB_df)


    ### Associate Behavior Labels With Frames ###

    # intruder labels aren't used by this pipeline; see export_features_csv.py
    behavior_frames_groupA, _ = extract_behavior_frames(behaviors_groupA, events_groupA, time_spans_groupA, etho_groupA_s1_dfs)
    behavior_frames_groupC, _ = extract_behavior_frames(behaviors_groupC, events_groupC, time_spans_groupC, etho_groupC_s1_dfs)
    behavior_frames_groupB, _ = extract_behavior_frames(behaviors_groupB, events_groupB, time_spans_groupB, etho_groupB_s1_dfs)

    behavior_frames_groupA = [features for trial in behavior_frames_groupA for features in trial]
    behavior_frames_groupC = [features for trial in behavior_frames_groupC for features in trial]
    behavior_frames_groupB = [features for trial in behavior_frames_groupB for features in trial]

    behavior_frames_all = np.concatenate([behavior_frames_groupA, behavior_frames_groupC, behavior_frames_groupB])

    print(np.shape(behavior_frames_all))


    ### Extract Desired Features ###

    # num_frames is for wavelet and is not needed
    etho_groupA, num_frames_groupA = extract_feature_set(etho_groupA_dfs)
    etho_groupC, num_frames_groupC = extract_feature_set(etho_groupC_dfs)
    etho_groupB, num_frames_groupB = extract_feature_set(etho_groupB_dfs)


    etho_groupA = [features for trial in etho_groupA for features in trial]
    etho_groupC = [features for trial in etho_groupC for features in trial]
    etho_groupB = [features for trial in etho_groupB for features in trial]

    etho_all = np.concatenate([etho_groupA, etho_groupC, etho_groupB])
    num_frames_all = np.concatenate([num_frames_groupA, num_frames_groupC, num_frames_groupB])

    print(np.shape(etho_all))


    ### Remove Frames With No Behavior Label ###

    behavior_frames_no_nan = []
    etho_data_no_nan = []

    for i in range(len(behavior_frames_all)):
        if(behavior_frames_all[i] != "nan"):

            etho_data_no_nan.append(etho_all[i])
            behavior_frames_no_nan.append(behavior_frames_all[i])


    etho_data_no_nan = np.array(etho_data_no_nan)
    behavior_labels, color_labels = np.unique(behavior_frames_no_nan, return_inverse=True)

    return etho_data_no_nan, behavior_frames_no_nan, behavior_labels, color_labels


if __name__ == "__main__":
    main()
