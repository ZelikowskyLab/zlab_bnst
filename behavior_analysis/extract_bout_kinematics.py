"""Step 4d of the pipeline (see README.md): heading and movement of both mice
on every frame of every attack bout, as extra t-SNE features.

The approach path (resident's nose in the intruder's frame) can't tell which
way the resident is facing, or which mouse is doing the moving. This script
adds that, from the cleaned positions in data/attack_bouts/bout_frames.csv:

  data/attack_bouts/bout_kinematics.csv   one row per frame of each bout

Columns (angles in degrees, speeds in cm/s):

  resident_facing_angle     angle from the resident's heading to the line
                            toward the intruder's center. 0 = facing the
                            intruder, +/-180 = facing away; positive = the
                            intruder is to the resident's left
  intruder_facing_angle     the same for the intruder: angle from its heading
                            to the line toward the resident's center
  relative_heading          resident's heading minus intruder's heading.
                            0 = pointing the same way, +/-180 = opposite
  resident_speed            speed of each mouse's center point in the arena
  intruder_speed
  resident_approach_speed   part of each mouse's velocity directed at the
  intruder_approach_speed   other mouse's center (negative = moving away)
  resident_turn_rate        how fast each mouse's heading is changing
  intruder_turn_rate        (positive = counterclockwise, i.e. turning left)
  center_distance           distance between the two centers (cm)

Heading is the center -> nose direction, the same definition the intruder's
frame uses. Positions are smoothed lightly before taking derivatives.

Run directly:

    python3 extract_bout_kinematics.py
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from extract_attack_bouts import BOUTS_DIR

SMOOTH_WINDOW = 7   # frames (~0.23 s), Savitzky-Golay; also used for the derivatives
SMOOTH_POLY = 2


def wrap_degrees(angle):
    """Wrap an angle (or difference of angles) in degrees into (-180, 180]."""
    return -((-angle + 180) % 360 - 180)


def bout_kinematics(frames):
    """Kinematic features for one bout (frames in time order)."""
    dt = float(np.median(np.diff(frames["trial_time"].to_numpy())))

    def xy(mouse, point):
        return frames[[f"{mouse}_x_{point}", f"{mouse}_y_{point}"]].to_numpy(dtype=float)

    def smooth(a, deriv=0):
        return savgol_filter(a, SMOOTH_WINDOW, SMOOTH_POLY, deriv=deriv, delta=dt, axis=0)

    out = {}
    centers, velocities, headings = {}, {}, {}
    for mouse in ("resident", "intruder"):
        center, nose = smooth(xy(mouse, "center")), smooth(xy(mouse, "nose"))
        centers[mouse] = center
        velocities[mouse] = smooth(xy(mouse, "center"), deriv=1)          # cm/s
        heading_vec = nose - center
        headings[mouse] = np.degrees(np.arctan2(heading_vec[:, 1], heading_vec[:, 0]))   # arena angle
        out[f"{mouse}_speed"] = np.linalg.norm(velocities[mouse], axis=1)
        # turn rate from the unwrapped heading, so a 179 -> -179 crossing isn't a 358-degree turn
        unwrapped = np.degrees(np.unwrap(np.radians(headings[mouse])))
        out[f"{mouse}_turn_rate"] = savgol_filter(unwrapped, SMOOTH_WINDOW, SMOOTH_POLY, deriv=1, delta=dt)

    to_intruder = centers["intruder"] - centers["resident"]
    distance = np.linalg.norm(to_intruder, axis=1)
    unit = to_intruder / np.where(distance == 0, 1e-8, distance)[:, None]
    bearing = np.degrees(np.arctan2(to_intruder[:, 1], to_intruder[:, 0]))

    out["resident_facing_angle"] = wrap_degrees(bearing - headings["resident"])
    out["intruder_facing_angle"] = wrap_degrees(bearing + 180 - headings["intruder"])   # bearing + 180: from intruder to resident
    out["relative_heading"] = wrap_degrees(headings["resident"] - headings["intruder"])
    out["resident_approach_speed"] = np.sum(velocities["resident"] * unit, axis=1)
    out["intruder_approach_speed"] = np.sum(velocities["intruder"] * -unit, axis=1)
    out["center_distance"] = distance

    columns = ["resident_facing_angle", "intruder_facing_angle", "relative_heading", "resident_speed", "intruder_speed",
               "resident_approach_speed", "intruder_approach_speed",
               "resident_turn_rate", "intruder_turn_rate", "center_distance"]
    result = pd.DataFrame({c: out[c] for c in columns})
    result.insert(0, "frame_in_window", frames["frame_in_window"].to_numpy())
    result.insert(0, "bout_id", frames["bout_id"].to_numpy())
    return result


def main():
    bout_frames = pd.read_csv(BOUTS_DIR / "bout_frames.csv")
    kinematics = pd.concat(
        [bout_kinematics(frames.sort_values("frame_in_window")) for _, frames in bout_frames.groupby("bout_id", sort=True)],
        ignore_index=True,
    )
    kinematics.to_csv(BOUTS_DIR / "bout_kinematics.csv", index=False)
    print(f"wrote {kinematics['bout_id'].nunique()} bouts x {kinematics.groupby('bout_id').size().iloc[0]} frames "
          f"to {BOUTS_DIR / 'bout_kinematics.csv'}")
    return kinematics


if __name__ == "__main__":
    main()
