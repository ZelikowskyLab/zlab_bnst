"""Where the data is and how trials are found: shared by every pipeline
script (see README.md), so they all agree on the groups, folders and
tracking/scoring pairing.

  - DATA_ROOT, ETHO_ROOT: the data folder (outside the repo) and data/Ethovision in it
  - GROUPS: the blinded groups that the UMAP uses
  - group_subfolders, find_trial_pairs, find_trial_files: locate each
    group's tracking and scoring CSVs and pair them by trial number
  - file_slug, trial_label: names for a trial's outputs

(Moved here from export_all_tracking_widgets.py, unchanged.)
"""

import glob
import os
import re
from pathlib import Path

from natsort import natsorted

from extract_features import align_by_trial, _trial_number

# The data stays outside the repo. Set MINISCOPE_DATA_ROOT where it lives elsewhere (e.g. the server).
DATA_ROOT = Path(os.environ.get("MINISCOPE_DATA_ROOT", "/Users/amonast/Desktop/miniscope/data"))
ETHO_ROOT = DATA_ROOT / "Ethovision"

GROUPS = [
    ("Updated-C1/A", "Group A"),
    ("Updated-C1/C", "Group C"),
    ("Updated-C2/B", "Group B"),
    ("Updated-C2/C", "Group C"),
]

# Every group keeps tracking and scoring in separate subfolders (see
# organize_group_folders.py); opto's were named before that convention.
SUBFOLDER_NAMES = {
    "Opto": ("opto_etho_tracking", "opto-scored"),
}


def group_subfolders(group_folder):
    """(tracking_dir, scoring_dir) for a group folder under ETHO_ROOT."""
    tracking_name, scoring_name = SUBFOLDER_NAMES.get(group_folder, ("tracking", "scoring"))
    return ETHO_ROOT / group_folder / tracking_name, ETHO_ROOT / group_folder / scoring_name


def find_trial_pairs(group_folder):
    """Returns (tracking_files, behavior_files) for a group, aligned by trial
    number so index i in each list is the same trial and both lists are the
    same length. A trial missing either file is skipped."""
    tracking_dir, scoring_dir = group_subfolders(group_folder)
    s1_files_all = natsorted(glob.glob(str(tracking_dir / "*_subject1.csv")))
    behavior_files_all = natsorted(glob.glob(str(scoring_dir / "*.csv")))
    return align_by_trial(s1_files_all, behavior_files_all)


def find_trial_files(group_folder):
    """Just the tracking files, for callers that don't need behavior data."""
    s1_files, _ = find_trial_pairs(group_folder)
    return s1_files


def trial_label(path):
    return f"Trial {_trial_number(path)}"


def file_slug(path):
    """Filesystem-safe name for a trial's outputs, from its tracking filename
    without the per-mouse "_subject1" suffix."""
    stem = Path(path).stem.replace("_subject1", "")
    slug = re.sub(r"\s+", "_", stem.strip())
    slug = re.sub(r"[^A-Za-z0-9_-]", "", slug)
    return slug.lower()
