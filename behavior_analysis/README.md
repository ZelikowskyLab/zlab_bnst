# Resident–intruder behavior  (Figures 1 and 3)

Turns EthoVision tracking and manual behavior scoring into per-frame
kinematic features, which then feed the downstream analyses (UMAP +
watershed clustering, and the attack-approach PCA).

All scripts read it from `DATA_ROOT` in
`pipeline_paths.py`; set the `MINISCOPE_DATA_ROOT` environment variable where it
lives somewhere else. Figures save to `figures/` in this folder,
which git ignores. Run scripts and notebooks from this folder.

Still in `Desktop/miniscope`, and not expected to run until moved here:
`organize_group_folders.py` and the t-SNE notebooks.

| Step | What | Script(s) | Output |
|---|---|---|---|
| 1 | [Data organization](#1-data-organization) | `convert_xlsx_to_csv.py` | per-mouse tracking CSVs + scoring CSVs next to each `.xlsx` |
| 2 | [Feature extraction](#2-feature-extraction) | `export_features_csv.py`, `extract_features.py` | one per-frame feature CSV per trial in `data/features_export/` |
| 3 | [UMAP + watershed Figure 1](#3-figure-1----umap--watershed-all-frames) | `run_umap.py` → `explore_watershed.ipynb` → `umap_figures.ipynb` | UMAP run folders in `data/umap_results/`; figures in `figures/` |
| 4 | [Attack-approach t-SNE Figure 3](#4-attack-approach-t-sne-in-progress) | *in progress* | |

---

# 1. Data organization

Group letters are **blinded** (assigned from
`data/Ethovision/Blinded_Behavior_fs_sic_c1andc2.csv`; the unblinding key is
kept separately and not used by any script).

| Group | Folder (`data/Ethovision/…`) | Experiment | Scoring software | Trials (scored) |
|---|---|---|---|---|
| A | `Updated-C1/A` | cohort 1, `fs_sic_behavior` | Noldus Observer |  9 |
| C | `Updated-C1/C` | cohort 1, `fs_sic_behavior` | Noldus Observer |  12 |
| B | `Updated-C2/B` | cohort 2, `fs_sic_behavior_c2` | EthoVision manual scoring |  12 |
| Opto | `Opto` | `fs_dbnst_opto_pan` (not blinded) | Noldus Observer | 8 |

Each group folder has a tracking and a scoring subfolder, with every `.xlsx`
next to the CSVs made from it:

```
<group>/tracking/   Raw data-…-Trial N.xlsx  ->  …_subject1.csv (resident), …_subject2.csv (intruder)
<group>/scoring/    scoring export .xlsx     ->  .csv
```

Opto's subfolders are named `opto_etho_tracking/` and `opto-scored/`
(mapped in `SUBFOLDER_NAMES`, `export_all_tracking_widgets.py`).
Opto is exported with the other groups but not used in the 
  UMAP analysis. 

Tracking and scoring are paired by the `Trial N` in their file names. 

### Convert Excel → CSV

```bash
python3 convert_xlsx_to_csv.py              # check only: compare with the CSVs on disk, writes nothing
python3 convert_xlsx_to_csv.py --write      # write CSVs that are missing or different
```

---

# 2. Feature extraction

```bash
python3 export_features_csv.py              # all groups, including Opto
python3 export_features_csv.py Opto         # only the listed group folders
```

Output: `data/features_export/<group folder>/<trial>.csv`, one row per
tracking frame. **Nothing is cropped or dropped.** This includes the
stretch at the start of each trial (~2 min) before the intruder is put in,
where the intruder's position is held at one value; filter on `trial_time`
downstream if needed.

The processing (functions in `extract_features.py`) runs per trial:

## 2a. Clean the tracking (`interpolate_all_points`)
1. **Center point interpolated first.** 
2. **Remove jumps** (`sequential_despike`). A point is dropped if reaching
   it from the *last accepted* point would require moving faster than
   **60 cm/s**.
3. **Fill gaps.**
   - Center: linear interpolation (`fill_missing`). Gaps at the start or end
     are filled with the nearest real value.
   - Nose and tail (`relative_fill`): the *offset from the center* is
     interpolated, then added back onto the center's path. That way a
     filled nose or tail follows the body's real movement during a gap,
     rather than assuming the mouse held still.

### 2b. Label each frame (`extract_behavior_frames`)

- Each `state stop` is matched to the `state start` with the **same
  behavior name**, since scorers often overlap behaviors.
- **`behavior`** (resident): when behaviors overlap, the **most recently
  started** one is the label. When it stops, a longer behavior that is
  still active (e.g. Thigmotaxis) becomes the label again.
- **`intruder_behavior`**: `White Mouse Aggression` /
  `White Mouse Behavior of Interest`, kept in their own column so they
  never hide the resident's behavior.
- Blank where nothing is scored.
- Spellings that differ between cohorts are merged: `Self-Grooming` →
  `Self Grooming`, `Tail-rattling` → `Tail Rattling`,
  `Aggressive Grooming Barbing` → `Aggressive Grooming/Barbing`.
  `Thigmotaxis` and `Thigmotaxis (without freezing)` 
  are deliberately kept separate.
- The script stops with an error if a behavior stops without starting,
  starts twice, or never stops.

### 2c. Compute features (`extract_feature_set`)

Velocities and accelerations use a Savitzky–Golay filter (window 25 frames,
polynomial order 3). "Resident" = subject 1, "intruder" = subject 2.

| Columns | Features |
|---|---|
| `frame`, `trial_time`, `behavior`, `intruder_behavior` | frame index, time (s), labels from 2b |
| `resident_/intruder_` + `nose/center/tail_velocity` | speed of each point |
| `resident_/intruder_` + `nose/center/tail_acceleration` | second derivative of each point's position, reported as a magnitude |
| `resident_body_angle`, `intruder_body_angle` | angle between nose→center and tail→center (degrees; 180 = straight body) |
| `resident_nose_radial_velocity`, `intruder_nose_radial_velocity` | nose moving toward or away from its own center (stretching/contracting) |
| `resident_nose_tail_distance`, `intruder_nose_tail_distance` | body length |
| `intermouse_nose_{nose,center,tail}_distance` | resident nose to the intruder's nose / center / tail |
| `intermouse_nose_{nose,center,tail}_velocity` / `_acceleration` | rate of change of those distances |
| `intermouse_nose_{nose,center,tail}_radial_velocity` | relative velocity along the line between the two points |
| `relative_head_angle` | angle between the two mice's heading (center→nose) vectors |
| `resident_nose_relative_to_intruder_x`, `_y` | resident's nose in the intruder's body frame: intruder center at (0, 0), intruder facing +y (cm) |


---

# 3. Figure 1 -  UMAP + watershed 

Embeds every frame's 33 features in 2D, splits the map into behavioral
clusters, and compares how groups A, B and C are spread across them. Opto
is not included. The slow UMAP runs once (step 3a); its clusters were explored
with different settings (step 3b), and the chosen setting makes the figures (step 3c).

## 3a. UMAP embedding (`run_umap.py`)

```bash
python3 run_umap.py --drop-unannotated                                     # defaults, all frames
python3 run_umap.py --num-samples 50000                  # quick pass on a random subsample
python3 run_umap.py --n-neighbors 30 --min-dist 0.05 --tag test
```

1. Load every trial's feature CSV for the groups in `GROUPS`. `--drop-unannotated` keeps only frames with a resident `behavior` scored.
2. Z-score each feature, so features measured in different units carry
   equal weight (`--no-zscore` to skip).
3. Fit UMAP: parameters used were `--n-neighbors` 50, `--min-dist` 0.001, `--metric` euclidean
   (defaults).

**Output:** a new folder per run, never overwritten:
`data/umap_results/<date_time>_nn<n_neighbors>_md<min_dist>_<metric>_<n>d[_tag]/`

| File | Contents |
|---|---|
| `config.json` | every parameter used |
| `point_metadata.csv` | per frame: group, trial, `behavior`, `intruder_behavior`, frame, `trial_time`, `umap_x`, `umap_y`. `trial` + `frame` locate the frame's row in `data/features_export/`. |
| `umap_by_group.png`, `umap_by_behavior.png` | embedding colored by group / resident behavior |

## 3b. Choose watershed settings (`explore_watershed.ipynb`)
watershed clustering  (`run_watershed`):
1. Bin the embedding into a 2D histogram (`bins`, default 200) and smooth it
   with a Gaussian (`gaussian_stddev`, in bins).
2. Find the tallest local peaks of the smoothed density, up to
   `num_clusters`. Each peak seeds one cluster, numbered from the tallest
   (1) down. You get fewer clusters if there are fewer peaks.
3. Split the map with a watershed on the **inverted** density
   (`compactness`). Each cluster grows downhill from its peak, and
   boundaries fall in the low-density valleys between peaks. Each frame
   gets the cluster of its bin.

Evaluate results in Interactive `explore_watershed.ipynb`. Point `RESULTS_DIR` at a UMAP run folder, then move the
sliders (`bins`, `gaussian_stddev`, `num_clusters`, `compactness`). Each
setting redraws the clusters, the density with cluster boundaries, the
behavior map, and a cluster × feature heatmap.
Final settings used are put into `CHOSEN` in `umap_figures.ipynb`.


## 3c. Figures (`umap_figures.ipynb`, `umap_clusters.py` )

`umap_clusters.py` holds the watershed, occupancy,
behavior-family colors, group names/colors and plotting helpers

`umap_figures.ipynb` with chosen settings (currently 200 bins, smoothing 1.5,
clusters, compactness 0.05)  It clusters
the embedding, computes each trial's cluster occupancy, and saves to
`figures/`:

| File | Figure |
|---|---|
| `color_coded_umaps_by_group.png` | embedding colored by behavior, one panel per group |
| `barplot_cluster_occupancy.png` | per-trial cluster occupancy by group (bars + points) |
| `cluster_occupancy_significance.png` | the same, with significant (\*) and trend (p < 0.1) group differences marked |
| `density_heatmaps_groups.png` | each group's own density with the clusters |
| `density_behavior_grid.png` + `_legend.png` | density and behavior maps, one row per group, plus the full behavior legend |
| `cluster_overview.png` | clusters, density with boundaries |
| `cluster_feature_heatmap.png` | mean z-scored feature per cluster, rows ordered by similarity |

Statistics (in the notebook): a mixed ANOVA of occupancy (group ×
cluster, trials as subjects) and pingouin `pairwise_tests` within each
cluster, FDR-corrected. 




# 4. Figure 3 Attack-approach PCA 

PCA on resident approaches the intruder just before
attacking, and compares approach types between groups C, B and Opto. 


**Input:** `resident_nose_relative_to_intruder_x/_y` from the feature
export. This is the resident's nose position with the intruder's center at
(0, 0) and the intruder facing +y.

**Attack bouts:** one per resident `Attack` start in the fixed `behavior`
column, 143 in total.

| Group | Bouts | Trials with attacks |
|---|---|---|
| C | 18 | 3 of 12 |
| B | 64 | 5 of 12 (most from 3) |
| Opto | 61 | 4 of 8 (most from 3) |

**Planned steps**

| Step | Status |
|---|---|
| 1. Extract bouts (`extract_attack_bouts.py` → `data/attack_bouts/`): the 60 frames (2 s) before each attack start, with group, trial, time, both mice's cleaned positions, and flags for intruder aggression at the same time (mutual fighting) and for a previous attack inside the window | done: 143 bouts |
| 2. Approach viewer (`view_attack_bouts.ipynb`): one slider through all bouts, showing each path from the intruder's view and both mice in the arena, to check the tracking (e.g. intruder nose/tail swaps, which flip the path 180°). The title gives the trial and the length of the attack that follows | notebook written |

