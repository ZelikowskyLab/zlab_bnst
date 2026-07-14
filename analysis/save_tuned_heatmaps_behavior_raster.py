from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from itertools import product
from load_traces import load_aligned_act, build_intervals
from scipy.ndimage import gaussian_filter1d

def get_first_prosocial_window(intervals: pd.DataFrame,
                               act: pd.DataFrame,
                               window_sec: float,
                               pre_sec: float = 0.0) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Find the first Pro-Social interval onset, extract pre_sec before and
    window_sec after onset. Returns (t_rel, T_window x N_cells) or (None, None).
    """
    prosocial = intervals[intervals["behavior_type"] == "Pro-Social"].sort_values("start")
    if prosocial.empty:
        return None, None

    onset   = prosocial.iloc[0]["start"]
    t_start = onset - pre_sec
    t_end   = onset + window_sec

    cell_cols  = [c for c in act.columns if str(c).isdigit() or isinstance(c, (int, np.integer))]
    window_act = act[(act.index >= t_start) & (act.index < t_end)][cell_cols]

    t_rel = window_act.index.to_numpy() - onset
    return t_rel, window_act.values

def get_tuned_df(animals,data_dir,sessions=['R1','R2']):
    index_cols = ['animal', 'session', 'behavior_type']
    rows = []
    for ani, session in product(animals, sessions):
        pf = f"{data_dir}/p_values_{ani}_{session}.csv"
        zf = f"{data_dir}/z_scores_{ani}_{session}.csv"
        df_p = pd.read_csv(pf).set_index(index_cols)
        df_z = pd.read_csv(zf).set_index(index_cols)

        chunk = pd.concat({'pval': df_p, 'zscore': df_z}, names=['metric'])
        chunk = chunk.reorder_levels(['animal', 'session', 'behavior_type', 'metric'])
        rows.append(chunk)
    df = pd.concat(rows).sort_index()

    sig = df.xs('pval', level='metric').le(0.05)
    pos = df.xs('zscore', level='metric').gt(0)
    tuned = ((sig & pos).apply(lambda row: row.index[row].tolist(), axis=1)
                .rename('tuned_cells')
                .reset_index())

    ## exlcude these animals cells as aggessive tuned b/c one was the intruder mouse and one only had one tiny bout 
    tuned = tuned[~((tuned["animal"].isin(["C2M4", "C2F1"])) & (tuned["behavior_type"] == "Aggressive"))]
    return tuned,sig
    
def main():
    matplotlib.rcParams['svg.fonttype']='none'

    color_map = {
        'Aggressive':        "#fe0806",
        'Pro-Social':        "#55a0fb",
        'Social Avoidance':  '#b856d7',
        'Asocial':             '#e7e6e6',}
    traces_dir    = '/Users/amonast/Desktop/miniscope/Traces'
    behavior_csv  = '/Users/amonast/Desktop/miniscope/data/intervals_allmice.csv'
    data_info_csv = '/Users/amonast/Desktop/miniscope/dbnst_data_info.csv'
    data_dir      = '/Users/amonast/Desktop/miniscope/data'
    animals       = pd.read_csv(data_info_csv)['Mouse'].unique()
    
    pre_sec_ind     = 30
    window_sec_ind  = 120
    n_points_ind    = 2000
    sort_window_ind = 30.0
    t_common_ind    = np.linspace(-pre_sec_ind, window_sec_ind, n_points_ind)

    def time_to_pix(t):
        return np.interp(t, t_common_ind, np.arange(n_points_ind))

    tuned,df = get_tuned_df(animals, data_dir)

    for ani in tuned['animal'].unique():
        ani_sessions = sorted(tuned[tuned['animal'] == ani]['session'].unique())
        with PdfPages(f'figures/{ani}_heatmap.pdf') as pdf:
            for session in ani_sessions:
                act_ani = load_aligned_act(ani, session, traces_dir, behavior_csv, data_info_csv)
                intervals_ani = build_intervals(ani, session, behavior_csv)

                act_sec = act_ani.loc[(ani, session)].copy()
                act_sec.index = act_sec.index.total_seconds()
                cell_cols_ani = [c for c in act_sec.columns if str(c).isdigit() or isinstance(c, (int, np.integer))]
                act_sec[cell_cols_ani] = gaussian_filter1d(act_sec[cell_cols_ani].to_numpy(), sigma=4, axis=0)

                t_rel, window = get_first_prosocial_window(intervals_ani, act_sec,
                                                           window_sec=window_sec_ind, pre_sec=pre_sec_ind)
                if t_rel is None:
                    print(f'{ani} {session}: no Pro-Social onset, skipping')
                    continue

                onset = intervals_ani[intervals_ani['behavior_type'] == 'Pro-Social'].sort_values('start').iloc[0]['start']

                interp = np.vstack([
                    np.interp(t_common_ind, t_rel, window[:, i])
                    for i in range(window.shape[1])
                ])

                sort_mask  = (t_common_ind >= 0) & (t_common_ind <= sort_window_ind)
                sort_order = np.argsort(interp[:, sort_mask].sum(axis=1))[::-1]
                sorted_mat = interp[sort_order]

                ivs_rel = intervals_ani.copy()
                ivs_rel['start_rel'] = ivs_rel['start'] - onset
                ivs_rel['end_rel']   = ivs_rel['end']   - onset
                ivs_window = ivs_rel[
                    (ivs_rel['end_rel']   > -pre_sec_ind) &
                    (ivs_rel['start_rel'] < window_sec_ind) &
                    (ivs_rel['behavior_type'].isin(color_map))
                ]

                # ── Layout: behavior | heatmap | colorbar ─────────────────────────
                fig = plt.figure(figsize=(11, 6))
                gs  = GridSpec(2, 2, figure=fig,
                               height_ratios=[1, 8], width_ratios=[20, 1],
                               hspace=0.05, wspace=0.3)
                ax_heat = fig.add_subplot(gs[1, 0])
                ax_beh  = fig.add_subplot(gs[0, 0], sharex=ax_heat)
                ax_cbar = fig.add_subplot(gs[1, 1])

                # ── Heatmap ───────────────────────────────────────────────────────
                n_ticks     = 7
                tick_idx    = np.linspace(0, n_points_ind - 1, n_ticks, dtype=int)
                tick_labels = [f'{t_common_ind[i]:.0f}' for i in tick_idx]

                sns.heatmap(sorted_mat, ax=ax_heat, cmap='inferno', 
                            xticklabels=False, vmin=0,vmax=6, yticklabels=False, cbar=False)
                fig.colorbar(ax_heat.collections[0], cax=ax_cbar, label='z-score')

                ax_heat.set_xticks(tick_idx)
                ax_heat.set_xticklabels(tick_labels, rotation=0)
                onset_pix = time_to_pix(0)
                ax_heat.axvline(onset_pix, color='black', linewidth=1, linestyle='--')
                ax_heat.set_xlabel('Time relative to first Pro-Social onset (s)')
                ax_heat.set_ylabel('Cell (sorted by 0–30s activity)')

                # ── Behavior raster ───────────────────────────────────────────────
                seen = set()
                for _, row in ivs_window.iterrows():
                    bt = row['behavior_type']
                    x0 = time_to_pix(row['start_rel'])
                    x1 = time_to_pix(row['end_rel'])
                    ax_beh.barh(y=0, width=x1 - x0, left=x0, height=0.8,
                                color=color_map[bt], label=bt if bt not in seen else '', alpha=0.8)
                    seen.add(bt)
                ax_beh.axvline(onset_pix, color='black', linewidth=1, linestyle='--')
                ax_beh.set_ylim([-0.5, 0.5])
                ax_beh.set_yticks([])
                plt.setp(ax_beh.get_xticklabels(), visible=False)
                ax_beh.set_ylabel('Behavior', fontsize=8, rotation=0, labelpad=45, va='center')
                ax_beh.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
                ax_beh.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)

                fig.suptitle(f'{ani} {session} — n={sorted_mat.shape[0]} cells')
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

if __name__=='__main__':
    main()