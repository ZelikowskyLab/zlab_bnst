import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sb
from load_traces import build_intervals,load_aligned_act

def filter_intervals(intervals, beh_col, beh_type, min_length=2, pre_window=0, all_intervals=None):
    mask = (intervals[beh_col] == beh_type) & ((intervals['end'] - intervals['start']) >= min_length)
    filtered = intervals[mask]

    if pre_window > 0 and all_intervals is not None:
        others = all_intervals[all_intervals[beh_col] != beh_type]
        def clean_pre(row):
            return not ((others['end'] > row['start'] - pre_window) & (others['start'] < row['start'])).any()
        filtered = filtered[filtered.apply(clean_pre, axis=1)]

    return filtered


def plot_heatmap_eta(act, intervals, tuned_cells, ax, window=(-2, 2), fps=30):
    ts = act.index.get_level_values('timestamp')
    common_t = np.linspace(window[0], window[1], int((window[1] - window[0]) * fps) + 1)
    bouts = []
    for _, row in intervals.iterrows():
        onset   = pd.Timedelta(seconds=row['start'])
        t_start = onset + pd.Timedelta(seconds=window[0])
        t_end   = onset + pd.Timedelta(seconds=window[1])
        mask    = (ts >= t_start) & (ts <= t_end)
        t_sec   = (ts[mask] - onset).total_seconds().to_numpy()
        data    = act[mask][tuned_cells].to_numpy()
        interp  = np.column_stack([np.interp(common_t, t_sec, data[:, i])
                                   for i in range(data.shape[1])])
        bouts.append(interp)

    if not bouts:
        return

    avg = np.mean(bouts, axis=0)  # (n_timepoints, n_cells)
    avg_df = pd.DataFrame(avg.T, index=tuned_cells, columns=common_t.round(3))
    avg_df = avg_df.loc[avg_df.idxmax(axis=1).sort_values().index]

    sb.heatmap(avg_df, cmap='RdBu_r', center=0, ax=ax)

def plot_cell_bouts(act, intervals, cell, ax, window=(-2, 2), fps=30):
    ts = act.index.get_level_values('timestamp')
    common_t = np.linspace(window[0], window[1], int((window[1] - window[0]) * fps) + 1)
    bouts = []
    for _, row in intervals.iterrows():
        onset   = pd.Timedelta(seconds=row['start'])
        mask    = (ts >= onset + pd.Timedelta(seconds=window[0])) & (ts <= onset + pd.Timedelta(seconds=window[1]))
        t_sec   = (ts[mask] - onset).total_seconds().to_numpy()
        data    = act[mask][cell].to_numpy()
        bouts.append(np.interp(common_t, t_sec, data))

    if not bouts:
        return

    mat = np.vstack(bouts)  # (n_bouts, n_timepoints)
    tick_step = fps // 2
    xtick_idx = np.arange(0, len(common_t), tick_step)
    sb.heatmap(mat, cmap='RdBu_r', center=0, ax=ax,
               xticklabels=False, yticklabels=np.arange(1, len(bouts) + 1))
    ax.set_xticks(xtick_idx)
    ax.set_xticklabels(common_t[xtick_idx].round(1), rotation=0)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Bout')
    ax.axvline(x=np.searchsorted(common_t, 0), color='white', linewidth=1, linestyle='--')


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
    return tuned 
    
def main():
    CATEGORIES = ['Aggressive', 'Pro-Social', 'Social Avoidance']

    data_dir = '/Users/amonast/Desktop/miniscope/data'
    traces_dir = '/Users/amonast/Desktop/miniscope/Traces'
    behavior_csv = '/Users/amonast/Desktop/miniscope/data/intervals_allmice.csv'
    info_csv = '/Users/amonast/Desktop/miniscope/dbnst_data_info.csv'
    info = pd.read_csv(info_csv)
    animals = info['Mouse'].unique()
    sessions = ['R1','R2']
    
    tuned = get_tuned_df(animals,data_dir,sessions=sessions)
    
    with PdfPages('bout_heatmaps.pdf') as pdf:
        for ani, session in product(animals, sessions):
            act = load_aligned_act(ani, session, traces_dir, behavior_csv, info_csv)
            intervals = build_intervals(ani, session, behavior_csv)

            ani_tuned = (tuned[(tuned['animal']==ani) & (tuned['session']==session)]
                         .set_index('behavior_type')['tuned_cells'])
            present_cats = set(intervals['behavior_type'].unique())
            cats_to_plot = [(cat, ani_tuned[cat]) for cat in CATEGORIES
                            if cat in present_cats and cat in ani_tuned.index and ani_tuned[cat]]

            for cat, tuned_cells in cats_to_plot:
                filtered_beh = filter_intervals(intervals, 'behavior_type', cat,
                                                min_length=1, pre_window=2, all_intervals=intervals)
                for cell in tuned_cells:
                    n_bouts = len(filtered_beh)
                    fig, ax = plt.subplots(figsize=(8, max(2, n_bouts * 0.4)))
                    fig.suptitle(f'{ani} {session}  |  {cat}  |  Cell {cell}')
                    plot_cell_bouts(act, filtered_beh, cell, ax)
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)
        
        
if __name__=='__main__':
    main()