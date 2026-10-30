"""
Plot every cell's trace against an animal's behavior and save one PDF per animal,
with each session's cells grouped into its own section (separated by a divider page).

Usage:
    python plot_all_cells_by_session.py
    python plot_all_cells_by_session.py --animals C1F2 C1M1 --sessions R1 R2
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter1d

sys.path.extend(['/Users/amonast/Documents/GitHub/zlab_miniscope'])
from load_traces import load_aligned_act

matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['pdf.fonttype'] = 42

ANIMALS = ['C1F2', 'C1M1', 'C1M2', 'C2F1', 'C2M4', 'C3F3', 'C3F4',
           'C3M2', 'C3M3', 'C3M4']
SESSIONS = ['R1', 'R2']

BEHAVIOR_COLORS = {
    'Aggressive': 'red',
    'Pro-Social': 'royalblue',
    'Social Avoidance': 'purple',
    'Baseline': 'gray',
}

TRACES_DIR = Path('/Users/amonast/Desktop/miniscope/Traces')
BEHAVIOR_CSV = Path('/Users/amonast/Desktop/miniscope/data/intervals_allmice.csv')
INFO_CSV = Path('/Users/amonast/Desktop/miniscope/dbnst_data_info.csv')
OUTPUT_DIR = Path('/Users/amonast/Desktop/miniscope/figures')


def plot_cell_trace(ax, full_trace, act, title):
    time = full_trace.index.get_level_values('timestamp').total_seconds()
    smoothed = gaussian_filter1d(full_trace.to_numpy(dtype=float), sigma=3)
    ax.plot(time, smoothed, lw=1, color='k')

    patches = []
    for behavior, color in BEHAVIOR_COLORS.items():
        intervals = act[act['behavior_type'] == behavior]
        if intervals.empty:
            continue
        in_behavior = intervals.index.get_level_values('timestamp').total_seconds()
        breaks = np.where(np.diff(in_behavior) > 0.1)[0] + 1
        chunks = np.split(in_behavior, breaks)
        for chunk in chunks:
            ax.axvspan(chunk[0], chunk[-1], alpha=0.2, color=color, lw=0)
        patches.append(mpatches.Patch(color=color, alpha=0.2, label=behavior))

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Z-score')
    ax.set_title(title)
    ax.legend(handles=patches, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def section_divider(animal, session):
    fig = plt.figure(figsize=(14, 3))
    fig.text(0.5, 0.5, f'{animal} — Session {session}',
              ha='center', va='center', fontsize=20)
    plt.axis('off')
    return fig


def cell_columns(act):
    cols = [c for c in act.columns if str(c).isdigit()]
    return sorted(cols, key=int)


def plot_animal(animal, sessions, output_dir, traces_dir, behavior_csv, info_csv):
    pdf_path = output_dir / f'{animal}_all_cells.pdf'
    with PdfPages(pdf_path) as pdf:
        for session in sessions:
            try:
                act = load_aligned_act(
                    animal=animal,
                    session=session,
                    traces_dir=traces_dir,
                    behavior_csv=behavior_csv,
                    info_csv=info_csv,
                )
            except (FileNotFoundError, StopIteration, KeyError) as e:
                print(f'  skipping {animal} {session}: {e}')
                continue

            pdf.savefig(section_divider(animal, session))
            plt.close('all')

            cells = cell_columns(act)
            for cell in cells:
                fig, ax = plt.subplots(figsize=(14, 3))
                plot_cell_trace(ax, act[cell], act,
                                 title=f'{animal} {session} | Cell {cell}')
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
            print(f'  {animal} {session}: {len(cells)} cells')
    print(f'Saved {pdf_path}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--animals', nargs='+', default=ANIMALS)
    parser.add_argument('--sessions', nargs='+', default=SESSIONS)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument('--traces-dir', type=Path, default=TRACES_DIR)
    parser.add_argument('--behavior-csv', type=Path, default=BEHAVIOR_CSV)
    parser.add_argument('--info-csv', type=Path, default=INFO_CSV)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for animal in args.animals:
        print(f'{animal}:')
        plot_animal(animal, args.sessions, args.output_dir,
                    args.traces_dir, args.behavior_csv, args.info_csv)


if __name__ == '__main__':
    main()
