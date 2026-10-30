"""
Plot every inhibited/mixed cell's trace against behavior, labeled with its z-score(s)
and p-value(s), into a single PDF across all animals.

Mirrors plot_tuned_and_mixed_cells.py but for inhibited cells: significantly
(p<=0.05) NEGATIVELY (z<0) modulated by a behavior_type, instead of positively.

Cells are grouped into sections, in this order:
    Aggressive -> Pro-Social -> Social Avoidance -> Mixed -> Other

A cell is "single-category" if it is significantly (p<=0.05), negatively (z<0)
modulated by exactly one behavior_type for that animal/session; it goes in the
matching section. A cell inhibited by more than one behavior_type in the same
animal/session goes in "Mixed" instead, labeled with all of its inhibitions.

Usage:
    python plot_inhibited_and_mixed_cells.py
    python plot_inhibited_and_mixed_cells.py --animals C1F2 C1M1 --output figures/inhibited_mixed.pdf
"""
import argparse
from itertools import product
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from plot_all_cells_by_session import (
    ANIMALS, BEHAVIOR_CSV, INFO_CSV, SESSIONS, TRACES_DIR, plot_cell_trace,
)
from load_traces import load_aligned_act

matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['pdf.fonttype'] = 42

DATA_DIR = Path('/Users/amonast/Desktop/miniscope/data')
OUTPUT_PDF = Path('/Users/amonast/Desktop/miniscope/figures/all_animals_inhibited_mixed_cells.pdf')

CATEGORY_ORDER = ['Aggressive', 'Pro-Social', 'Social Avoidance', 'Mixed', 'Other']


def get_inhibited_df(animals, data_dir, sessions=SESSIONS):
    index_cols = ['animal', 'session', 'behavior_type']
    rows = []
    for ani, session in product(animals, sessions):
        pf = data_dir / f'p_values_{ani}_{session}.csv'
        zf = data_dir / f'z_scores_{ani}_{session}.csv'
        if not (pf.exists() and zf.exists()):
            continue
        df_p = pd.read_csv(pf).set_index(index_cols)
        df_z = pd.read_csv(zf).set_index(index_cols)

        chunk = pd.concat({'pval': df_p, 'zscore': df_z}, names=['metric'])
        chunk = chunk.reorder_levels(['animal', 'session', 'behavior_type', 'metric'])
        rows.append(chunk)
    df = pd.concat(rows).sort_index()

    sig = df.xs('pval', level='metric').le(0.05)
    neg = df.xs('zscore', level='metric').lt(0)
    inhibited = ((sig & neg).apply(lambda row: row.index[row].tolist(), axis=1)
                 .rename('inhibited_cells')
                 .reset_index())

    # C2M4/C2F1: one was the intruder mouse, one had only a tiny aggressive bout
    inhibited = inhibited[~((inhibited['animal'].isin(['C2M4', 'C2F1'])) &
                             (inhibited['behavior_type'] == 'Aggressive'))]
    return inhibited, df


def build_cell_categories(inhibited):
    """(animal, session) -> {cell: set(behavior_type)} for every inhibited cell."""
    cell_to_behaviors = {}
    for _, row in inhibited.iterrows():
        key = (row['animal'], row['session'])
        d = cell_to_behaviors.setdefault(key, {})
        for cell in row['inhibited_cells']:
            d.setdefault(cell, set()).add(row['behavior_type'])
    return cell_to_behaviors


def categorize(cell_to_behaviors):
    """Return {category: [(animal, session, cell, behaviors), ...]}."""
    sections = {c: [] for c in CATEGORY_ORDER}
    for (animal, session), cells in cell_to_behaviors.items():
        for cell, behaviors in cells.items():
            if len(behaviors) > 1:
                category = 'Mixed'
            else:
                category = next(iter(behaviors))
            sections[category].append((animal, session, cell, behaviors))

    for entries in sections.values():
        entries.sort(key=lambda e: (e[0], e[1], int(e[2])))
    return sections


def format_title(animal, session, cell, category, behaviors, df):
    def zp(behavior):
        z = df.loc[(animal, session, behavior, 'zscore'), cell]
        p = df.loc[(animal, session, behavior, 'pval'), cell]
        return f'{behavior}: z={z:.2f}, p={p:.3g}'

    stats = ' | '.join(zp(b) for b in sorted(behaviors))
    label = 'Mixed' if category == 'Mixed' else category
    return f'{animal} {session} | Cell {cell} | {label} | {stats}'


def section_divider(label, n):
    fig = plt.figure(figsize=(14, 3))
    fig.text(0.5, 0.5, f'{label}-inhibited cells (n={n})',
              ha='center', va='center', fontsize=20)
    plt.axis('off')
    return fig


def get_act(cache, animal, session, traces_dir, behavior_csv, info_csv):
    key = (animal, session)
    if key not in cache:
        try:
            cache[key] = load_aligned_act(
                animal=animal, session=session,
                traces_dir=traces_dir, behavior_csv=behavior_csv, info_csv=info_csv,
            )
        except (FileNotFoundError, StopIteration, KeyError) as e:
            print(f'  skipping {animal} {session}: {e}')
            cache[key] = None
    return cache[key]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--animals', nargs='+', default=ANIMALS)
    parser.add_argument('--sessions', nargs='+', default=SESSIONS)
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--output', type=Path, default=OUTPUT_PDF)
    parser.add_argument('--traces-dir', type=Path, default=TRACES_DIR)
    parser.add_argument('--behavior-csv', type=Path, default=BEHAVIOR_CSV)
    parser.add_argument('--info-csv', type=Path, default=INFO_CSV)
    args = parser.parse_args(argv)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    inhibited, df = get_inhibited_df(args.animals, args.data_dir, args.sessions)
    sections = categorize(build_cell_categories(inhibited))

    act_cache = {}
    with PdfPages(args.output) as pdf:
        for category in CATEGORY_ORDER:
            entries = sections[category]
            if not entries:
                print(f'{category}: no cells, skipping section')
                continue

            pdf.savefig(section_divider(category, len(entries)))
            plt.close('all')

            print(f'{category}: {len(entries)} cells')
            for animal, session, cell, behaviors in entries:
                act = get_act(act_cache, animal, session,
                               args.traces_dir, args.behavior_csv, args.info_csv)
                if act is None:
                    continue

                fig, ax = plt.subplots(figsize=(14, 3))
                title = format_title(animal, session, cell, category, behaviors, df)
                plot_cell_trace(ax, act[cell], act, title=title)
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
