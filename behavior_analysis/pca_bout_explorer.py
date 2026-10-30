import base64
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from pipeline_paths import DATA_ROOT

# Interactive PCA explorer: PC scores of every bout on the left (color = any bout property),
# click a bout to draw its approach path on the right (color = time before the attack).
# Writes one self-contained HTML file into the PCA run folder; open it in any browser.

def main():
    BOUTS_DIR = DATA_ROOT / "attack_bouts"
    PCA_RESULTS = DATA_ROOT / "pca_results"
    RUN_DIR = sorted(PCA_RESULTS.glob("*_kinematics_zscore"))[-1]   # latest run of pca_attack_bouts.py
    N_PCS = 10                                                       # PCs offered in the axis menus
    MOUSE_EXTENT = [-2, 2, 3.4, -3.0]                                # same as pca_attack_bouts.py

    bouts = pd.read_csv(BOUTS_DIR / "bouts.csv")
    paths = np.load(BOUTS_DIR / "bout_paths.npz")["paths"]           # [n_bouts, 60, 2], bouts.csv order
    scores = pd.read_csv(RUN_DIR / "pc_scores.csv")
    explained = pd.read_csv(RUN_DIR / "explained_variance.csv")["share"].to_numpy()
    assert (scores["bout_id"].to_numpy() == bouts["bout_id"].to_numpy()).all(), "pc_scores.csv is not in bouts.csv order"

    fps = float(np.median(bouts["attack_frames"] / bouts["attack_duration"]))
    n_frames = paths.shape[1]
    seconds_before_attack = (np.arange(n_frames) - (n_frames - 1)) / fps    # last frame = the frame before the attack
    limit = float(np.ceil(np.nanpercentile(np.abs(paths), 99) / 5) * 5)

    # properties to color the PC scatter by: (label, values, categorical?)
    color_options = {
        "group": ("group", bouts["group"], True),
        "animal": ("animal / trial", scores["animal"], True),
        "intruder_aggression": ("intruder aggressive at attack start", bouts["intruder_aggression_at_start"].astype(str), True),
        "previous_attack": ("previous attack in window", bouts["previous_attack_in_window"].astype(str), True),
        "attack_duration": ("attack duration (s)", bouts["attack_duration"], False),
        "attack_start_time": ("attack start time in trial (s)", bouts["attack_start_time"], False),
    }

    # mouse outline, flipped so it matches imshow(extent=MOUSE_EXTENT) on an upward y axis
    buffer = io.BytesIO()
    ImageOps.flip(Image.open(Path(__file__).resolve().parent / "white-mouse-grey-outline-md.png")).save(buffer, format="PNG")
    mouse_png = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

    data = {
        "bout_id": bouts["bout_id"].tolist(),
        "animal": scores["animal"].tolist(),
        "pcs": {f"PC{k + 1}": np.round(scores[f"PC{k + 1}"].to_numpy(), 4).tolist() for k in range(N_PCS)},
        "explained": np.round(explained[:N_PCS], 4).tolist(),
        "paths": np.round(paths, 3).tolist(),
        "seconds": np.round(seconds_before_attack, 3).tolist(),
        "color_options": {key: {"label": label, "values": values.tolist(), "categorical": categorical}
                          for key, (label, values, categorical) in color_options.items()},
        "limit": limit,
        "mouse": {"src": mouse_png, "x": MOUSE_EXTENT[0], "y": MOUSE_EXTENT[2],
                  "sizex": MOUSE_EXTENT[1] - MOUSE_EXTENT[0], "sizey": MOUSE_EXTENT[2] - MOUSE_EXTENT[3]},
        "run": RUN_DIR.name,
    }

    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(data))
    out_path = RUN_DIR / "pca_bout_explorer.html"
    out_path.write_text(html)
    print(f"saved {out_path}")


HTML_TEMPLATE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>PCA Bout Explorer</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>
  body { font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 16px; background: #fff; color: #222; }
  .controls { display: flex; gap: 18px; align-items: center; font-size: 14px; margin-bottom: 8px; flex-wrap: wrap; }
  select { font-size: 14px; }
  .panels { display: flex; gap: 8px; flex-wrap: wrap; }
  #scatter, #path { width: 620px; height: 580px; }
  #info { font-size: 13px; color: #555; margin-top: 4px; }
</style>
</head>
<body>
<div class="controls">
  <label>color by <select id="colorBy"></select></label>
  <label>x <select id="xPC"></select></label>
  <label>y <select id="yPC"></select></label>
  <span id="info">click a bout to show its approach path</span>
</div>
<div class="panels">
  <div id="scatter"></div>
  <div id="path"></div>
</div>
<script>
const D = __DATA__;
const N = D.bout_id.length;
const PALETTE = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
                 "#aec7e8","#ffbb78","#98df8a","#ff9896","#c5b0d5","#c49c94","#f7b6d2","#c7c7c7","#dbdb8d","#9edae5"];
let selected = null;

const colorBy = document.getElementById("colorBy"), xPC = document.getElementById("xPC"), yPC = document.getElementById("yPC");
for (const [key, opt] of Object.entries(D.color_options)) colorBy.add(new Option(opt.label, key));
Object.keys(D.pcs).forEach((pc, k) => {
  const text = `${pc} (${(100 * D.explained[k]).toFixed(1)}%)`;
  xPC.add(new Option(text, pc)); yPC.add(new Option(text, pc));
});
yPC.value = "PC2";

function hoverText(i) { return `bout ${D.bout_id[i]}<br>${D.animal[i]}`; }

function drawScatter() {
  const opt = D.color_options[colorBy.value], xs = D.pcs[xPC.value], ys = D.pcs[yPC.value];
  const traces = [];
  if (opt.categorical) {
    const cats = [...new Set(opt.values)].sort();
    cats.forEach((cat, c) => {
      const idx = [...Array(N).keys()].filter(i => opt.values[i] === cat);
      traces.push({ type: "scatter", mode: "markers", name: String(cat), x: idx.map(i => xs[i]), y: idx.map(i => ys[i]),
                    customdata: idx, text: idx.map(hoverText), hoverinfo: "text",
                    marker: { size: 8, color: PALETTE[c % PALETTE.length], line: { width: 0.5, color: "#333" } } });
    });
  } else {
    const idx = [...Array(N).keys()];
    traces.push({ type: "scatter", mode: "markers", name: opt.label, x: xs, y: ys, customdata: idx,
                  text: idx.map(i => hoverText(i) + `<br>${opt.label}: ${opt.values[i]}`), hoverinfo: "text", showlegend: false,
                  marker: { size: 8, color: opt.values, colorscale: "Viridis", line: { width: 0.5, color: "#333" },
                            colorbar: { title: { text: opt.label, side: "right" }, thickness: 12 } } });
  }
  // ring around the clicked bout
  traces.push({ type: "scatter", mode: "markers", name: "selected", showlegend: false, hoverinfo: "skip",
                x: selected === null ? [] : [xs[selected]], y: selected === null ? [] : [ys[selected]],
                marker: { size: 18, color: "rgba(0,0,0,0)", line: { width: 2.5, color: "black" } } });
  const kx = Object.keys(D.pcs).indexOf(xPC.value), ky = Object.keys(D.pcs).indexOf(yPC.value);
  Plotly.react("scatter", traces, {
    title: { text: "Attack bouts in PC space", font: { size: 15 } },
    xaxis: { title: `${xPC.value} (${(100 * D.explained[kx]).toFixed(1)}%)`, zeroline: false },
    yaxis: { title: `${yPC.value} (${(100 * D.explained[ky]).toFixed(1)}%)`, zeroline: false },
    legend: { title: { text: opt.label }, font: { size: 11 } },
    hovermode: "closest", margin: { l: 60, r: 20, t: 50, b: 55 }, plot_bgcolor: "white",
  }, { responsive: true });
}

function drawPath() {
  const L = D.limit;
  const layout = {
    xaxis: { range: [-L, L], title: "left / right of intruder (cm)", zeroline: false, constrain: "domain" },
    yaxis: { range: [-L, L], title: "behind / ahead of intruder (cm)", zeroline: false, scaleanchor: "x", scaleratio: 1 },
    images: [{ source: D.mouse.src, xref: "x", yref: "y", x: D.mouse.x, y: D.mouse.y, sizex: D.mouse.sizex, sizey: D.mouse.sizey,
               sizing: "stretch", layer: "below" }],
    margin: { l: 60, r: 20, t: 50, b: 55 }, plot_bgcolor: "white", showlegend: false, hovermode: "closest",
  };
  if (selected === null) {
    layout.title = { text: "Approach path (click a bout)", font: { size: 15 } };
    Plotly.react("path", [], layout, { responsive: true });
    return;
  }
  const p = D.paths[selected], x = p.map(v => v[0]), y = p.map(v => v[1]), n = p.length;
  layout.title = { text: `Bout ${D.bout_id[selected]} · ${D.animal[selected]}`, font: { size: 15 } };
  Plotly.react("path", [
    { type: "scatter", mode: "lines", x, y, hoverinfo: "skip", line: { color: "#bbb", width: 1.5 } },
    { type: "scatter", mode: "markers", x, y, text: D.seconds.map(s => `${s.toFixed(2)} s`), hoverinfo: "text",
      marker: { size: 7, color: D.seconds, colorscale: "Viridis",
                colorbar: { title: { text: "time before attack (s)", side: "right" }, thickness: 12 } } },
    { type: "scatter", mode: "markers", x: [x[n - 1]], y: [y[n - 1]], hoverinfo: "skip",   // the frame before the attack
      marker: { size: 12, color: "rgba(0,0,0,0)", line: { width: 2, color: "black" } } },
  ], layout, { responsive: true });
}

function selectBout(i) {
  selected = i;
  document.getElementById("info").textContent = `bout ${D.bout_id[i]} · ${D.animal[i]} · run ${D.run}`;
  drawScatter(); drawPath();
}

drawScatter(); drawPath();
document.getElementById("scatter").on("plotly_click", ev => {
  const pt = ev.points[0];
  if (pt.customdata !== undefined) selectBout(pt.customdata);
});
[colorBy, xPC, yPC].forEach(el => el.addEventListener("change", drawScatter));
</script>
</body>
</html>
"""

if __name__ == '__main__':
    main()
