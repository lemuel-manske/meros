"""Browser report for the pairwise results of experiment 002."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypedDict


type Representation = Literal["reference", "composite", "enhanced_composite"]

REPRESENTATIONS: tuple[Representation, ...] = ("reference", "composite", "enhanced_composite")


class ComparisonRow(TypedDict):
    source_individual_id: str
    target_individual_id: str
    source_selection_id: str
    target_selection_id: str
    same_individual: bool
    same_video: bool
    representation: Representation
    source_keypoints: int
    target_keypoints: int
    forward_good: int
    backward_good: int
    mutual_matches: int
    inliers: int
    ransac_attempted: bool


def write_report(rows: Sequence[ComparisonRow], output: Path, *, source_commit: str) -> None:
    payload = json.dumps({"rows": rows, "source_commit": source_commit}).replace("<", "\\u003c")

    output.write_text(HTML.replace("__RESULTS__", payload), encoding="utf-8")


HTML = r"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Experiment 002 · Pair comparisons</title>
<style>
:root { color-scheme: light; font: 15px/1.5 system-ui, sans-serif; color: #183145;
  background: #f2f5f7; }
body { max-width: 1440px; margin: auto; padding: 32px; }
h1 { font-size: 32px; margin: 8px 0; } h2 { margin: 32px 0 12px; }
h3 { margin: 0 0 14px; } p { max-width: 900px; }
a { color: #155e75; } .muted, small { color: #506676; }
.grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }
.panel, .pair { background: white; padding: 20px; border: 1px solid #d4dfe6;
  border-radius: 12px; } .pair { margin-bottom: 20px; scroll-margin-top: 20px; }
.matrix { overflow: auto; } table { border-collapse: collapse; width: 100%; }
th, td { padding: 7px; text-align: center; border-bottom: 1px solid #e1e8ed; }
.matrix td { padding: 2px; min-width: 30px; height: 36px; }
.matrix a { display: block; padding: 5px; text-decoration: none; color: #183145; }
.same { outline: 2px solid #007d7a; outline-offset: -2px; }
.legend { display: inline-block; padding: 2px 8px; margin-right: 10px; }
.badge { background: #e4eeee; padding: 3px 8px; border-radius: 5px; margin-left: 10px;
  font-size: 13px; font-weight: normal; }
.images { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
figure { margin: 0; min-width: 0; } figcaption { font-size: 12px; margin-bottom: 6px; }
img { display: block; width: 100%; height: 170px; object-fit: contain;
  background: #243441; border-radius: 6px; }
.stats { margin-top: 14px; font-size: 14px; } .stats th { font-weight: normal; }
.stats td { font-weight: bold; } details { margin-top: 12px; }
summary { cursor: pointer; color: #506676; } .key { columns: 2; padding-left: 24px; }
.key li { break-inside: avoid; margin: 6px 0; overflow-wrap: anywhere; }
@media (max-width: 950px) { .grid { grid-template-columns: 1fr; } .key { columns: 1; }
  body { padding: 16px; } img { height: 220px; } }
@media print { .pair { break-inside: avoid; } body { padding: 0; } }
</style>
<header>
<p class="muted">MEROS / EXPERIMENT 002</p>
<h1>Does body pattern distinguish individuals?</h1>
<p>Compare the same observation pair across reference crops, composites and enhanced composites.
Start with the matrices, then select a cell to inspect its images and matching statistics.</p>
<p id="counts"></p>
<p><a href="metrics.csv">Raw metrics CSV</a> · Source commit: <code id="commit"></code></p>
</header>
<h2>Geometrically consistent matches</h2>
<p>Each cell shows affine RANSAC <strong>inliers</strong>. All three matrices use the same color scale
(white = 0, darker blue = more inliers). Hover for mutual matches and RANSAC status.
<span class="legend same">Same individual</span> Unoutlined cells compare different individuals.
A dash means no comparison, not zero matches.</p>
<div id="matrices" class="grid"></div>
<ol id="key" class="key"></ol>
<p class="muted">Zero inliers can mean RANSAC was not attempted because too few mutual matches
were available, or that no valid affine model was retained. Inlier ratio is inliers / mutual matches;
it is undefined when mutual matches are zero. These are matching statistics, not identity probabilities.
Same-video pairs may share encounter conditions and are not independent cross-encounter evidence.</p>
<h2>Inspect each pair</h2>
<div id="pairs"></div>
<script id="results" type="application/json">__RESULTS__</script>
<script>
const data = JSON.parse(document.getElementById('results').textContent);
const rows = data.rows;
const representations = [
  ['reference', 'Reference'], ['composite', 'Composite'],
  ['enhanced_composite', 'Enhanced composite']
];
const el = (tag, text, cls) => {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (cls) node.className = cls;
  return node;
};
const selections = new Map();
const pairs = new Map();
for (const row of rows) {
  selections.set(row.source_selection_id, row.source_individual_id);
  selections.set(row.target_selection_id, row.target_individual_id);
  const key = JSON.stringify([row.source_selection_id, row.target_selection_id]);
  if (!pairs.has(key)) pairs.set(key, []);
  pairs.get(key).push(row);
}
const ids = [...selections.keys()].sort((a, b) =>
  selections.get(a).localeCompare(selections.get(b), undefined, {numeric: true}) ||
  a.localeCompare(b, undefined, {numeric: true}));
const labels = new Map(ids.map((id, index) => [id, `S${index + 1}`]));
const pairIds = new Map([...pairs.keys()].map((key, index) => [key, `pair-${index + 1}`]));
const same = [...pairs.values()].filter(pair => pair[0].same_individual).length;
document.getElementById('counts').textContent =
  `${ids.length} selections · ${pairs.size} pairs (${same} same individual, ` +
  `${pairs.size - same} different individuals) · 3 representations per pair`;
document.getElementById('commit').textContent = data.source_commit;
for (const id of ids) {
  document.getElementById('key').append(el('li',
    `${labels.get(id)} — individual ${selections.get(id)} · ${id}`));
}
const maximum = Math.max(1, ...rows.map(row => row.inliers));
const status = row => row.ransac_attempted ? 'RANSAC attempted' : 'RANSAC not attempted';
for (const [representation, title] of representations) {
  const panel = el('section', undefined, 'panel matrix');
  panel.append(el('h3', title));
  const table = el('table');
  table.setAttribute('aria-label', `${title}: affine RANSAC inliers by selection pair`);
  const header = el('tr');
  header.append(el('th', ''));
  ids.forEach(id => header.append(el('th', labels.get(id))));
  table.append(header);
  for (const [i, source] of ids.entries()) {
    const tr = el('tr');
    tr.append(el('th', labels.get(source)));
    for (const [j, target] of ids.entries()) {
      const td = el('td');
      const direct = JSON.stringify([source, target]);
      const reverse = JSON.stringify([target, source]);
      const key = pairs.has(direct) ? direct : reverse;
      const row = pairs.get(key)?.find(row => row.representation === representation);
      if (j <= i || !row) {
        td.textContent = '—';
      } else {
        const link = el('a', String(row.inliers));
        link.href = `#${pairIds.get(key)}`;
        link.title = `${labels.get(source)} / ${labels.get(target)}: ` +
          `${row.inliers} inliers, ${row.mutual_matches} mutual matches; ${status(row)}`;
        link.setAttribute('aria-label', link.title);
        td.style.background = `hsl(195 55% ${97 - 37 * row.inliers / maximum}%)`;
        if (row.same_individual) td.className = 'same';
        td.append(link);
      }
      tr.append(td);
    }
    table.append(tr);
  }
  panel.append(table);
  document.getElementById('matrices').append(panel);
}
for (const [key, pair] of pairs) {
  const first = pair[0];
  const section = el('section', undefined, 'pair');
  section.id = pairIds.get(key);
  const heading = el('h3',
    `${labels.get(first.source_selection_id)} ↔ ${labels.get(first.target_selection_id)}`);
  heading.append(el('span', first.same_individual ? 'Same individual' : 'Different individuals',
    'badge'), el('span', first.same_video ? 'Same video' : 'Different videos', 'badge'));
  section.append(heading);
  const grid = el('div', undefined, 'grid');
  for (const [representation, title] of representations) {
    const row = pair.find(row => row.representation === representation);
    const column = el('div');
    column.append(el('h3', title));
    const images = el('div', undefined, 'images');
    for (const id of [row.source_selection_id, row.target_selection_id]) {
      const figure = el('figure');
      figure.append(el('figcaption', `${labels.get(id)} · Individual ${selections.get(id)}`));
      const image = el('img');
      const path = id.split('/').map(encodeURIComponent).join('/');
      image.src = `representations/${path}/${representation}.png`;
      image.alt = `${title}: ${id}`;
      image.loading = 'lazy';
      figure.append(image);
      images.append(figure);
    }
    column.append(images);
    const stats = el('table', undefined, 'stats');
    const ratio = row.mutual_matches ? `${(100 * row.inliers / row.mutual_matches).toFixed(1)}%`
      : 'N/A';
    for (const [label, value] of [
      ['Mutual matches', row.mutual_matches], ['Affine inliers', row.inliers],
      ['Inlier ratio', ratio]
    ]) {
      const tr = el('tr');
      tr.append(el('th', label), el('td', String(value)));
      stats.append(tr);
    }
    column.append(stats, el('small', status(row)));
    const details = el('details');
    details.append(el('summary', 'Matching diagnostics'), el('p',
      `Keypoints: ${row.source_keypoints} source / ${row.target_keypoints} target. ` +
      `Ratio-test matches: ${row.forward_good} forward / ${row.backward_good} backward.`));
    column.append(details);
    grid.append(column);
  }
  section.append(grid);
  document.getElementById('pairs').append(section);
}
</script>
</html>
"""
