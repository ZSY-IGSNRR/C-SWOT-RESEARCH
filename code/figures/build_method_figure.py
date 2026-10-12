from pathlib import Path
import sys, os

R = Path(__file__).resolve().parent
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BLUE = '#287C9E'
ORANGE = '#CE803A'
TEAL = '#25887F'
GRAY = '#667580'
INK = '#243746'
LIGHT = '#DCE4E8'
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white',
                     'text.color': INK, 'svg.fonttype': 'none'})
f = plt.figure(figsize=(7.1, 5.05))
ax = f.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, 1000), ylim=(711, 0))
ax.axis('off')
bg = Path(os.environ['SWOT_RELEASE_BACKGROUND'])
ax.imshow(plt.imread(bg), extent=(18, 512, 363, 34), zorder=0)

def tx(x, y, text, size=9, c=INK, bold=False, ha='left'):
    return ax.text(x, y, text, fontsize=size, color=c,
                   fontweight='bold' if bold else 'normal',
                   va='top', ha=ha, linespacing=1.3)

def arrow(a, b, color=GRAY, lw=.9):
    ax.annotate('', xy=b, xytext=a,
                arrowprops={'arrowstyle': '->', 'color': color,
                            'lw': lw, 'mutation_scale': 8})

tx(26, 15, 'a  SWOT and gauge observations', 9.3, bold=True)
tx(30, 75, 'SWOT', 9, BLUE, True)
tx(38, 369, 'WSE  ·  width  ·  slope', 8.5, BLUE, True)
ax.plot(375, 379, marker='v', ms=4.2, color=TEAL)
tx(391, 369, 'Gauge Q', 8.5, TEAL, True)
# A gauge marker is schematic, and its label remains editable in the SVG.
ax.plot(442, 324, marker='v', ms=4.2, color=TEAL,
        markeredgecolor='white', markeredgewidth=.6)
ax.plot([442, 442, 375], [324, 352, 371], color=TEAL, lw=.7)
arrow((518, 213), (544, 213))

tx(555, 15, 'b  Local hydraulic calibration', 9.3, bold=True)
tx(555, 65, 'Calibration WSE and width → W(h)\nGauge Q → effective geometry and resistance', 8.1, c=GRAY)
ax.fill([555, 600, 650, 858, 911, 963, 963, 555],
        [144, 144, 254, 254, 144, 144, 292, 292], color='#E7DDC9')
ax.fill([627, 884, 858, 650], [202, 202, 254, 254], color='#9ECFD9')
ax.plot([555, 600, 650, 858, 911, 963], [144, 144, 254, 254, 144, 144], color='#9B896E', lw=1.3)
ax.plot([627, 884], [202, 202], color=BLUE, lw=1.2)
tx(758, 172, 'W(h)', 9, bold=True, ha='center')
tx(758, 216, 'A(h)', 12, c=BLUE, bold=True, ha='center')
tx(758, 267, 'Effective bed reference', 7.3, c='#695B44', ha='center')
tx(555, 310, r'$Q=\frac{1}{n}\,A(h)^{5/3}\,W(h)^{-2/3}\,S^{1/2}$', 11.3, TEAL)
tx(555, 350, r'Fitted parameters: $n$, $\beta$, $\mathrm{WSE}_0$, $\mathrm{width}_0$', 8.1, c=GRAY)
arrow((758, 375), (758, 407))
ax.plot([26, 966], [398, 398], color=LIGHT, lw=.7)

tx(26, 415, 'c  Discharge prediction and evaluation', 9.3, bold=True)
tx(28, 462, 'WSE-stratified validation', 8.6, bold=True)
tx(355, 462, 'Forward temporal validation', 8.6, bold=True)
tx(682, 462, 'Calibration WSE domain', 8.6, bold=True)
tx(28, 494, 'Five WSE strata · 10 assignments', 7.6, c=GRAY)
tx(355, 494, 'Earlier records → later test blocks', 7.6, c=GRAY)
tx(682, 494, 'Validation WSE relative to calibration', 7.4, c=GRAY)

# Each schematic stratum contains calibration and held-out records.
for j in range(5):
    center = 65 + j * 52
    tx(center+6, 526, str(j+1), 7.1, c=GRAY, ha='center')
    for k, (dx, dy) in enumerate([(0, 0), (15, 0), (0, 20), (15, 20)]):
        held_out = k == j % 4
        ax.plot(center + dx, 555 + dy, 'o', ms=4.0,
                color=ORANGE if held_out else BLUE,
                markerfacecolor='white' if held_out else BLUE,
                markeredgewidth=.95)
    if j < 4:
        ax.plot([center+33, center+33], [549, 582], color=LIGHT, lw=.7)
arrow((50, 604), (308, 604))
tx(179, 616, 'Increasing WSE', 7.4, c=GRAY, ha='center')

# Expanding-window time test: refit using all records before each block.
edges = [362, 475, 524, 573, 622]
for row in range(3):
    y = 551 + row * 27
    boundary = edges[row + 1]
    stop = edges[row + 2]
    ax.plot([edges[0], edges[-1]], [y, y], color=LIGHT, lw=.7, zorder=0)
    ax.add_patch(Rectangle((edges[0], y-5), boundary-edges[0]-3, 10,
                           facecolor=BLUE, edgecolor=BLUE, lw=.8))
    ax.add_patch(Rectangle((boundary+3, y-5), stop-boundary-6, 10,
                           facecolor='white', edgecolor=ORANGE, lw=1))
arrow((362, 631), (630, 631))
tx(636, 624, 'Time', 7.4, c=GRAY)
tx(355, 661, 'Refit before each later block', 7.4, c=GRAY)

ax.plot([701, 958], [570, 570], color=GRAY, lw=.9)
ax.plot([776, 874], [570, 570], color=BLUE, lw=5)
for xx in [776, 874]:
    ax.plot([xx, xx], [558, 582], color=GRAY, lw=.8)
tx(720, 594, 'Below', 8, c=ORANGE, ha='center')
tx(825, 594, 'Within', 8, c=BLUE, ha='center')
tx(934, 594, 'Above', 8, c=ORANGE, ha='center')
tx(825, 626, 'Calibration WSE range', 7.4, c=GRAY, ha='center')

ax.plot(38, 670, 'o', color=BLUE, ms=4)
tx(51, 661, 'Calibration', 7.4)
ax.plot(184, 670, 'o', color=ORANGE, mfc='white', ms=4, mew=.95)
tx(198, 661, 'Validation', 7.4)

out = Path(os.environ['SWOT_RELEASE_OUTPUT']).resolve()
out.mkdir(exist_ok=True)
f.savefig(out / 'Figure_1.png', dpi=400)
f.savefig(out / 'Figure_1.svg')
plt.close(f)
print('Figure 1 exported at 2840 × 2020 pixels and as SVG')
