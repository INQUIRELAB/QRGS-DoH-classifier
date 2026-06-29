# -*- coding: utf-8 -*-
"""Regenerate Figure 2 (QRGS workflow + pseudocode) with balanced panel geometry.

Both panels span the same vertical extent. Pseudocode emphasis is BOLD ONLY
(no color highlighting) per Sara's request. Workflow connectors are solid
triangle chevrons (variant A) or thick block arrows (variant B).

Outputs PDF/SVG/PNG into figures/New Figs/Figure 2/.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Circle

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42

OUT_DIR = r"Z:\Cyber-Code\manuscript\Final_manuscript\figures\New Figs\Figure 2"
os.makedirs(OUT_DIR, exist_ok=True)

FIG_W, FIG_H = 12.6, 6.4
TITLE_FS = 14
BOX_TITLE_FS = 11.5
BOX_SUB_FS = 10.5
CODE_FS = 11.0
GROUP_FS = 11.5

# bold-only emphasis: identical color, weight carries the highlighting
INK = "#111111"
STYLES = {
    "kw":    dict(color=INK, fontweight="bold"),
    "var":   dict(color=INK),                     # math italic via mathtext
    "num":   dict(color=INK, fontweight="bold"),
    "algo":  dict(color=INK, fontweight="bold"),
    "plain": dict(color=INK),
}

header = [
    [("kw", "Input:"), ("plain", "  traffic matrix "),
     ("var", r"$X \in \mathbb{R}^{n\times 28}$"), ("plain", ", labels "),
     ("var", r"$y \in \{-1,+1\}^{n}$")],
    [("kw", "Input:"), ("plain", "  training sizes "), ("var", "$T$"),
     ("plain", ", seeds "), ("var", "$S$"), ("plain", ", grid "),
     ("var", "$G$"), ("plain", " with "), ("var", "$|G|=$"), ("num", "72")],
    [("kw", "Output:"), ("plain", "  ranked configs "), ("var", "$g$"),
     ("plain", " with "), ("var", r"$(\mu_g, \sigma_g)$")],
]
code = [
    (0, [("kw", "for"), ("plain", " seed "), ("var", "$s \\in S$"),
         ("plain", " = "), ("num", "{7, 13, 37, 42, 101}"), ("kw", "  do")]),
    (1, [("kw", "for"), ("plain", " training size "), ("var", "$t \\in T$"),
         ("plain", " = "), ("num", "{100, 200, 500, 1000}"), ("kw", "  do")]),
    (2, [("plain", "draw "), ("var", "$K=3$"),
         ("plain", " disjoint, class-balanced train/test folds")]),
    (2, [("algo", "PCA"), ("plain", " fit on "), ("kw", "training"),
         ("plain", " split "),
         ("var", r"$\mathbb{R}^{28}\!\rightarrow\!\mathbb{R}^{6}$"),
         ("plain", "; project held-out")]),
    (2, [("kw", "for"), ("plain", " configuration "), ("var", "$g \\in G$"),
         ("kw", "  do")]),
    (3, [("plain", "scale "),
         ("var", r"$\tilde{x}=(x-x_{\min})/(x_{\max}-x_{\min}) \in [0,1]$")]),
    (3, [("plain", "encode "), ("var", r"$\theta_j = w_j\,\tilde{x}_j + b_j$"),
         ("plain", "; train "), ("var", r"$f_\theta$"), ("plain", " ("),
         ("algo", "COBYLA"), ("plain", ", "), ("var", "$5(N{+}1)$"),
         ("plain", ")")]),
    (3, [("plain", "predict "),
         ("var", r"$\hat{y} = \mathrm{sign}\,\langle Z \rangle$"),
         ("plain", " on held-out fold")]),
    (3, [("var",
          r"$\mathrm{BA} = (\mathrm{recall} + \mathrm{specificity})/2$")]),
    (0, [("plain", "aggregate over "), ("var", "$K_{\\mathrm{tot}}=15$"),
         ("plain", ": "),
         ("var", r"$\mu_g = (1/15)\sum_i \mathrm{BA}_i$")]),
    (0, [("var",
          r"$\sigma_g = \left((1/14)\sum_i (\mathrm{BA}_i-\mu_g)^2\right)^{1/2}$")]),
    (0, [("plain", "rank by "), ("var", "$\\mu_g$"), ("plain", ";  select "),
         ("var", r"$g^\star = \mathrm{argmax}_g\,\mu_g$")]),
    (0, [("plain", "paired gap "),
         ("var", r"$d_i = \mathrm{BA}_i^{(1)} - \mathrm{BA}_i^{(2)}$"),
         ("plain", ",  "), ("var", r"$\bar{d}=(1/15)\sum_i d_i$")]),
]
foot = (r"recall $=\mathrm{TP}/(\mathrm{TP{+}FN})$;   "
        r"specificity $=\mathrm{TN}/(\mathrm{TN{+}FP})$;   "
        r"$C=[\,\mathrm{TN},\mathrm{FP};\,\mathrm{FN},\mathrm{TP}\,]$")

boxes = [
    ("Raw DoH flow records", "28 numeric features + label", "#bbe6bb", "#2e7d32"),
    ("Binary labels", "malicious = 1,  benign = −1", "#cdeef5", "#0e7490"),
    ("Five seeds × three folds", r"$T \in$ {100, 200, 500, 1000};  $K{=}3$ disjoint folds",
     "#e8e8e8", "#555555"),
    ("PCA representation", "fit per training fold (six components)", "#cdeef5", "#0e7490"),
    ("Feature scaling", "[0, 1] or [0, 2π], fit on training split", "#cdeef5", "#0e7490"),
    ("QRGS grid (72 configurations)", "qubits, layers, scaler, entanglement, reuploading",
     "#e3c7ec", "#7b1fa2"),
    ("Train EstimatorQNN", "COBYLA on the training split", "#ffe9cf", "#c05f00"),
    ("Held-out evaluation", "balanced accuracy, recall, specificity, AUC", "#fbd2d2", "#b71c1c"),
]


ARROW_INK = "#3f3f46"

# Block-arrow proportions (axis-fraction units). Tuned by visual inspection
# of the rendered panel (a) so the shaft is clearly visible and the head is
# not over-flared, within the small inter-box gap (~0.022 axis-height).
BLOCK = dict(sw=0.010, hw=0.022, hh=0.013, top_off=0.004, bot_off=0.001)


def draw_connector(ax, xc, gy0, gy1, gym, style):
    """Render one inter-box connector in the chosen style.

    gy0 = top of the gap (just below the upper box),
    gy1 = bottom of the gap (just above the lower box), gym = midpoint.
    """
    tf = dict(transform=ax.transAxes, clip_on=False)
    if style == "block":
        # slim, balanced flowchart block arrow: visible rectangular shaft
        # + proportionate (not over-flared) head. Tuned for the tight gap.
        sw = BLOCK["sw"]                 # shaft half-width
        hw = BLOCK["hw"]                 # head half-width
        hh = BLOCK["hh"]                 # head height
        top, bot = gy0 + BLOCK["top_off"], gy1 - BLOCK["bot_off"]
        shoulder = bot + hh
        ax.add_patch(Polygon(
            [(xc - sw, top), (xc + sw, top), (xc + sw, shoulder),
             (xc + hw, shoulder), (xc, bot), (xc - hw, shoulder),
             (xc - sw, shoulder)],
            closed=True, facecolor=ARROW_INK, edgecolor="none", **tf))
    elif style == "open_chevron":
        # downward open V drawn with two round-capped strokes (double)
        w = 0.026
        for dy in (0.0, 0.010):
            yt = gym + 0.007 - dy
            yb = gym - 0.006 - dy
            ax.plot([xc - w, xc, xc + w], [yt, yb, yt],
                    color=ARROW_INK, lw=2.6, solid_capstyle="round",
                    solid_joinstyle="round", **tf)
    elif style == "circle_line":
        # thin line ending in a small filled disc on the next box edge
        ax.plot([xc, xc], [gy0 + 0.002, gy1 - 0.002], color=ARROW_INK,
                lw=1.6, **tf)
        ax.add_patch(Circle((xc, gy1 - 0.002), 0.006, facecolor=ARROW_INK,
                            edgecolor="none", **tf))
    elif style == "taper":
        # smooth tapered (wedge) arrow: narrow tail widening to a point
        tail = gy0 + 0.003
        tip = gy1 - 0.001
        wtail, whead = 0.006, 0.034
        shoulder = tip + 0.018
        ax.add_patch(Polygon(
            [(xc - wtail, tail), (xc + wtail, tail),
             (xc + whead, shoulder), (xc, tip), (xc - whead, shoulder)],
            closed=True, facecolor=ARROW_INK, edgecolor="none", **tf))
    else:  # "chevron" (filled triangle) — original
        tw, th = 0.030, 0.0165
        ax.add_patch(Polygon(
            [(xc - tw / 2, gym + th / 2), (xc + tw / 2, gym + th / 2),
             (xc, gym - th / 2)],
            closed=True, facecolor=ARROW_INK, edgecolor="none", **tf))


def build(arrow_style, out_name):
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.42, 0.58], wspace=0.06,
                          left=0.005, right=0.995, top=0.93, bottom=0.02)
    ax_a = fig.add_subplot(gs[0, 0]); ax_a.set_axis_off()
    ax_b = fig.add_subplot(gs[0, 1]); ax_b.set_axis_off()
    for ax in (ax_a, ax_b):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # ------------------------------------------------------------ panel (a)
    ax_a.text(0.5, 1.045, "(a)QRGS workflow", fontsize=TITLE_FS,
              fontweight="bold", ha="center", va="bottom",
              transform=ax_a.transAxes)

    n = len(boxes)
    top, bottom = 0.985, 0.015
    box_h = 0.086                       # slightly shorter boxes -> larger gaps,
    gap = (top - bottom - n * box_h) / (n - 1)   # room for elongated arrows
    x0, x1 = 0.175, 0.985
    xc = (x0 + x1) / 2

    centers = []
    for i, (title, sub, fc, ec) in enumerate(boxes):
        y_top = top - i * (box_h + gap)
        y_c = y_top - box_h / 2
        centers.append(y_c)
        ax_a.add_patch(FancyBboxPatch((x0, y_top - box_h), x1 - x0, box_h,
                                      boxstyle="round,pad=0.004,rounding_size=0.013",
                                      facecolor=fc, edgecolor=ec, linewidth=1.4,
                                      transform=ax_a.transAxes, clip_on=False))
        ax_a.text(xc, y_c + 0.0195, title, fontsize=BOX_TITLE_FS,
                  fontweight="bold", ha="center", va="center", color="#111111")
        ax_a.text(xc, y_c - 0.0215, sub, fontsize=BOX_SUB_FS,
                  ha="center", va="center", color="#222222")
        if i < n - 1:
            gy0 = y_top - box_h - 0.0055          # top of gap
            gy1 = y_top - box_h - gap + 0.0055    # bottom of gap
            gym = (gy0 + gy1) / 2
            draw_connector(ax_a, xc, gy0, gy1, gym, arrow_style)

    def bracket(ax, y_hi, y_lo, label, color):
        xb = 0.115
        ax.plot([xb, xb], [y_lo, y_hi], color=color, lw=2.0, clip_on=False)
        ax.plot([xb, xb + 0.022], [y_hi, y_hi], color=color, lw=2.0, clip_on=False)
        ax.plot([xb, xb + 0.022], [y_lo, y_lo], color=color, lw=2.0, clip_on=False)
        ax.text(xb - 0.045, (y_hi + y_lo) / 2, label, rotation=90,
                fontsize=GROUP_FS, fontweight="bold", color=color,
                ha="center", va="center")

    bracket(ax_a, top + 0.003, centers[1] - box_h / 2 - 0.003,
            "Preprocessing", "#0e7490")
    bracket(ax_a, centers[2] + box_h / 2 + 0.003, bottom - 0.003,
            "Seed-by-fold learning", "#7b1fa2")

    # ------------------------------------------------------------ panel (b)
    ax_b.text(0.5, 1.045, "(b)QRGS pseudocode", fontsize=TITLE_FS,
              fontweight="bold", ha="center", va="bottom",
              transform=ax_b.transAxes)
    ax_b.add_patch(FancyBboxPatch((0.005, 0.012), 0.99, 0.973,
                                  boxstyle="round,pad=0.004,rounding_size=0.015",
                                  facecolor="#fbfbfd", edgecolor="#555555",
                                  linewidth=1.3, transform=ax_b.transAxes,
                                  clip_on=False))

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_w_px = ax_b.get_window_extent(renderer=renderer).width

    def put_segments(ax, x, y, segments, fontsize):
        for style, text in segments:
            t = ax.text(x, y, text, fontsize=fontsize, ha="left", va="center",
                        **STYLES[style])
            w_px = t.get_window_extent(renderer=renderer).width
            x += w_px / ax_w_px

    rows = len(header) + len(code) + 1          # +1 footnote
    y_hi, y_lo = 0.945, 0.052
    dy = (y_hi - y_lo) / (rows - 1 + 1.0)       # extra slack for divider gaps
    y = y_hi
    x_num, x_text = 0.065, 0.105
    indent_w = 0.045

    for segs in header:
        put_segments(ax_b, 0.035, y, segs, CODE_FS)
        y -= dy
    ax_b.plot([0.03, 0.965], [y + dy * 0.45, y + dy * 0.45],
              color="#999999", lw=0.9)
    for i, (lvl, segs) in enumerate(code, start=1):
        ax_b.text(x_num, y, str(i), fontsize=CODE_FS - 0.5, ha="right",
                  va="center", color="#777777")
        put_segments(ax_b, x_text + lvl * indent_w, y, segs, CODE_FS)
        y -= dy
    ax_b.plot([0.03, 0.965], [y + dy * 0.45, y + dy * 0.45],
              color="#999999", lw=0.9)
    ax_b.text(0.5, y, foot, fontsize=CODE_FS - 1.0, ha="center", va="center",
              color="#444444")

    for ext in ("pdf", "svg", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"{out_name}.{ext}"),
                    dpi=400, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print("saved", out_name)


build("block", "Figure_2_arrow_block")
if os.environ.get("FIG2_ALL"):
    build("open_chevron", "Figure_2_arrow_open_chevron")
    build("circle_line", "Figure_2_arrow_circle_line")
    build("taper", "Figure_2_arrow_taper")
    build("chevron", "Figure_2_bold_chevron")      # previous style, for reference
