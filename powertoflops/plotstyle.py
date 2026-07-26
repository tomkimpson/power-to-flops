"""House plotting style for the paper's figures (Nature/Science-quality).

A thin wrapper over `SciencePlots <https://github.com/garrettj403/SciencePlots>`_
plus a few repo-specific overrides, so the figure scripts share one look instead
of each re-deriving colours, fonts, and sizes. Wired into all five figure
scripts:

    scripts/plot_beta_band_figures.py  (Figs 1-3, Exp 1/2/3)
    scripts/plot_beta_phys.py          (Fig 4, the headline beta_phys(h) curve)
    scripts/plot_capability_ladder.py  (Fig 5, the capability ladder)

Usage (after ``matplotlib.use("Agg")``)::

    from powertoflops.plotstyle import apply_house_style, C, WIDTH_ICML_COL, save
    apply_house_style()
    ...
    save(fig, "exchange_band")

We drive matplotlib with ``plt.style.use(['science', 'nature', 'no-latex'])``: the
``nature`` style gives the sans-serif Nature look, and ``no-latex`` renders text
through matplotlib's own mathtext so regenerating figures needs no LaTeX install
(reproducibility, see the README). Fonts are embedded as editable TrueType
(``pdf.fonttype = 42``) as journals require.
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  (registers the 'science'/'nature' styles)
from cycler import cycler

# Okabe-Ito colourblind-safe qualitative palette (https://jfly.uni-koeln.de/color/).
OKABE_ITO = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
]

# Named handles for semantic use in the scripts.
C = {
    "black": "#000000",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#9a9a9a",
}

# Native width for a single-column figure in the two-column ICML layout of
# paper/ (6.75 in text width -> \columnwidth ~ 3.25 in), so
# \includegraphics[width=\columnwidth] renders at ~1:1 (no text rescaling).
WIDTH_ICML_COL = 3.25

_FIG_DIR = pathlib.Path(__file__).resolve().parent.parent / "figures"


def apply_house_style() -> None:
    """Apply the shared SciencePlots + repo-override rcParams. Idempotent."""
    plt.style.use(["science", "nature", "no-latex"])
    plt.rcParams.update({
        # sans-serif Nature look; math glyphs in a matching sans face
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "stixsans",
        # editable embedded TrueType (journal requirement)
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # output quality
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # legible at our ~5-inch single-column width (Nature's native column is ~3.3 in)
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        # ticks inward, minor ticks on; open L-shaped axis (left + bottom only),
        # the Nature/Science look -- no top/right spines or ticks
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": False,
        "ytick.right": False,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # colourblind-safe cycle
        "axes.prop_cycle": cycler(color=OKABE_ITO),
    })


def save(fig, stem: str) -> pathlib.Path:
    """Write ``figures/{stem}.pdf`` (vector, primary) and ``.png`` (600-dpi preview).

    Returns the PDF path. The figures directory is resolved from the repo root, so
    this works regardless of the current working directory.

    The PDF ``CreationDate`` is suppressed so re-running a plot script whose data
    has not changed produces a byte-identical file. Otherwise every rerun rewrites
    a full blob (PDFs delta-compress poorly) and a reviewer cannot tell a cosmetic
    rerun from a data change.
    """
    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = _FIG_DIR / f"{stem}.pdf"
    fig.savefig(pdf, metadata={"CreationDate": None})
    fig.savefig(_FIG_DIR / f"{stem}.png")
    return pdf
