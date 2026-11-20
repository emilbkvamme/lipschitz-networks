# The plot format of the report, used by every study (toy and simulated data).
# Use it with plt.rcParams.update(REPORT_STYLE) at the start of a script, or with plt.rc_context(REPORT_STYLE) around one figure.

# One style for all report figures: the same text size everywhere, a closed black frame, no grid, no legend box
REPORT_STYLE = {
    "font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    "axes.spines.top": True, "axes.spines.right": True, "axes.edgecolor": "black",
    "xtick.color": "black", "ytick.color": "black",
    "axes.grid": False, "axes.axisbelow": True,
    "legend.frameon": False, "legend.handlelength": 1.6, "figure.dpi": 150, "savefig.bbox": "tight",
    # Embed the text as TrueType fonts. The default (Type 3) is drawn blurry or jagged by many PDF viewers
    "pdf.fonttype": 42,
    # Resolution of the images inside the PDFs (the 2D maps), the lines and text stay vector graphics
    "savefig.dpi": 300,
}

# The size of one panel in inches: about a 0.48\textwidth subfigure, so two fit side by side
PANEL_SIZE = (3.1, 2.8)
