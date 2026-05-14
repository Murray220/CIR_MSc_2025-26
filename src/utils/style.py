# CIR parameters (temporary here for now 14.05.2026)

def cir_delta(kappa: float, theta: float, sigma: float) -> float:
    return 4.0 * kappa * theta / sigma**2


def kl_alpha(kappa: float, theta: float, sigma: float) -> float:
    return (4.0 * kappa * theta - sigma**2) / 8.0



# Colour scheme and general formatting for plotting throughout the thesis

METHOD_COLOURS = {
    "FTE": "#4477AA", # Blue
    "HH": "#228833", # Green
    "ProjEuler": "#AA3377", # Magenta
    "KL": "#EE6677", # Salmon
    "ChoiKwok": "#AA8822", # Gold(ish)
    "Exact": "#000000", # Black
}

METHOD_LINESTYLES = {
    "FTE" : "-",
    "HH": "--",
    "ProjEuler": "-.",
    "KL": ":",
    "ChoiKwok": "--",
    "Exact": "-"
}

REGIME_COLOURS = {
    "A": "#0072B2", # Feller well-satisified
    "B": "#56B4E9", # Feller satisfied
    "C": "#CCBB44", # Feller Boundary
    "D": "#E69F00", # Feller violated
    "E": "#D55E00", # Stong Feller violation
}

METHOD_LABELS = {
    "FTE": "Full Truncation Euler",
    "HH": "H-H Milstein",
    "ProjEuler": "Projected Euler",
    "KL": "Kelly-Lord",
    "ChoiKwok": "Choi-Kwok",
    "Exact": "Exact",
}

DEFAULT_LINEWDITH = 1.8
EXACT_LINEWIDTH = 2.2
GRID_ALPHA = 0.25