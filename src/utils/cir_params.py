# CIR parameters (temporary here for now 14.05.2026)

def cir_delta(kappa: float, theta: float, sigma: float) -> float:
    return 4.0 * kappa * theta / sigma**2


def kl_alpha(kappa: float, theta: float, sigma: float) -> float:
    return (4.0 * kappa * theta - sigma**2) / 8.0

