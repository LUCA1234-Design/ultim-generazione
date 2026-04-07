"""
metalearning/strategy_genome.py — Genetic Representation of Trading Strategies.

Encodes trading strategies as "genomes" — arrays of parameters (genes)
that can be combined, mutated, and conditionally expressed based on regime.

Gene map (indices):
  0:  fusion_threshold     [0.3, 0.8]
  1:  pattern_weight       [0.1, 0.6]
  2:  confluence_weight    [0.1, 0.6]
  3:  regime_weight        [0.05, 0.3]
  4:  risk_weight          [0.05, 0.3]
  5:  atr_sl_mult          [1.0, 3.0]
  6:  atr_tp_mult          [1.5, 5.0]
  7:  rsi_overbought       [60, 85]
  8:  rsi_oversold         [15, 40]
  9:  squeeze_min_bars     [5, 20]
  10: cooldown_mult        [0.5, 2.0]
  11: kelly_fraction       [0.1, 0.5]
  12: max_position_pct     [0.1, 1.0]
  13: regime_switch_lr     [0.01, 0.2]
  14: trend_regime_bias    [-0.2, 0.2]   (epigenetic: active only in trending regime)

Integration: StrategyEvolver uses this module to represent and evolve
strategy variants. MAMLAdapter can read genomes to initialise agent weights.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("StrategyGenome")

# ---- Gene definitions -------------------------------------------------------

GENE_NAMES = [
    "fusion_threshold",
    "pattern_weight",
    "confluence_weight",
    "regime_weight",
    "risk_weight",
    "atr_sl_mult",
    "atr_tp_mult",
    "rsi_overbought",
    "rsi_oversold",
    "squeeze_min_bars",
    "cooldown_mult",
    "kelly_fraction",
    "max_position_pct",
    "regime_switch_lr",
    "trend_regime_bias",
]

GENE_BOUNDS: List[Tuple[float, float]] = [
    (0.30, 0.80),  # fusion_threshold
    (0.10, 0.60),  # pattern_weight
    (0.10, 0.60),  # confluence_weight
    (0.05, 0.30),  # regime_weight
    (0.05, 0.30),  # risk_weight
    (1.00, 3.00),  # atr_sl_mult
    (1.50, 5.00),  # atr_tp_mult
    (60.0, 85.0),  # rsi_overbought
    (15.0, 40.0),  # rsi_oversold
    (5.0,  20.0),  # squeeze_min_bars
    (0.50, 2.00),  # cooldown_mult
    (0.10, 0.50),  # kelly_fraction
    (0.10, 1.00),  # max_position_pct
    (0.01, 0.20),  # regime_switch_lr
    (-0.2, 0.20),  # trend_regime_bias (epigenetic)
]

# Epigenetic genes: only expressed in specific regimes
_EPIGENETIC_GENES = {
    "trend_regime_bias": ["trending"],  # gene 14 only active in trending regime
}

N_GENES = len(GENE_NAMES)
_GENE_INDEX = {name: i for i, name in enumerate(GENE_NAMES)}

# Default genome (safe starting point)
_DEFAULT_GENOME = np.array([
    0.55,   # fusion_threshold
    0.30,   # pattern_weight
    0.30,   # confluence_weight
    0.15,   # regime_weight
    0.15,   # risk_weight
    1.50,   # atr_sl_mult
    2.50,   # atr_tp_mult
    70.0,   # rsi_overbought
    30.0,   # rsi_oversold
    10.0,   # squeeze_min_bars
    1.00,   # cooldown_mult
    0.25,   # kelly_fraction
    0.50,   # max_position_pct
    0.05,   # regime_switch_lr
    0.00,   # trend_regime_bias
])


# ---- Core genetic operations ------------------------------------------------

def encode(strategy_params: Dict) -> np.ndarray:
    """
    Encode a strategy parameter dict into a genome array.

    Parameters
    ----------
    strategy_params : dict mapping param name → value

    Returns
    -------
    np.ndarray of shape (N_GENES,), values clipped to bounds.
    """
    genome = _DEFAULT_GENOME.copy()
    for name, val in strategy_params.items():
        if name in _GENE_INDEX:
            i = _GENE_INDEX[name]
            lo, hi = GENE_BOUNDS[i]
            genome[i] = float(np.clip(val, lo, hi))
    return genome


def decode(genome: np.ndarray) -> Dict:
    """
    Decode a genome array into a strategy parameter dict.

    Parameters
    ----------
    genome : np.ndarray of shape (N_GENES,)

    Returns
    -------
    dict mapping gene name → value
    """
    if len(genome) < N_GENES:
        genome = np.pad(genome, (0, N_GENES - len(genome)))

    result = {}
    for i, name in enumerate(GENE_NAMES):
        lo, hi = GENE_BOUNDS[i]
        result[name] = float(np.clip(genome[i], lo, hi))
    return result


def crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    method: str = "uniform",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produce two offspring from two parent genomes.

    Parameters
    ----------
    parent_a, parent_b : parent genomes (N_GENES,)
    method : 'uniform' | 'single_point' | 'blend'

    Returns
    -------
    (child_a, child_b) as (N_GENES,) arrays
    """
    n = min(len(parent_a), len(parent_b), N_GENES)
    a = parent_a[:n].copy()
    b = parent_b[:n].copy()

    if method == "uniform":
        mask = np.random.random(n) < 0.5
        child_a = np.where(mask, a, b)
        child_b = np.where(mask, b, a)

    elif method == "single_point":
        point = np.random.randint(1, n)
        child_a = np.concatenate([a[:point], b[point:]])
        child_b = np.concatenate([b[:point], a[point:]])

    elif method == "blend":
        alpha = np.random.uniform(-0.1, 1.1, n)
        child_a = alpha * a + (1 - alpha) * b
        child_b = (1 - alpha) * a + alpha * b
    else:
        raise ValueError(f"Unknown crossover method: {method}")

    return _clip_genome(child_a), _clip_genome(child_b)


def mutate(
    genome: np.ndarray,
    rate: float = 0.1,
    sigma_fraction: float = 0.1,
) -> np.ndarray:
    """
    Apply Gaussian mutation to a genome.

    Parameters
    ----------
    genome          : (N_GENES,) genome array
    rate            : probability of mutating each gene
    sigma_fraction  : mutation std as fraction of gene range

    Returns
    -------
    mutated genome (new array)
    """
    g = genome.copy()
    for i in range(min(len(g), N_GENES)):
        if np.random.random() < rate:
            lo, hi = GENE_BOUNDS[i]
            sigma = (hi - lo) * sigma_fraction
            g[i] += np.random.normal(0, sigma)
    return _clip_genome(g)


def express(
    genome: np.ndarray,
    regime: str,
) -> Dict:
    """
    Express genome into active parameter dict, respecting epigenetic rules.

    Epigenetic genes are zeroed/nullified when not in their active regime.

    Parameters
    ----------
    genome : (N_GENES,) genome array
    regime : current market regime ('trending', 'ranging', 'volatile')

    Returns
    -------
    dict of expressed (active) parameters
    """
    decoded = decode(genome)

    # Apply epigenetic silencing
    for gene_name, active_regimes in _EPIGENETIC_GENES.items():
        if regime not in active_regimes:
            # Gene is silenced in this regime → use neutral value
            if gene_name == "trend_regime_bias":
                decoded[gene_name] = 0.0
            else:
                lo, hi = GENE_BOUNDS[_GENE_INDEX[gene_name]]
                decoded[gene_name] = (lo + hi) / 2  # neutral midpoint

    return decoded


def random_genome() -> np.ndarray:
    """Generate a random genome within bounds."""
    genome = np.array([
        np.random.uniform(lo, hi)
        for lo, hi in GENE_BOUNDS
    ])
    return genome


# ---- Helpers -----------------------------------------------------------------

def _clip_genome(genome: np.ndarray) -> np.ndarray:
    """Clip all genes to their respective bounds."""
    g = genome.copy()
    for i in range(min(len(g), N_GENES)):
        lo, hi = GENE_BOUNDS[i]
        g[i] = float(np.clip(g[i], lo, hi))
    return g


def fitness_score(params: Dict, performance: Dict) -> float:
    """
    Compute a simple fitness score for a genome based on performance metrics.

    Parameters
    ----------
    params      : decoded genome dict
    performance : dict with 'win_rate', 'sharpe_ratio', 'max_drawdown'

    Returns
    -------
    float fitness score (higher = better)
    """
    win_rate = float(performance.get("win_rate", 0.5))
    sharpe = float(performance.get("sharpe_ratio", 0.0))
    max_dd = float(performance.get("max_drawdown", 0.2))

    # Composite: reward win rate and Sharpe, penalise drawdown
    score = (
        0.4 * win_rate
        + 0.4 * max(sharpe, 0) / 3.0   # normalise Sharpe by 3
        - 0.2 * max_dd
    )
    return float(np.clip(score, 0.0, 1.0))
