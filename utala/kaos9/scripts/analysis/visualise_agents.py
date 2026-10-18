"""
Visualisations for Utala KAOS 9 agent analysis.

Usage:
    cd utala/kaos9
    python scripts/analysis/visualise_agents.py

Generates three charts saved to docs/images/:
    strength_ladder.png  — corrected win rates vs Heuristic for all agents
    placement_heatmap.png — where DQN places rocketmen on the 3×3 grid
    qvalue_heatmap.png   — DQN Q-values for placement at game start
"""

import sys
import random
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts/train/variant_a')

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

from utala.actions import ActionType, get_action_space
from utala.agents.heuristic_agent import HeuristicAgent
from utala.agents.random_agent import RandomAgent
from utala.agents.monte_carlo_agent import FastMonteCarloAgent
from utala.deep_learning.dqn_agent import DQNAgent
from utala.engine import GameEngine
from utala.evaluation.harness import Harness
from utala.state import GameConfig, Player

OUTPUT_DIR = Path("docs/images")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG   = GameConfig(fixed_dogfight_order=False)
DQN_PATH = Path("results/dqn_v2d/dqn_v2_best.pth")

# ---------------------------------------------------------------------------
# 1. Strength ladder
# ---------------------------------------------------------------------------

def plot_strength_ladder():
    """Horizontal bar chart of corrected win rates vs Heuristic."""
    agents = [
        ("Random",        47,   "#aaaaaa"),
        ("MC-Fast",       30,   "#e07b54"),
        ("TinyNN-v1 (stale)", 38.5, "#f5a623"),
        ("TinyNN-v2 (new)",   48,   "#f5a623"),
        ("DQN-v3 (best ckpt)", 56, "#4a90d9"),
        ("Heuristic",     None, "#2ecc71"),
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axvline(50, color="#cccccc", linestyle="--", linewidth=1, zorder=0, label="50% threshold")

    ys = list(range(len(agents)))
    for i, (name, pct, colour) in enumerate(agents):
        if pct is None:
            ax.barh(i, 100, color=colour, alpha=0.2)
            ax.text(2, i, f"{name}  (reference)", va="center", fontsize=10, color="#555555")
        else:
            ax.barh(i, pct, color=colour, alpha=0.85)
            ax.text(pct + 1.5, i, f"{pct}%", va="center", fontsize=10, fontweight="bold")
            ax.text(2, i, name, va="center", fontsize=10, color="white", fontweight="bold")

    ax.set_xlim(0, 105)
    ax.set_xlabel("Win rate vs Heuristic (%)", fontsize=11)
    ax.set_title("Agent Strength — Variant A (corrected, 2026-10-07)", fontsize=12, fontweight="bold")
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.text(51, -0.6, "50%", color="#aaaaaa", fontsize=9)

    fig.tight_layout()
    out = OUTPUT_DIR / "strength_ladder.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# 2. Placement heatmap
# ---------------------------------------------------------------------------

def plot_placement_heatmap(num_games: int = 500):
    """
    Run DQN vs Heuristic for num_games. For each game, record the DQN's
    FIRST placement (all 9 squares available — pure preference, no constraint).
    Produces a 3×3 heatmap of where it chooses to go first.
    """
    dqn          = DQNAgent.load(str(DQN_PATH))
    heur         = HeuristicAgent(config=CONFIG)
    action_space = get_action_space(CONFIG)
    harness      = Harness(config=CONFIG, verbose=False)

    first_counts = np.zeros((3, 3), dtype=float)
    all_counts   = np.zeros((3, 3), dtype=float)

    class RecordingAgent:
        def __init__(self, agent):
            self.agent        = agent
            self.name         = agent.name
            self.placed_count = 0
            self.first_counts = first_counts
            self.all_counts   = all_counts

        def select_action(self, state, legal_actions, player):
            action_idx = self.agent.select_action(state, legal_actions, player)
            action = action_space.get_action(action_idx)
            if action.action_type == ActionType.PLACE_ROCKETMAN:
                self.all_counts[action.row][action.col] += 1
                if self.placed_count == 0:
                    self.first_counts[action.row][action.col] += 1
                self.placed_count += 1
            return action_idx

        def game_start(self, player, seed=None):
            self.placed_count = 0
            if hasattr(self.agent, 'game_start'):
                self.agent.game_start(player, seed)

        def game_end(self, state, winner):
            if hasattr(self.agent, 'game_end'):
                self.agent.game_end(state, winner)

    recorder = RecordingAgent(dqn)
    for game_idx in range(num_games):
        harness.run_game(recorder, heur, seed=42 + game_idx)

    # Two side-by-side subplots: first placement vs all placements
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    labels = [["TL", "T", "TR"], ["L", "C", "R"], ["BL", "B", "BR"]]

    for ax, counts, title in [
        (axes[0], first_counts / first_counts.sum() * 100, "First placement"),
        (axes[1], all_counts / all_counts.sum() * 100, "All placements (order)"),
    ]:
        im = ax.imshow(counts, cmap="Blues", vmin=0)
        plt.colorbar(im, ax=ax, label="% of placements")
        for r in range(3):
            for c in range(3):
                ax.text(c, r, f"{counts[r, c]:.1f}%\n({labels[r][c]})",
                        ha="center", va="center", fontsize=9,
                        color="white" if counts[r, c] > counts.max() * 0.6 else "black")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=11, fontweight="bold")

    fig.suptitle(f"DQN-v2 Placement Preferences — {num_games} games as P1 vs Heuristic",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    out = OUTPUT_DIR / "placement_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# 3. Q-value heatmap at game start
# ---------------------------------------------------------------------------

def plot_qvalue_heatmap():
    """
    At an empty board (game start), show DQN Q-values for placing each
    rocketman power on each grid square. One subplot per rocketman power (2–10).
    """
    dqn          = DQNAgent.load(str(DQN_PATH))
    action_space = get_action_space(CONFIG)

    # Fresh game state — empty board
    engine = GameEngine(seed=42, config=CONFIG)
    state  = engine.get_state_copy()

    features     = dqn.feature_extractor.extract(state, Player.ONE)
    state_tensor = torch.FloatTensor(features)

    with torch.no_grad():
        q_values = dqn.q_network.forward(state_tensor).numpy()

    # Map placement actions to (power, row, col) → Q-value
    powers = list(range(2, 11))  # 2–10
    q_grid = np.full((len(powers), 3, 3), np.nan)

    for action_idx, action in enumerate(action_space.actions):
        if action.action_type == ActionType.PLACE_ROCKETMAN:
            pi = powers.index(action.rocketman_power)
            q_grid[pi, action.row, action.col] = q_values[action_idx]

    fig, axes = plt.subplots(3, 3, figsize=(10, 9))
    fig.suptitle("DQN-v2 Q-values for Placement at Game Start (P1, empty board)",
                 fontsize=12, fontweight="bold")

    vmin, vmax = np.nanmin(q_grid), np.nanmax(q_grid)

    for pi, power in enumerate(powers):
        ax = axes[pi // 3][pi % 3]
        grid = q_grid[pi]
        im = ax.imshow(grid, cmap="RdYlGn", vmin=vmin, vmax=vmax)
        ax.set_title(f"Power {power}", fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for r in range(3):
            for c in range(3):
                ax.text(c, r, f"{grid[r, c]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=axes, shrink=0.6, label="Q-value")
    fig.tight_layout()
    out = OUTPUT_DIR / "qvalue_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating utala agent visualisations...")
    print()
    print("1/3  Strength ladder")
    plot_strength_ladder()

    print("2/3  Placement heatmap (500 games — takes ~10s)")
    plot_placement_heatmap(num_games=500)

    print("3/3  Q-value heatmap")
    plot_qvalue_heatmap()

    print()
    print(f"All charts saved to {OUTPUT_DIR}/")
