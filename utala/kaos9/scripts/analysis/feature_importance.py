#!/usr/bin/env python3
"""
Phase 3.3.2 — Feature Importance via Ablation

Tests what happens when we remove features from the Linear Value agent.

Experiments:
1. Category ablation: remove entire feature categories
2. Top-N: train with only the N most important features
3. Find the minimal feature set that retains ~40% vs Heuristic
"""

import sys
import time
import numpy as np
import random as pyrandom

from src.utala.agents.linear_value_agent import LinearValueAgent
from src.utala.agents.random_agent import RandomAgent
from src.utala.agents.heuristic_agent import HeuristicAgent
from src.utala.evaluation.harness import Harness
from src.utala.state import Player
from src.utala.learning.state_action_features import StateActionFeatureExtractor


# ── Training config (same as original) ──
TRAIN_GAMES = 5000
EVAL_GAMES = 200   # per matchup, balanced
LR = 0.01
DISCOUNT = 0.95
EPSILON = 0.1
SEED = 42


def p(msg=""):
    print(msg, flush=True)


def train_and_eval(agent, label, seed_offset=0):
    """Train agent for TRAIN_GAMES, evaluate vs Heuristic and Random."""
    harness = Harness(verbose=False)
    heuristic = HeuristicAgent("H", seed=99)
    rand_agent = RandomAgent("R")

    agent.set_training_mode(True)

    for g in range(1, TRAIN_GAMES + 1):
        opponent = heuristic if g % 10 < 7 else rand_agent
        agent_as_p1 = (g % 2 == 0)
        seed = 900000 + seed_offset + g

        if agent_as_p1:
            p1, p2, ap = agent, opponent, Player.ONE
        else:
            p1, p2, ap = opponent, agent, Player.TWO

        agent.start_episode()
        result = harness.run_game(p1, p2, seed=seed)

        if result.winner is None:
            outcome = 0.5
        elif result.winner == ap:
            outcome = 1.0
        else:
            outcome = 0.0

        agent.observe_outcome(outcome)

    # Evaluate
    agent.set_training_mode(False)
    results = {}

    for opp_name, opp in [("Heuristic", HeuristicAgent("HE", seed=77)),
                           ("Random", RandomAgent("RE"))]:
        wins = 0
        for i in range(EVAL_GAMES):
            agent_as_p1 = i < EVAL_GAMES // 2
            seed = 800000 + seed_offset + i

            if agent_as_p1:
                p1, p2, ap = agent, opp, Player.ONE
            else:
                p1, p2, ap = opp, agent, Player.TWO

            agent.start_episode()
            result = harness.run_game(p1, p2, seed=seed)
            if result.winner == ap:
                wins += 1

        results[opp_name] = wins / EVAL_GAMES

    return results


def create_masked_agent(name, mask, seed_offset=0):
    """Create a LinearValueAgent that zeros out masked features during extraction."""
    agent = LinearValueAgent(name, learning_rate=LR, discount=DISCOUNT,
                             epsilon=EPSILON, seed=SEED, deck_awareness=False)

    # Monkey-patch the feature extractor to zero out masked features
    original_extract = agent.feature_extractor.extract

    def masked_extract(state, action, player):
        features = original_extract(state, action, player)
        for idx in mask:
            if idx < len(features):
                features[idx] = 0.0
        return features

    agent.feature_extractor.extract = masked_extract
    return agent


def main():
    p("=" * 72)
    p("Phase 3.3.2 — Feature Importance via Ablation")
    p("=" * 72)
    p()
    p(f"Training: {TRAIN_GAMES} games | Eval: {EVAL_GAMES} games per matchup")
    p()

    # Feature names and indices (32 features, no deck awareness)
    feature_names = [
        "phase_placement", "phase_dogfight", "turn_normalized",
        "my_rocketmen_count", "opp_rocketmen_count", "material_advantage",
        "my_squares_controlled", "opp_squares_controlled", "control_advantage",
        "contested_squares",
        "placement_power_low", "placement_power_mid", "placement_power_high",
        "placement_center", "placement_edge", "placement_corner",
        "placement_contests_square", "placement_takes_control",
        "placement_forms_line_2", "placement_forms_line_3",
        "dogfight_uses_rocket", "dogfight_uses_flare",
        "dogfight_power_diff_positive", "dogfight_power_diff_negative",
        "dogfight_kaos_cards_remaining", "dogfight_strategic_square",
        "strong_move_when_winning", "defensive_move_when_losing",
        "contests_with_high_power", "early_game_aggression",
        "late_game_caution",
        "bias",
    ]

    categories = {
        "State": list(range(0, 10)),
        "Placement": list(range(10, 20)),
        "Dogfight": list(range(20, 26)),
        "Interaction": list(range(26, 31)),
    }

    # Weight magnitudes from the trained model (for top-N selection)
    weight_magnitudes = [
        0.0612, 0.1242, 0.1335, 0.0813, 0.2109, 0.3071,  # state 0-5
        0.2050, 0.0812, 0.2559, 0.0810,                    # state 6-9
        0.0180, 0.0187, 0.0102, 0.0135, 0.0194, 0.0245,    # placement 10-15
        0.0345, 0.0214, 0.0007, 0.0057,                    # placement 16-19
        0.0038, 0.1244, 0.0117, 0.0283, 0.0565, 0.0128,    # dogfight 20-25
        0.0909, 0.0866, 0.0062, 0.0011, 0.0092,            # interaction 26-30
        0.1741,                                              # bias 31
    ]

    # ── 1. Baseline (full model, no ablation) ──
    p("─" * 72)
    p("1. BASELINE (all 32 features)")
    p("─" * 72)
    t0 = time.time()
    baseline = LinearValueAgent("Baseline", learning_rate=LR, discount=DISCOUNT,
                                epsilon=EPSILON, seed=SEED, deck_awareness=False)
    baseline_results = train_and_eval(baseline, "Baseline", seed_offset=0)
    p(f"  vs Heuristic: {baseline_results['Heuristic']:.1%}")
    p(f"  vs Random:    {baseline_results['Random']:.1%}")
    p(f"  Time: {time.time()-t0:.0f}s")
    p()

    # ── 2. Category ablation ──
    p("─" * 72)
    p("2. CATEGORY ABLATION (remove one category at a time)")
    p("─" * 72)
    p()

    cat_results = {}
    for cat_name, indices in categories.items():
        t0 = time.time()
        agent = create_masked_agent(f"No-{cat_name}", indices, seed_offset=len(cat_results)*10000)
        results = train_and_eval(agent, f"No-{cat_name}", seed_offset=len(cat_results)*10000)
        delta_h = results['Heuristic'] - baseline_results['Heuristic']
        cat_results[cat_name] = results
        p(f"  Without {cat_name:12s}: {results['Heuristic']:5.1%} vs H ({delta_h:+.1%})  |  {results['Random']:5.1%} vs R  [{time.time()-t0:.0f}s]")

    p()
    p("  Category importance (by damage when removed):")
    sorted_cats = sorted(cat_results.items(), key=lambda x: x[1]['Heuristic'])
    for cat_name, results in sorted_cats:
        delta = results['Heuristic'] - baseline_results['Heuristic']
        bar = "█" * max(0, int(-delta * 100))
        p(f"    {cat_name:12s}: {delta:+6.1%}  {bar}")
    p()

    # ── 3. Top-N features ──
    p("─" * 72)
    p("3. TOP-N FEATURES (train with only the N most important)")
    p("─" * 72)
    p()

    # Rank features by weight magnitude (excluding bias which is always kept)
    ranked_indices = sorted(range(31), key=lambda i: weight_magnitudes[i], reverse=True)

    topn_results = {}
    for n in [3, 5, 7, 10, 15, 20]:
        keep = set(ranked_indices[:n])
        keep.add(31)  # always keep bias
        mask = [i for i in range(32) if i not in keep]

        t0 = time.time()
        agent = create_masked_agent(f"Top-{n}", mask, seed_offset=n*10000)
        results = train_and_eval(agent, f"Top-{n}", seed_offset=n*10000)
        delta_h = results['Heuristic'] - baseline_results['Heuristic']
        topn_results[n] = results

        kept_names = [feature_names[i] for i in ranked_indices[:n]]
        p(f"  Top-{n:2d}: {results['Heuristic']:5.1%} vs H ({delta_h:+.1%})  |  {results['Random']:5.1%} vs R  [{time.time()-t0:.0f}s]")
        if n <= 7:
            p(f"         Features: {', '.join(kept_names)}")
    p()

    # ── 4. Summary ──
    p("─" * 72)
    p("4. SUMMARY TABLE")
    p("─" * 72)
    p()
    p(f"{'Model':<25}{'Features':>10}{'vs Heuristic':>15}{'vs Random':>12}{'Delta H':>10}")
    p("-" * 72)
    p(f"{'Baseline (all 32)':<25}{'32':>10}{baseline_results['Heuristic']:>14.1%}{baseline_results['Random']:>12.1%}{'—':>10}")

    for cat_name in ["State", "Placement", "Dogfight", "Interaction"]:
        r = cat_results[cat_name]
        n_feat = 32 - len(categories[cat_name])
        delta = r['Heuristic'] - baseline_results['Heuristic']
        p(f"{'No ' + cat_name:<25}{n_feat:>10}{r['Heuristic']:>14.1%}{r['Random']:>12.1%}{delta:>+9.1%}")

    for n in [3, 5, 7, 10, 15, 20]:
        r = topn_results[n]
        delta = r['Heuristic'] - baseline_results['Heuristic']
        p(f"{'Top-' + str(n):<25}{n+1:>10}{r['Heuristic']:>14.1%}{r['Random']:>12.1%}{delta:>+9.1%}")
    p()

    # Find minimal set
    p("─" * 72)
    p("5. MINIMAL FEATURE SET ANALYSIS")
    p("─" * 72)
    p()
    threshold = baseline_results['Heuristic'] * 0.9  # within 90% of baseline
    p(f"  Baseline: {baseline_results['Heuristic']:.1%} vs Heuristic")
    p(f"  Threshold (90% of baseline): {threshold:.1%}")
    p()
    for n in sorted(topn_results.keys()):
        wr = topn_results[n]['Heuristic']
        ok = "✓" if wr >= threshold else "✗"
        p(f"  Top-{n:2d}: {wr:5.1%}  {ok}")
    p()


if __name__ == "__main__":
    main()
