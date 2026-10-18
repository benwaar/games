#!/usr/bin/env python3
"""
Phase 3.3.1 — Weight Analysis & Interpretability

Loads the trained Linear Value agent and analyzes:
1. All feature weights ranked by magnitude
2. Feature categories (state, placement, dogfight, interaction, deck)
3. Comparison to Heuristic agent's hand-coded logic
4. Surprising or counterintuitive patterns
"""

import json
import sys
import numpy as np


def load_model(path):
    with open(path) as f:
        data = json.load(f)
    weights = np.array(data["model_data"]["weights"], dtype=np.float32)
    return data, weights


def main():
    model_path = "models/LinearValue-v1-checkpoint-5000_20260324_090050.json"
    data, weights = load_model(model_path)

    # Build feature names (must match StateActionFeatureExtractor with deck_awareness=False)
    # The production model was trained before Phase 3.2 deck awareness was added
    feature_names = [
        # State features (10)
        "phase_placement",
        "phase_dogfight",
        "turn_normalized",
        "my_rocketmen_count",
        "opp_rocketmen_count",
        "material_advantage",
        "my_squares_controlled",
        "opp_squares_controlled",
        "control_advantage",
        "contested_squares",
        # Placement action features (10)
        "placement_power_low",
        "placement_power_mid",
        "placement_power_high",
        "placement_center",
        "placement_edge",
        "placement_corner",
        "placement_contests_square",
        "placement_takes_control",
        "placement_forms_line_2",
        "placement_forms_line_3",
        # Dogfight action features (6)
        "dogfight_uses_rocket",
        "dogfight_uses_flare",
        "dogfight_power_diff_positive",
        "dogfight_power_diff_negative",
        "dogfight_kaos_cards_remaining",
        "dogfight_strategic_square",
        # Interaction features (5)
        "strong_move_when_winning",
        "defensive_move_when_losing",
        "contests_with_high_power",
        "early_game_aggression",
        "late_game_caution",
        # Bias (1)
        "bias",
    ]

    # Check if model has deck awareness features
    if len(weights) > len(feature_names):
        deck_features = [
            "my_high_cards_ratio", "my_low_cards_ratio", "my_expected_value",
            "my_deck_variance", "my_deck_strength",
            "opp_high_cards_ratio", "opp_low_cards_ratio", "opp_expected_value",
            "opp_deck_variance", "opp_deck_strength",
        ]
        # Insert before bias
        feature_names = feature_names[:-1] + deck_features + ["bias"]

    n = len(weights)
    assert n == len(feature_names), f"Weight count {n} != feature count {len(feature_names)}"

    # ── Header ──
    p = lambda msg="": print(msg, flush=True)
    p("=" * 72)
    p("Phase 3.3.1 — Linear Value Agent Weight Analysis")
    p("=" * 72)
    p()
    p(f"Model: {model_path}")
    p(f"Features: {n}")
    p(f"Training games: {data['model_data'].get('training_stats', {}).get('games_trained', '?')}")
    p()

    # ── 1. All weights ranked by magnitude ──
    p("─" * 72)
    p("1. ALL FEATURES RANKED BY ABSOLUTE WEIGHT")
    p("─" * 72)
    p()
    indices = np.argsort(np.abs(weights))[::-1]
    p(f"{'Rank':<6}{'Feature':<35}{'Weight':>10}{'|Wt|':>10}")
    p("-" * 61)
    for rank, idx in enumerate(indices, 1):
        name = feature_names[idx]
        w = weights[idx]
        p(f"{rank:<6}{name:<35}{w:>+10.4f}{abs(w):>10.4f}")
    p()

    # ── 2. By category ──
    categories = {
        "State (game situation)": feature_names[:10],
        "Placement (action)": feature_names[10:20],
        "Dogfight (action)": feature_names[20:26],
        "Interaction (state × action)": feature_names[26:31],
    }
    if n > 32:
        categories["Deck awareness"] = feature_names[31:41]
        categories["Bias"] = [feature_names[41]]
    else:
        categories["Bias"] = [feature_names[31]]

    p("─" * 72)
    p("2. WEIGHTS BY CATEGORY")
    p("─" * 72)
    p()
    for cat_name, cat_features in categories.items():
        cat_weights = [weights[feature_names.index(f)] for f in cat_features]
        cat_abs = [abs(w) for w in cat_weights]
        p(f"  {cat_name}")
        p(f"    Mean |weight|: {np.mean(cat_abs):.4f}   Max: {np.max(cat_abs):.4f}   Sum: {np.sum(cat_abs):.4f}")
        for f, w in sorted(zip(cat_features, cat_weights), key=lambda x: -abs(x[1])):
            bar = "█" * int(abs(w) / max(abs(weights)) * 30)
            sign = "+" if w >= 0 else "-"
            p(f"    {sign} {f:<33}{w:>+8.4f}  {bar}")
        p()

    # ── 3. Top positive / negative ──
    p("─" * 72)
    p("3. TOP POSITIVE WEIGHTS (features correlated with winning)")
    p("─" * 72)
    pos_indices = np.argsort(weights)[::-1]
    for i in range(min(10, n)):
        idx = pos_indices[i]
        if weights[idx] <= 0:
            break
        p(f"  {weights[idx]:>+8.4f}  {feature_names[idx]}")
    p()

    p("─" * 72)
    p("4. TOP NEGATIVE WEIGHTS (features correlated with losing)")
    p("─" * 72)
    neg_indices = np.argsort(weights)
    for i in range(min(10, n)):
        idx = neg_indices[i]
        if weights[idx] >= 0:
            break
        p(f"  {weights[idx]:>+8.4f}  {feature_names[idx]}")
    p()

    # ── 4. Comparison to Heuristic ──
    p("─" * 72)
    p("5. COMPARISON TO HEURISTIC AGENT LOGIC")
    p("─" * 72)
    p()

    comparisons = [
        ("Heuristic: center > edges > corners",
         ["placement_center", "placement_edge", "placement_corner"],
         "Does Linear Value agree on position priorities?"),
        ("Heuristic: stronger rocketmen in better positions",
         ["placement_power_high", "placement_power_mid", "placement_power_low"],
         "Does Linear Value prefer high-power placements?"),
        ("Heuristic: contest opponent squares",
         ["placement_contests_square", "placement_takes_control"],
         "Does Linear Value value contesting?"),
        ("Heuristic: material + control advantage",
         ["material_advantage", "control_advantage", "my_squares_controlled"],
         "Does Linear Value agree these matter?"),
        ("Heuristic: 3-in-a-row awareness (line formation)",
         ["placement_forms_line_2", "placement_forms_line_3"],
         "Does Linear Value learn line formation? (Note: these features are hardcoded to 0.0)"),
        ("Heuristic: use weapons when behind",
         ["dogfight_uses_rocket", "dogfight_uses_flare"],
         "Does Linear Value prefer rockets or flares?"),
        ("Heuristic: fight harder for important squares",
         ["dogfight_strategic_square", "dogfight_power_diff_positive", "dogfight_power_diff_negative"],
         "Does Linear Value recognize square importance?"),
    ]

    for heuristic_rule, features, question in comparisons:
        p(f"  {heuristic_rule}")
        p(f"  Q: {question}")
        for f in features:
            idx = feature_names.index(f)
            w = weights[idx]
            p(f"    {f:<35} {w:>+8.4f}")
        p()

    # ── 5. Dead features ──
    p("─" * 72)
    p("6. DEAD / NEAR-ZERO FEATURES (|weight| < 0.01)")
    p("─" * 72)
    p()
    dead = [(feature_names[i], weights[i]) for i in range(n) if abs(weights[i]) < 0.01]
    if dead:
        for name, w in dead:
            p(f"  {name:<35} {w:>+8.4f}")
        p(f"\n  {len(dead)} of {n} features are near-zero — candidates for removal")
    else:
        p("  None — all features have meaningful weights")
    p()

    # ── 6. Summary statistics ──
    p("─" * 72)
    p("7. SUMMARY STATISTICS")
    p("─" * 72)
    p()
    p(f"  Total features:        {n}")
    p(f"  Mean |weight|:         {np.mean(np.abs(weights)):.4f}")
    p(f"  Median |weight|:       {np.median(np.abs(weights)):.4f}")
    p(f"  Max weight:            {np.max(weights):+.4f} ({feature_names[np.argmax(weights)]})")
    p(f"  Min weight:            {np.min(weights):+.4f} ({feature_names[np.argmin(weights)]})")
    p(f"  Std of weights:        {np.std(weights):.4f}")
    p()

    # Concentration: what % of total weight magnitude is in top 5/10 features?
    sorted_abs = np.sort(np.abs(weights))[::-1]
    total_abs = np.sum(sorted_abs)
    top5_pct = np.sum(sorted_abs[:5]) / total_abs * 100
    top10_pct = np.sum(sorted_abs[:10]) / total_abs * 100
    p(f"  Weight concentration:")
    p(f"    Top  5 features: {top5_pct:.1f}% of total |weight|")
    p(f"    Top 10 features: {top10_pct:.1f}% of total |weight|")
    p()


if __name__ == "__main__":
    main()
