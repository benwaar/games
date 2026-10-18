#!/usr/bin/env python3
"""
Phase 5.1 — Improve the DQN

Changes from Phase 4 (train_dqn.py):
  - Checkpoint saving: best model saved during training (the 53% at 30K was never saved)
  - Longer Stage 2: extended 10K-30K → 10K-40K (best results came from Heuristic + self-play mix)
  - LR scheduling: halve LR at Stage 3 start to consolidate gains
  - Polyak averaging: soft target updates (TAU=0.01) instead of hard copy every 1K steps
  - Prioritized replay: sample high TD-error transitions more often

Flags at the top let you toggle each improvement independently for ablation.

Phase 4 baseline: 47% final / 53% peak vs Heuristic
Phase 5 target: consistent >50% vs Heuristic over 200-game eval
"""

import sys
import time
import random
from pathlib import Path

sys.path.insert(0, 'src')

from utala.deep_learning.dqn_agent import DQNAgent
from utala.agents.random_agent import RandomAgent
from utala.agents.heuristic_agent import HeuristicAgent
from utala.evaluation.harness import Harness
from utala.state import GameConfig, Player

# ---------------------------------------------------------------------------
# Phase 5 improvement flags — toggle to isolate each contribution
#
# Run 1 results (all True, 60K games): 36.5% final / 48% peak
#   - Polyak + prioritized replay amplified self-play instability
#   - Loss exploded at 28K (Polyak target tracks too fast during self-play)
#   - Extended Stage 2 ran 10K games past the natural instability onset
#
# Run 2 results (checkpoint + LR, 50K): best 56% (game 27,500), final 42%
#   - Stage 3 (80% self-play) consistently destroys Stage 2 gains
#   - Peak always in late Stage 2 (17,500–27,500 games)
#   - 200-game eval of run 2 best: 42.5% vs Heuristic (56% was noisy 100-game sample)
#
# Run 3 results (Stage 2 only, 30K): best 50% (game 10K), final 38%
#   - Sweep of run-2b checkpoints (3×200 game eval) confirmed:
#       game 30K: 48.0% ±1.0% — most consistent peak
#       game 32.5K: 35.2% ±0.8% — Stage 3 (80% self-play) crashes immediately
#   - Root cause: Stage 3 shifts to 80% self-play, instantly destabilises the agent
#
# Run 4: fix Stage 3 — keep 50/50 Heuristic/self-play mix, just halve LR at 30K.
#   Hypothesis: the opponent mix is the culprit, not the extra training.
#   50K games total; Stage 3 = Stage 2 opponent mix + lower LR.
# ---------------------------------------------------------------------------
USE_CHECKPOINT_SAVING  = True   # Save best model during training
USE_EXTENDED_STAGE2    = False  # Stage 2 ends at 30K
USE_LR_SCHEDULING      = True   # Halve LR at Stage 3 start (game 30K)
USE_POLYAK_AVERAGING   = False  # Disabled: destabilises self-play
USE_PRIORITIZED_REPLAY = False  # Disabled: amplifies oscillation

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG = GameConfig(fixed_dogfight_order=False)

TOTAL_GAMES   = 50000
EVAL_INTERVAL = 2500
EVAL_GAMES    = 100
SEED_BASE     = 500000

# Curriculum boundaries
STAGE_1_END = 10000
STAGE_2_END = 40000 if USE_EXTENDED_STAGE2 else 30000

# DQN hyperparameters
LEARNING_RATE   = 0.001
DISCOUNT        = 0.95
REPLAY_CAPACITY = 30000
BATCH_SIZE      = 64
HIDDEN_DIM      = 128
N_STEP          = 3

# Polyak tau (only used when USE_POLYAK_AVERAGING=True)
POLYAK_TAU   = 0.01
TARGET_UPDATE = 1000  # Used only when polyak is disabled

OUTPUT_DIR = Path("results/dqn_v2d")


def get_opponent(game_num, agents, dqn_agent):
    if game_num <= STAGE_1_END:
        # Stage 1: 70% Heuristic, 30% Random
        return agents['Heuristic'] if random.random() < 0.7 else agents['Random']
    else:
        # Stage 2 + Stage 3: keep 50% Heuristic, 50% self-play throughout.
        # Stage 3 (run 4 fix) uses the same mix as Stage 2 — only LR changes.
        # The original 80% self-play in Stage 3 caused immediate performance crash
        # (35.2% at game 32,500 after 48% at game 30,000 in every run).
        return agents['Heuristic'] if random.random() < 0.5 else dqn_agent


def get_epsilon(game_num):
    if game_num <= STAGE_1_END:
        progress = game_num / STAGE_1_END
        return 1.0 - 0.9 * progress
    elif game_num <= STAGE_2_END:
        progress = (game_num - STAGE_1_END) / (STAGE_2_END - STAGE_1_END)
        return 0.1 - 0.05 * progress
    else:
        progress = (game_num - STAGE_2_END) / (TOTAL_GAMES - STAGE_2_END)
        return 0.05 - 0.04 * progress


def get_stage(game_num):
    if game_num <= STAGE_1_END:
        return 1
    elif game_num <= STAGE_2_END:
        return 2
    return 3


def evaluate(harness, agent, opponents, num_games, seed_base):
    agent.set_training(False)
    results = {}

    for opp_name, opponent in opponents.items():
        wins = draws = losses = 0
        for i in range(num_games):
            agent_as_p1 = (i < num_games // 2)
            seed = seed_base + i
            if agent_as_p1:
                result = harness.run_game(agent, opponent, seed=seed)
                if result.winner == Player.ONE:
                    wins += 1
                elif result.winner is None:
                    draws += 1
                else:
                    losses += 1
            else:
                result = harness.run_game(opponent, agent, seed=seed)
                if result.winner == Player.TWO:
                    wins += 1
                elif result.winner is None:
                    draws += 1
                else:
                    losses += 1

        win_rate = wins / num_games
        results[opp_name] = {'wins': wins, 'draws': draws, 'losses': losses, 'win_rate': win_rate}
        print(f"  vs {opp_name:10s}: {win_rate:.1%} ({wins}-{draws}-{losses})")

    agent.set_training(True)
    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 5.1 — IMPROVED DQN (Variant A)")
    print("=" * 80)
    print()
    print("Active improvements:")
    print(f"  Checkpoint saving:   {USE_CHECKPOINT_SAVING}")
    print(f"  Extended Stage 2:    {USE_EXTENDED_STAGE2}  (ends at {STAGE_2_END}K)")
    print(f"  LR scheduling:       {USE_LR_SCHEDULING}")
    print(f"  Polyak averaging:    {USE_POLYAK_AVERAGING}  (tau={POLYAK_TAU})")
    print(f"  Prioritized replay:  {USE_PRIORITIZED_REPLAY}")
    print()
    print(f"Curriculum: Stage 1 (0-{STAGE_1_END}), "
          f"Stage 2 ({STAGE_1_END}-{STAGE_2_END}), "
          f"Stage 3 ({STAGE_2_END}-{TOTAL_GAMES})")
    print()

    agent = DQNAgent(
        name="DQN-v2-Phase5",
        learning_rate=LEARNING_RATE,
        discount=DISCOUNT,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=1.0,          # Managed externally via get_epsilon()
        replay_capacity=REPLAY_CAPACITY,
        batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE,
        hidden_dim=HIDDEN_DIM,
        n_step=N_STEP,
        seed=42,
        config=CONFIG,
        polyak_tau=POLYAK_TAU if USE_POLYAK_AVERAGING else 0.0,
        use_prioritized_replay=USE_PRIORITIZED_REPLAY,
    )

    print(f"State dim: {agent.state_dim}, Action dim: {agent.action_dim}")
    print(f"Network params: {sum(p.numel() for p in agent.q_network.parameters()):,}")
    print()

    harness  = Harness(verbose=False, config=CONFIG)
    opponents = {
        'Random':    RandomAgent("Random-Train", seed=42),
        'Heuristic': HeuristicAgent("Heuristic-Train", seed=42, config=CONFIG),
    }
    eval_opponents = {
        'Random':    RandomAgent("Random-Eval"),
        'Heuristic': HeuristicAgent("Heuristic-Eval", config=CONFIG),
    }

    eval_history    = []
    best_vs_heur    = 0.0
    start_time      = time.time()
    lr_scheduled    = False  # Track whether Stage-3 LR drop has been applied
    random.seed(42)

    # Anneal prioritized_beta from 0.4 → 1.0 over full training
    def get_beta(game_num):
        return 0.4 + 0.6 * (game_num / TOTAL_GAMES)

    # Initial evaluation
    print("Initial evaluation:")
    initial = evaluate(harness, agent, eval_opponents, EVAL_GAMES, SEED_BASE)
    eval_history.append({'games': 0, 'results': initial})
    print()

    for game_num in range(1, TOTAL_GAMES + 1):
        agent.epsilon = get_epsilon(game_num)

        # Anneal prioritized replay beta
        if USE_PRIORITIZED_REPLAY:
            agent.prioritized_beta = get_beta(game_num)

        # LR scheduling: halve LR at start of Stage 3
        if USE_LR_SCHEDULING and not lr_scheduled and game_num > STAGE_2_END:
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] *= 0.5
            lr_scheduled = True
            print(f"\n[LR schedule] Stage 3 start: LR reduced to {agent.optimizer.param_groups[0]['lr']:.5f}\n")

        # Select and run one training game
        opponent = get_opponent(game_num, opponents, agent)
        agent_as_p1 = (game_num % 2 == 0)
        seed = SEED_BASE + game_num

        if agent_as_p1:
            harness.run_game(agent, opponent, seed=seed)
        else:
            harness.run_game(opponent, agent, seed=seed)

        # Progress log every 500 games
        if game_num % 500 == 0:
            stats   = agent.get_stats()
            elapsed = time.time() - start_time
            stage   = get_stage(game_num)
            print(f"Game {game_num:6d}/{TOTAL_GAMES} | "
                  f"Stage {stage} | "
                  f"ε={agent.epsilon:.3f} | "
                  f"Loss={stats['avg_loss_recent']:.4f} | "
                  f"Buffer={stats['buffer_size']:6d} | "
                  f"{game_num/elapsed:.1f} g/s")

        # Evaluation checkpoint
        if game_num % EVAL_INTERVAL == 0:
            print()
            print(f"Evaluation after {game_num} games (Stage {get_stage(game_num)}):")
            results = evaluate(
                harness, agent, eval_opponents,
                EVAL_GAMES, SEED_BASE + 100000 + game_num
            )
            eval_history.append({'games': game_num, 'results': results})

            vs_heur = results['Heuristic']['win_rate']

            # Save checkpoint at every eval interval
            if USE_CHECKPOINT_SAVING:
                ckpt_path = OUTPUT_DIR / f"dqn_v2_{game_num}.pth"
                agent.save(str(ckpt_path))

                # Save best separately
                if vs_heur > best_vs_heur:
                    best_vs_heur = vs_heur
                    best_path = OUTPUT_DIR / "dqn_v2_best.pth"
                    agent.save(str(best_path))
                    print(f"  ** New best: {vs_heur:.1%} vs Heuristic — saved to {best_path}")

            print()

    # Final evaluation (200 games)
    print("=" * 80)
    print("FINAL EVALUATION (200 games)")
    print("=" * 80)
    print()

    final   = evaluate(harness, agent, eval_opponents, EVAL_GAMES * 2, SEED_BASE + 200000)
    elapsed = time.time() - start_time

    if USE_CHECKPOINT_SAVING:
        agent.save(str(OUTPUT_DIR / "dqn_v2_final.pth"))

    print()
    print(f"Training complete: {elapsed/60:.1f} minutes")
    print(f"Best vs Heuristic during training: {best_vs_heur:.1%}")
    print()

    # Progression table
    print(f"{'Games':<10} {'Stage':<8} {'vs Random':<12} {'vs Heuristic':<15}")
    print("-" * 48)
    for entry in eval_history:
        g     = entry['games']
        stage = get_stage(g) if g > 0 else '-'
        r     = entry['results']['Random']['win_rate']
        h     = entry['results']['Heuristic']['win_rate']
        print(f"{g:<10} {str(stage):<8} {r:>6.1%}       {h:>6.1%}")

    final_heur  = final['Heuristic']['win_rate']
    final_rand  = final['Random']['win_rate']
    print(f"{'FINAL':<10} {'':<8} {final_rand:>6.1%}       {final_heur:>6.1%}")
    print()

    # Comparison vs Phase 4 baselines
    phase4_baseline = 47.0
    actual_pct      = final_heur * 100
    print(f"vs Heuristic: {actual_pct:.1f}%  (Phase 4 final: {phase4_baseline}%, peak: 53.0%)")
    print()

    if actual_pct >= 50:
        print("PASS: Consistent >50% vs Heuristic achieved")
    elif actual_pct > phase4_baseline:
        print(f"IMPROVEMENT: +{actual_pct - phase4_baseline:.1f}pp over Phase 4 (target: 50%)")
    else:
        print("NO IMPROVEMENT: revisit hyperparameters")

    print("=" * 80)


if __name__ == "__main__":
    main()
