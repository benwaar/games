"""
Train Linear Value Agent V2 with Enhanced Features (Phase 2.5).

Goal: Improve from 42% to 55%+ win rate vs Heuristic baseline.

Improvements:
- 49 enhanced features (up from 43)
- Proper action decoding and line detection
- Tactical pattern recognition
- 20K training games (up from 5K)
- Better evaluation frequency
"""

import sys
from pathlib import Path

from src.utala.agents.linear_value_agent_v2 import LinearValueAgentV2
from src.utala.agents.heuristic_agent import HeuristicAgent
from src.utala.agents.random_agent import RandomAgent
from src.utala.evaluation.harness import Harness
from src.utala.state import Player

# Training configuration
NUM_TRAINING_GAMES = 20000
EVAL_INTERVAL = 1000  # Eval every 1000 games
EVAL_GAMES = 200      # 200 games per eval
SAVE_INTERVAL = 2000  # Save checkpoint every 2000 games

# Hyperparameters
LEARNING_RATE = 0.01
DISCOUNT = 0.95
EPSILON = 0.1
SEED = 42

# Opponent mix for training (70% Heuristic, 30% Random)
OPPONENT_MIX = {
    "heuristic": 0.7,
    "random": 0.3,
}

OUTPUT_DIR = Path("results/linear_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_agent(agent: LinearValueAgentV2, game_num: int):
    """Evaluate agent against Heuristic baseline."""
    print(f"\n{'='*60}")
    print(f"Evaluation at {game_num} training games")
    print(f"{'='*60}")

    # Disable training for evaluation
    agent.set_training(False)

    # Create opponents
    heuristic = HeuristicAgent("Heuristic-Eval")
    random_agent = RandomAgent("Random-Eval")

    # Evaluate vs Heuristic
    print(f"\nEvaluating vs Heuristic ({EVAL_GAMES} games)...")
    harness = Harness(verbose=False)
    result = harness.run_balanced_match(
        agent_one=agent,
        agent_two=heuristic,
        num_games=EVAL_GAMES,
        starting_seed=SEED + game_num
    )

    total_wins = result.player_one_wins
    total_games = EVAL_GAMES
    win_rate = total_wins / total_games

    print(f"  Agent wins: {total_wins}/{total_games} ({win_rate:.1%})")
    print(f"  Heuristic wins: {result.player_two_wins}/{total_games} ({result.player_two_win_rate:.1%})")
    print(f"  Draws: {result.draws}/{total_games} ({result.draw_rate:.1%})")

    # Evaluate vs Random
    print(f"\nEvaluating vs Random ({EVAL_GAMES} games)...")
    result_random = harness.run_balanced_match(
        agent_one=agent,
        agent_two=random_agent,
        num_games=EVAL_GAMES,
        starting_seed=SEED + game_num + 1000
    )

    total_wins_random = result_random.player_one_wins
    win_rate_random = total_wins_random / total_games

    print(f"  Agent wins: {total_wins_random}/{total_games} ({win_rate_random:.1%})")

    # Get agent stats
    stats = agent.get_stats()
    print(f"\nAgent Statistics:")
    print(f"  Games trained: {stats['games_trained']}")
    print(f"  Weight updates: {stats['total_weight_updates']}")
    print(f"  Weight norm: {stats['weight_norm']:.3f}")
    print(f"  Weight mean: {stats['weight_mean']:.4f}")
    print(f"  Weight std: {stats['weight_std']:.4f}")

    # Re-enable training
    agent.set_training(True)

    return {
        "game_num": game_num,
        "win_rate_vs_heuristic": win_rate,
        "win_rate_vs_random": win_rate_random,
        "stats": stats,
    }


def main():
    print("="*60)
    print("Training Linear Value Agent V2 (Phase 2.5)")
    print("="*60)
    print(f"Training games: {NUM_TRAINING_GAMES:,}")
    print(f"Eval interval: {EVAL_INTERVAL:,}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Discount: {DISCOUNT}")
    print(f"Epsilon: {EPSILON}")
    print(f"Opponent mix: {int(OPPONENT_MIX['heuristic']*100)}% Heuristic, {int(OPPONENT_MIX['random']*100)}% Random")
    print()

    # Create agent
    agent = LinearValueAgentV2(
        name="LinearV2",
        learning_rate=LEARNING_RATE,
        discount=DISCOUNT,
        epsilon=EPSILON,
        seed=SEED,
    )

    print(f"Agent initialized with {agent.num_features} features")
    print(f"Feature names: {agent.feature_extractor.get_feature_names()[:10]}... (showing first 10)")
    print()

    # Create training opponents
    heuristic = HeuristicAgent("Heuristic-Train")
    random_agent = RandomAgent("Random-Train")

    # Initial evaluation (before training)
    eval_results = []
    initial_eval = evaluate_agent(agent, 0)
    eval_results.append(initial_eval)

    # Training loop
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}")

    games_completed = 0
    next_eval = EVAL_INTERVAL
    next_save = SAVE_INTERVAL

    import random
    import time

    rng = random.Random(SEED)
    start_time = time.time()

    while games_completed < NUM_TRAINING_GAMES:
        # Select opponent based on mix
        if rng.random() < OPPONENT_MIX["heuristic"]:
            opponent = heuristic
        else:
            opponent = random_agent

        # Alternate who goes first
        if rng.random() < 0.5:
            agents = [agent, opponent]
        else:
            agents = [opponent, agent]

        # Run one game
        harness = Harness(verbose=False)
        game_result = harness.run_game(
            agent_one=agents[0],
            agent_two=agents[1],
            seed=SEED + games_completed
        )
        winner = game_result.winner

        # Determine outcome for learning agent
        if winner is None:
            outcome = 0.5  # Draw
        elif (agents[0] == agent and winner == Player.ONE) or \
             (agents[1] == agent and winner == Player.TWO):
            outcome = 1.0  # Win
        else:
            outcome = 0.0  # Loss

        # Update agent
        agent.observe_outcome(outcome)

        games_completed += 1

        # Progress update
        if games_completed % 500 == 0:
            elapsed = time.time() - start_time
            games_per_sec = games_completed / elapsed
            eta_sec = (NUM_TRAINING_GAMES - games_completed) / games_per_sec
            print(f"  [{games_completed:5d}/{NUM_TRAINING_GAMES}] "
                  f"({games_completed/NUM_TRAINING_GAMES*100:.1f}%) "
                  f"| {games_per_sec:.1f} games/sec "
                  f"| ETA: {eta_sec/60:.1f} min")

        # Evaluation checkpoint
        if games_completed >= next_eval:
            eval_result = evaluate_agent(agent, games_completed)
            eval_results.append(eval_result)
            next_eval += EVAL_INTERVAL

        # Save checkpoint
        if games_completed >= next_save:
            checkpoint_path = OUTPUT_DIR / f"agent_checkpoint_{games_completed}.json"
            agent.save(str(checkpoint_path))
            print(f"\nSaved checkpoint: {checkpoint_path}")
            next_save += SAVE_INTERVAL

    # Final evaluation
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")

    final_eval = evaluate_agent(agent, NUM_TRAINING_GAMES)
    eval_results.append(final_eval)

    # Save final agent
    final_path = OUTPUT_DIR / "agent_final.json"
    agent.save(str(final_path))
    print(f"\nFinal agent saved: {final_path}")

    # Save evaluation results
    import json
    eval_path = OUTPUT_DIR / "evaluation_results.json"
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"Evaluation results saved: {eval_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("Training Summary")
    print(f"{'='*60}")
    print(f"Initial win rate vs Heuristic: {eval_results[0]['win_rate_vs_heuristic']:.1%}")
    print(f"Final win rate vs Heuristic:   {eval_results[-1]['win_rate_vs_heuristic']:.1%}")
    print(f"Improvement: {(eval_results[-1]['win_rate_vs_heuristic'] - eval_results[0]['win_rate_vs_heuristic'])*100:.1f} percentage points")
    print()
    print(f"Target: 55%+ win rate vs Heuristic")
    if eval_results[-1]['win_rate_vs_heuristic'] >= 0.55:
        print("✓ TARGET ACHIEVED!")
    else:
        print(f"✗ Target not reached (gap: {(0.55 - eval_results[-1]['win_rate_vs_heuristic'])*100:.1f} pp)")

    print(f"\nTotal training time: {(time.time() - start_time)/60:.1f} minutes")
    print()


if __name__ == "__main__":
    main()
