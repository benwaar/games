"""Quick 20K DQN training to test Phase 3"""
import sys
sys.path.insert(0, '.')

# Modify constants
import train_dqn_agent as train_module
train_module.TOTAL_GAMES = 20000
train_module.EVAL_INTERVAL = 2000
train_module.CHECKPOINT_INTERVAL = 10000
train_module.STAGE_1_END = 7000
train_module.STAGE_2_END = 14000

print("="*70)
print("QUICK TRAINING: 20K games (~2-3 minutes)")
print("="*70)
print()

# Run main training
train_module.main()
