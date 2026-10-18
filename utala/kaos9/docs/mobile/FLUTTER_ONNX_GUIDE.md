# Flutter + ONNX Deployment Guide

**One Codebase → Web + Android + iOS**

---

## Overview

**Goal:** Deploy DQN agent to Flutter game running on all platforms

**Tech Stack:**
- Flutter (Dart) - cross-platform framework
- ONNX Runtime - ML inference engine
- Single codebase for web, Android, iOS

**DQN Position:** Medium AI (31% vs Heuristic)

---

## Quick Start (TL;DR)

```bash
# 1. Export PyTorch to ONNX (Python)
python export_dqn_to_onnx.py  # → dqn_mobile.onnx (35KB)

# 2. Add to Flutter project
flutter pub add onnxruntime

# 3. Use in Dart
final agent = await DQNAgent.load('assets/dqn_mobile.onnx');
final action = await agent.selectAction(state, legalActions);

# 4. Deploy everywhere
flutter build web
flutter build apk
flutter build ios
```

---

## Step 1: Export PyTorch to ONNX (30 minutes)

### Create Export Script

```python
# export_dqn_to_onnx.py
import torch
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from src.utala.deep_learning.dqn_agent import DQNAgent

def export_dqn():
    """Export trained DQN to ONNX format."""
    print("Loading DQN agent...")
    agent = DQNAgent.load("results/dqn/dqn_final.pth")
    agent.set_training(False)
    agent.q_network.eval()

    # Dummy input (batch_size=1, features=53)
    dummy_input = torch.randn(1, 53)

    print("Exporting to ONNX...")
    torch.onnx.export(
        agent.q_network,
        dummy_input,
        "models/dqn_mobile.onnx",
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['state_features'],
        output_names=['q_values'],
        dynamic_axes={
            'state_features': {0: 'batch_size'},
            'q_values': {0: 'batch_size'}
        }
    )

    # Validate
    onnx_model = onnx.load("models/dqn_mobile.onnx")
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model exported and validated")

    # Quantize for mobile (FP32 → INT8)
    print("Quantizing model...")
    quantize_dynamic(
        model_input="models/dqn_mobile.onnx",
        model_output="models/dqn_mobile_quantized.onnx",
        weight_type=QuantType.QInt8
    )

    import os
    original_size = os.path.getsize("models/dqn_mobile.onnx") / 1024
    quantized_size = os.path.getsize("models/dqn_mobile_quantized.onnx") / 1024

    print(f"✓ Quantized model saved")
    print(f"  Original: {original_size:.1f} KB")
    print(f"  Quantized: {quantized_size:.1f} KB")
    print(f"  Reduction: {(1 - quantized_size/original_size)*100:.1f}%")

if __name__ == "__main__":
    export_dqn()
```

### Run Export

```bash
cd /Users/DBenoy/src/_research/_benwaar/utala-kaos-9-ai
python export_dqn_to_onnx.py
```

**Output:**
- `models/dqn_mobile.onnx` (~135KB)
- `models/dqn_mobile_quantized.onnx` (~35KB) ← Use this for mobile

---

## Step 2: Flutter Project Setup (1 hour)

### Add ONNX Runtime Dependency

```yaml
# pubspec.yaml
name: utala_kaos_9
description: Tactical grid combat game

dependencies:
  flutter:
    sdk: flutter
  onnxruntime: ^1.16.0  # Add this

flutter:
  assets:
    - assets/models/dqn_mobile_quantized.onnx  # Your ONNX model
```

### Install

```bash
flutter pub get
```

---

## Step 3: Create DQN Agent Wrapper (2 hours)

### Feature Extraction (Port from Python)

```dart
// lib/ai/feature_extractor.dart
import '../game/game_state.dart';

class StateFeatureExtractor {
  static const int featureDim = 53;

  List<double> extract(GameState state, Player player) {
    final features = List<double>.filled(featureDim, 0.0);
    int idx = 0;

    // Material features (18): count of each rocketman (2-10) for both players
    for (int value = 2; value <= 10; value++) {
      features[idx++] = _countPieces(state, player, value).toDouble();
      features[idx++] = _countPieces(state, player.opponent, value).toDouble();
    }

    // Control features (9): who controls each square
    for (int row = 0; row < 3; row++) {
      for (int col = 0; col < 3; col++) {
        features[idx++] = _getControlValue(state, row, col, player);
      }
    }

    // Position features (9): sum of piece values at each position
    for (int row = 0; row < 3; row++) {
      for (int col = 0; col < 3; col++) {
        features[idx++] = _getPositionValue(state, row, col, player);
      }
    }

    // Threat features (7)
    features[idx++] = _countThreats(state, player).toDouble();
    features[idx++] = _countThreats(state, player.opponent).toDouble();
    features[idx++] = _hasCenter(state, player) ? 1.0 : 0.0;
    features[idx++] = _hasCenter(state, player.opponent) ? 1.0 : 0.0;
    features[idx++] = _countCorners(state, player).toDouble();
    features[idx++] = _countCorners(state, player.opponent).toDouble();
    features[idx++] = _countEdges(state, player).toDouble();

    // Advanced features (10)
    features[idx++] = _materialAdvantage(state, player);
    features[idx++] = _controlAdvantage(state, player);
    features[idx++] = _centerContested(state) ? 1.0 : 0.0;
    features[idx++] = _strongPieceInCenter(state, player) ? 1.0 : 0.0;
    features[idx++] = _weakPieceInCorner(state, player) ? 1.0 : 0.0;
    features[idx++] = _kaosCardsRemaining(state, player).toDouble();
    features[idx++] = _kaosCardsRemaining(state, player.opponent).toDouble();
    features[idx++] = _highValueKaosAvailable(state, player) ? 1.0 : 0.0;
    features[idx++] = _lowValueKaosAvailable(state, player) ? 1.0 : 0.0;
    features[idx++] = state.phase == Phase.placement ? 1.0 : 0.0;

    assert(idx == featureDim);
    return features;
  }

  // Helper methods (port from Python)
  int _countPieces(GameState state, Player player, int value) {
    // Port from features.py
    return 0; // TODO: implement
  }

  double _getControlValue(GameState state, int row, int col, Player player) {
    // Port from features.py
    return 0.0; // TODO: implement
  }

  // ... implement remaining helpers
}
```

### DQN Agent Class

```dart
// lib/ai/dqn_agent.dart
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';
import '../game/game_state.dart';
import 'feature_extractor.dart';

class DQNAgent {
  late OrtSession _session;
  final StateFeatureExtractor _featureExtractor = StateFeatureExtractor();
  bool _isInitialized = false;

  /// Load model from assets
  Future<void> initialize() async {
    if (_isInitialized) return;

    // Load ONNX model from assets
    final modelBytes = await rootBundle.load('assets/models/dqn_mobile_quantized.onnx');
    final bytes = modelBytes.buffer.asUint8List();

    // Create ONNX session
    final sessionOptions = OrtSessionOptions();
    _session = OrtSession.fromBuffer(bytes, sessionOptions);

    _isInitialized = true;
    print('✓ DQN Agent initialized');
  }

  /// Select best action given game state
  Future<int> selectAction(GameState state, List<int> legalActions) async {
    if (!_isInitialized) {
      throw StateError('DQN Agent not initialized. Call initialize() first.');
    }

    // Extract features
    final features = _featureExtractor.extract(state, state.currentPlayer);

    // Create input tensor
    final inputOrt = OrtValueTensor.createTensorWithDataList(
      [features],  // Batch size 1
      [1, StateFeatureExtractor.featureDim],
    );

    // Run inference
    final runOptions = OrtRunOptions();
    final outputs = await _session.runAsync(
      runOptions,
      {'state_features': inputOrt},
    );

    // Extract Q-values
    final qValuesTensor = outputs?[0]?.value as List<List<double>>;
    final qValues = qValuesTensor[0];  // Batch size 1, so take first

    // Select best legal action
    int bestAction = legalActions[0];
    double bestQ = double.negativeInfinity;

    for (final action in legalActions) {
      if (qValues[action] > bestQ) {
        bestQ = qValues[action];
        bestAction = action;
      }
    }

    // Cleanup
    inputOrt.release();
    runOptions.release();
    outputs?.forEach((o) => o?.release());

    return bestAction;
  }

  /// Dispose resources
  void dispose() {
    if (_isInitialized) {
      _session.release();
      _isInitialized = false;
    }
  }
}
```

---

## Step 4: Integration into Game (1 hour)

### AI Manager

```dart
// lib/ai/ai_manager.dart
import 'dqn_agent.dart';
import '../game/game_state.dart';

enum AIDifficulty { easy, medium, hard, expert }

class AIManager {
  DQNAgent? _dqnAgent;

  Future<void> initialize() async {
    _dqnAgent = DQNAgent();
    await _dqnAgent!.initialize();
  }

  Future<int> getAIMove(GameState state, AIDifficulty difficulty) async {
    final legalActions = state.getLegalActions();

    switch (difficulty) {
      case AIDifficulty.easy:
        // Random agent
        return (legalActions..shuffle()).first;

      case AIDifficulty.medium:
        // DQN agent (neural network)
        return await _dqnAgent!.selectAction(state, legalActions);

      case AIDifficulty.hard:
        // Linear Value agent (TODO: implement)
        throw UnimplementedError('Linear Value not yet ported');

      case AIDifficulty.expert:
        // Heuristic agent (TODO: implement)
        throw UnimplementedError('Heuristic not yet ported');
    }
  }

  void dispose() {
    _dqnAgent?.dispose();
  }
}
```

### Game Screen Usage

```dart
// lib/screens/game_screen.dart
import 'package:flutter/material.dart';
import '../ai/ai_manager.dart';
import '../game/game_state.dart';

class GameScreen extends StatefulWidget {
  final AIDifficulty difficulty;

  const GameScreen({Key? key, required this.difficulty}) : super(key: key);

  @override
  State<GameScreen> createState() => _GameScreenState();
}

class _GameScreenState extends State<GameScreen> {
  late AIManager _aiManager;
  late GameState _gameState;
  bool _isAIThinking = false;

  @override
  void initState() {
    super.initState();
    _initializeGame();
  }

  Future<void> _initializeGame() async {
    _aiManager = AIManager();
    await _aiManager.initialize();
    _gameState = GameState.initial();
    setState(() {});
  }

  Future<void> _makeAIMove() async {
    if (_isAIThinking) return;

    setState(() => _isAIThinking = true);

    // Get AI decision
    final action = await _aiManager.getAIMove(_gameState, widget.difficulty);

    // Apply action
    _gameState = _gameState.applyAction(action);

    setState(() => _isAIThinking = false);
  }

  @override
  void dispose() {
    _aiManager.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Playing vs ${widget.difficulty.name.toUpperCase()} AI'),
      ),
      body: Column(
        children: [
          // Game board
          Expanded(child: GameBoard(state: _gameState)),

          // AI thinking indicator
          if (_isAIThinking)
            const LinearProgressIndicator(),
        ],
      ),
    );
  }
}
```

---

## Step 5: Platform-Specific Build (30 minutes each)

### Web

```bash
flutter build web --release
# Deploy to: build/web/
# Serve with: python -m http.server 8000 --directory build/web
```

**Web Notes:**
- ONNX Runtime uses WebAssembly
- First load downloads WASM (~10MB)
- Subsequent loads use cache
- Inference: ~5-10ms on desktop browsers

### Android

```bash
flutter build apk --release
# Output: build/app/outputs/flutter-apk/app-release.apk
```

**Android Notes:**
- ONNX Runtime native library included (~10MB)
- Inference: ~2-3ms on modern devices
- Works on Android 5.0+ (API 21+)

### iOS

```bash
flutter build ios --release
# Open in Xcode to sign and deploy
```

**iOS Notes:**
- ONNX Runtime framework included (~10MB)
- Inference: ~2-3ms on iPhone 11+
- Requires iOS 12.0+

---

## Complete Project Structure

```
utala_kaos_9_flutter/
├── lib/
│   ├── ai/
│   │   ├── dqn_agent.dart           # ONNX inference
│   │   ├── feature_extractor.dart   # Port from Python
│   │   └── ai_manager.dart          # Difficulty selector
│   ├── game/
│   │   ├── game_state.dart
│   │   ├── action.dart
│   │   └── rules.dart
│   ├── screens/
│   │   ├── menu_screen.dart
│   │   └── game_screen.dart
│   └── main.dart
├── assets/
│   └── models/
│       └── dqn_mobile_quantized.onnx  # 35KB model
├── pubspec.yaml                        # Dependencies
└── README.md
```

---

## Performance Benchmarks

### Model Size
- **ONNX quantized:** 35KB
- **ONNX Runtime:** ~10MB (embedded in app)
- **Total app size increase:** ~10MB

### Inference Speed

| Platform | Device | Inference Time | FPS Impact |
|----------|--------|----------------|------------|
| **Web** | Desktop Chrome | ~5-10ms | Negligible |
| **Web** | Mobile Chrome | ~10-20ms | Minimal |
| **Android** | Pixel 6 | ~2-3ms | None |
| **Android** | Budget phone | ~5-10ms | Minimal |
| **iOS** | iPhone 13 | ~2-3ms | None |
| **iOS** | iPhone X | ~3-5ms | Minimal |

### Memory Usage
- **Model loading:** ~500KB RAM
- **ONNX Runtime:** ~10MB RAM
- **Per inference:** ~1MB temporary
- **Total:** ~11MB (acceptable)

---

## Testing Checklist

### Before Deployment

- [ ] ONNX model exported and quantized
- [ ] Model file in `assets/models/`
- [ ] `pubspec.yaml` updated
- [ ] Feature extraction ported correctly
- [ ] DQN agent returns valid actions
- [ ] Web build runs in browser
- [ ] Android APK installs and runs
- [ ] iOS build passes validation
- [ ] Performance acceptable on target devices
- [ ] Memory usage reasonable

---

## Troubleshooting

### "ONNX Runtime not found"
```bash
flutter clean
flutter pub get
flutter build [platform] --release
```

### "Model file not found"
Check `pubspec.yaml` assets section:
```yaml
flutter:
  assets:
    - assets/models/dqn_mobile_quantized.onnx
```

### Slow inference on web
- Use quantized model (INT8)
- First load downloads WASM (one-time)
- Consider service worker for caching

### Large app size
- Quantized model is 35KB (tiny)
- ONNX Runtime adds ~10MB (unavoidable)
- Consider making DQN optional download

---

## Next Steps

### Phase 1: Get DQN Working (Week 1)
1. Export ONNX model
2. Create Flutter DQN wrapper
3. Port feature extraction
4. Test on one platform

### Phase 2: Port Other Agents (Week 2)
1. Linear Value (pure Dart, no ONNX)
2. Heuristic (pure Dart, no ONNX)
3. Complete AI difficulty ladder

### Phase 3: Polish (Week 3)
1. UI for difficulty selection
2. Performance optimization
3. Test on all platforms
4. Publish

---

## Comparison: ONNX vs Pure Dart

| Agent | Implementation | Size | Speed | Platform Support |
|-------|---------------|------|-------|-----------------|
| **DQN** | ONNX | 35KB + 10MB runtime | ~3ms | Web, Android, iOS |
| **Linear Value** | Pure Dart | 1KB JSON | <1ms | Web, Android, iOS |
| **Heuristic** | Pure Dart | 0KB | <1ms | Web, Android, iOS |

**Recommendation:**
- Use ONNX for DQN (learning opportunity)
- Use pure Dart for Linear Value and Heuristic (simpler, faster)
- One Flutter codebase for everything

---

## Summary

**✅ Yes, one Flutter codebase deploys to web, Android, and iOS**

**Steps:**
1. Export PyTorch DQN to ONNX (30 min)
2. Add `onnxruntime` to Flutter (5 min)
3. Create DQN agent wrapper in Dart (2 hours)
4. Build for each platform (30 min each)

**Result:**
- Medium difficulty AI (DQN, 31%)
- Single codebase
- Runs on all platforms
- ~10MB app size increase
- 2-3ms inference on mobile

**Timeline:** 1 week to get DQN working on all platforms
