# Deep Learning Model Deployment to Mobile

**Learning Plan: Export DQN Agent to Mobile Platforms**

---

## Overview

**Goal:** Learn to deploy deep learning models to mobile apps using open, cross-platform formats.

**Target Model:** DQN Agent (31% vs Heuristic, 34,518 parameters)

**Difficulty Position:** Medium AI (sits between Easy and Hard)

---

## Where DQN Fits in Difficulty Ladder

```
┌─────────────────────────────────────────────────────┐
│ Easy      │ MEDIUM (DL) │ Hard         │ Expert     │
│ Random    │ DQN         │ Linear Value │ Heuristic  │
│ 20%       │ 31%         │ 42%          │ 58%+       │
└─────────────────────────────────────────────────────┘
   Learn       AI/ML         Challenge      Master
```

**Marketing:** "Challenge our neural network AI"

**Technical Selling Point:** Shows off ML deployment skills, modern tech stack

**Player Experience:** Mid-difficulty with "smart AI" branding

---

## Mobile ML Format Options

### Option 1: ONNX (Open Neural Network Exchange) ⭐ Recommended
**Pros:**
- ✅ Open standard, vendor-neutral
- ✅ Works on Android, iOS, Web
- ✅ ONNX Runtime available for all platforms
- ✅ Single export, works everywhere
- ✅ Good performance

**Cons:**
- Requires ONNX Runtime dependency (~10MB)

**Best For:** Cross-platform apps (React Native, Flutter, or native)

---

### Option 2: TensorFlow Lite
**Pros:**
- ✅ Google-backed, well-documented
- ✅ Small runtime (~1MB)
- ✅ Good Android support
- ✅ Hardware acceleration (GPU/NPU)

**Cons:**
- Harder to use on iOS (CoreML better there)
- PyTorch → TFLite conversion more complex

**Best For:** Android-first apps

---

### Option 3: CoreML (iOS Only)
**Pros:**
- ✅ Native iOS integration
- ✅ Excellent performance
- ✅ Hardware acceleration
- ✅ Xcode integration

**Cons:**
- ❌ iOS only
- Need separate solution for Android

**Best For:** iOS-only apps or iOS-specific optimization

---

### Option 4: PyTorch Mobile
**Pros:**
- ✅ Direct PyTorch deployment
- ✅ No conversion needed
- ✅ Full PyTorch API

**Cons:**
- ❌ Large runtime (~50MB)
- Limited hardware acceleration
- Overkill for inference-only

**Best For:** Research/prototyping, not production

---

## Recommended Approach: ONNX Export

**Why ONNX:**
- One export works on Android, iOS, Web
- Open standard, future-proof
- Good performance and tooling
- Industry standard for ML deployment

---

## Implementation Plan

### Phase 1: Export PyTorch to ONNX (1-2 days)

#### Step 1: Install ONNX Tools
```bash
pip install onnx onnxruntime torch
```

#### Step 2: Export DQN Network
```python
# export_dqn_to_onnx.py
import torch
import onnx
from src.utala.deep_learning.dqn_agent import DQNAgent

# Load trained DQN
agent = DQNAgent.load("results/dqn/dqn_final.pth")
agent.set_training(False)

# Create dummy input (53 features)
dummy_input = torch.randn(1, 53)

# Export to ONNX
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

print("✓ Exported to models/dqn_mobile.onnx")

# Verify export
onnx_model = onnx.load("models/dqn_mobile.onnx")
onnx.checker.check_model(onnx_model)
print("✓ ONNX model validated")
```

#### Step 3: Test ONNX Inference
```python
# test_onnx_inference.py
import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession("models/dqn_mobile.onnx")

# Test inference
state = np.random.randn(1, 53).astype(np.float32)
q_values = session.run(
    ['q_values'],
    {'state_features': state}
)[0]

print(f"Q-values shape: {q_values.shape}")  # Should be (1, 86)
print(f"Best action: {q_values.argmax()}")
```

#### Step 4: Optimize Model (Optional)
```python
# Quantize to INT8 for smaller size, faster inference
from onnxruntime.quantization import quantize_dynamic

quantize_dynamic(
    model_input="models/dqn_mobile.onnx",
    model_output="models/dqn_mobile_quantized.onnx",
    weight_type=QuantType.QInt8
)

# Result: ~4x smaller model size
```

**Deliverables:**
- ✅ `models/dqn_mobile.onnx` (~135KB unquantized)
- ✅ `models/dqn_mobile_quantized.onnx` (~35KB quantized)
- ✅ Validation script
- ✅ Performance benchmark

---

### Phase 2: Android Integration (2-3 days)

#### Step 1: Add ONNX Runtime Dependency
```gradle
// app/build.gradle
dependencies {
    implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.16.0'
}
```

#### Step 2: Implement DQN Agent Wrapper
```kotlin
// DQNAgent.kt
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession

class DQNAgent(private val modelPath: String) {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        // Load ONNX model from assets
        val modelBytes = context.assets.open(modelPath).readBytes()
        session = env.createSession(modelBytes)
    }

    fun selectAction(stateFeatures: FloatArray, legalActions: IntArray): Int {
        // Create input tensor
        val inputTensor = OnnxTensor.createTensor(
            env,
            arrayOf(stateFeatures),
            longArrayOf(1, 53)
        )

        // Run inference
        val outputs = session.run(mapOf("state_features" to inputTensor))
        val qValues = outputs[0].value as Array<FloatArray>

        // Select best legal action
        var bestAction = legalActions[0]
        var bestQ = Float.NEGATIVE_INFINITY

        for (action in legalActions) {
            if (qValues[0][action] > bestQ) {
                bestQ = qValues[0][action]
                bestAction = action
            }
        }

        return bestAction
    }

    fun close() {
        session.close()
    }
}
```

#### Step 3: Feature Extraction (Port from Python)
```kotlin
// FeatureExtractor.kt
class StateFeatureExtractor {
    fun extract(state: GameState, player: Player): FloatArray {
        // Port logic from src/utala/learning/features.py
        // 53 features: material, control, position, threats
        val features = FloatArray(53)

        // Material features (18)
        // ... port Python logic ...

        // Control features (9)
        // ... port Python logic ...

        // Position features (9)
        // ... port Python logic ...

        // Threat features (7)
        // ... port Python logic ...

        // Advanced features (10)
        // ... port Python logic ...

        return features
    }
}
```

#### Step 4: Use in Game
```kotlin
// GameActivity.kt
class GameActivity : AppCompatActivity() {
    private lateinit var dqnAgent: DQNAgent

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize DQN agent
        dqnAgent = DQNAgent("dqn_mobile_quantized.onnx")
    }

    fun getAIMove(gameState: GameState): Int {
        // Extract features
        val features = featureExtractor.extract(gameState, Player.TWO)

        // Get legal actions
        val legalActions = gameState.getLegalActions()

        // Select action
        return dqnAgent.selectAction(features, legalActions)
    }

    override fun onDestroy() {
        super.onDestroy()
        dqnAgent.close()
    }
}
```

**Deliverables:**
- ✅ Android ONNX inference wrapper
- ✅ Feature extraction in Kotlin
- ✅ Integration example
- ✅ Performance profiling

---

### Phase 3: iOS Integration (2-3 days)

#### Option A: ONNX Runtime (Cross-Platform)
```swift
// DQNAgent.swift
import onnxruntime_objc

class DQNAgent {
    private var session: ORTSession?

    init(modelPath: String) throws {
        let env = try ORTEnv(loggingLevel: .warning)
        let options = try ORTSessionOptions()
        session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: options)
    }

    func selectAction(stateFeatures: [Float], legalActions: [Int]) throws -> Int {
        // Create input tensor
        let inputShape: [NSNumber] = [1, 53]
        let inputTensor = try ORTValue(
            tensorData: NSMutableData(bytes: stateFeatures, length: stateFeatures.count * 4),
            elementType: .float,
            shape: inputShape
        )

        // Run inference
        let outputs = try session?.run(
            withInputs: ["state_features": inputTensor],
            outputNames: ["q_values"],
            runOptions: nil
        )

        // Extract Q-values
        guard let qValuesData = try outputs?["q_values"]?.tensorData() as Data? else {
            throw NSError(domain: "DQN", code: 1, userInfo: nil)
        }

        let qValues = qValuesData.withUnsafeBytes {
            Array(UnsafeBufferPointer<Float>(start: $0, count: 86))
        }

        // Select best legal action
        var bestAction = legalActions[0]
        var bestQ: Float = -Float.infinity

        for action in legalActions {
            if qValues[action] > bestQ {
                bestQ = qValues[action]
                bestAction = action
            }
        }

        return bestAction
    }
}
```

#### Option B: CoreML (iOS-Optimized)
```bash
# Convert ONNX to CoreML
pip install coremltools

python3 -c "
import coremltools as ct
onnx_model = ct.converters.onnx.convert(
    model='models/dqn_mobile.onnx',
    minimum_ios_deployment_target='14.0'
)
onnx_model.save('models/DQNAgent.mlmodel')
"
```

```swift
// DQNAgent.swift (CoreML version)
import CoreML

class DQNAgent {
    private let model: DQNAgentModel

    init() throws {
        model = try DQNAgentModel(configuration: MLModelConfiguration())
    }

    func selectAction(stateFeatures: [Float], legalActions: [Int]) throws -> Int {
        // Create input
        let input = try MLMultiArray(shape: [1, 53], dataType: .float32)
        for (i, value) in stateFeatures.enumerated() {
            input[i] = NSNumber(value: value)
        }

        // Run inference
        let prediction = try model.prediction(state_features: input)
        let qValues = prediction.q_values

        // Select best legal action
        var bestAction = legalActions[0]
        var bestQ: Float = -Float.infinity

        for action in legalActions {
            let q = qValues[action].floatValue
            if q > bestQ {
                bestQ = q
                bestAction = action
            }
        }

        return bestAction
    }
}
```

**Deliverables:**
- ✅ iOS ONNX Runtime wrapper (cross-platform)
- ✅ CoreML conversion (iOS-optimized)
- ✅ Swift feature extraction
- ✅ Integration example

---

### Phase 4: Cross-Platform (React Native / Flutter) (3-4 days)

#### React Native with ONNX
```javascript
// Install: npm install onnxruntime-react-native
import { InferenceSession, Tensor } from 'onnxruntime-react-native';

class DQNAgent {
  constructor() {
    this.session = null;
  }

  async initialize() {
    this.session = await InferenceSession.create(
      './models/dqn_mobile.onnx'
    );
  }

  async selectAction(stateFeatures, legalActions) {
    // Create input tensor
    const inputTensor = new Tensor(
      'float32',
      Float32Array.from(stateFeatures),
      [1, 53]
    );

    // Run inference
    const feeds = { state_features: inputTensor };
    const results = await this.session.run(feeds);
    const qValues = results.q_values.data;

    // Select best legal action
    let bestAction = legalActions[0];
    let bestQ = -Infinity;

    for (const action of legalActions) {
      if (qValues[action] > bestQ) {
        bestQ = qValues[action];
        bestAction = action;
      }
    }

    return bestAction;
  }
}
```

#### Flutter with ONNX
```dart
// pubspec.yaml: onnxruntime: ^1.16.0
import 'package:onnxruntime/onnxruntime.dart';

class DQNAgent {
  late OrtSession _session;

  Future<void> initialize() async {
    final sessionOptions = OrtSessionOptions();
    _session = OrtSession.fromAsset(
      'assets/dqn_mobile.onnx',
      sessionOptions,
    );
  }

  Future<int> selectAction(List<double> stateFeatures, List<int> legalActions) async {
    // Create input
    final inputOrt = OrtValueTensor.createTensorWithDataList(
      [stateFeatures],
      [1, 53],
    );

    // Run inference
    final outputs = await _session.runAsync(
      OrtRunOptions(),
      {'state_features': inputOrt},
    );

    final qValues = outputs?[0]?.value as List<List<double>>;

    // Select best legal action
    int bestAction = legalActions[0];
    double bestQ = double.negativeInfinity;

    for (final action in legalActions) {
      if (qValues[0][action] > bestQ) {
        bestQ = qValues[0][action];
        bestAction = action;
      }
    }

    return bestAction;
  }
}
```

**Deliverables:**
- ✅ React Native ONNX wrapper
- ✅ Flutter ONNX wrapper
- ✅ JavaScript/Dart feature extraction
- ✅ Example app integration

---

## Learning Path Structure

### Week 1: ONNX Export & Validation
**Skills Learned:**
- PyTorch to ONNX conversion
- Model validation and testing
- ONNX format understanding
- Performance benchmarking

**Deliverables:**
- Export script
- Validation tests
- Quantized models
- Performance report

---

### Week 2: Android Implementation
**Skills Learned:**
- ONNX Runtime on Android
- Kotlin ML integration
- Feature extraction porting
- Mobile inference optimization

**Deliverables:**
- Android DQN agent class
- Feature extractor in Kotlin
- Demo app
- Performance profiling

---

### Week 3: iOS Implementation
**Skills Learned:**
- ONNX Runtime on iOS / CoreML
- Swift ML integration
- Cross-platform considerations
- iOS optimization techniques

**Deliverables:**
- iOS DQN agent class
- Feature extractor in Swift
- Demo app
- Comparison: ONNX vs CoreML

---

### Week 4 (Optional): Cross-Platform
**Skills Learned:**
- React Native / Flutter ML
- Unified codebase strategies
- Platform-specific optimizations
- Deployment at scale

**Deliverables:**
- React Native / Flutter wrapper
- Unified API design
- Multi-platform demo
- Best practices guide

---

## Performance Expectations

### Model Size
- **PyTorch (.pth):** ~500KB
- **ONNX:** ~135KB
- **ONNX Quantized (INT8):** ~35KB
- **CoreML:** ~140KB

### Inference Speed (Mobile Device)
- **iPhone 13 / Pixel 6:** ~2-3ms
- **iPhone 11 / Pixel 4:** ~3-5ms
- **Budget phones:** ~5-10ms

### Memory Usage
- **Model:** ~500KB RAM
- **ONNX Runtime:** ~10MB RAM
- **Total:** ~10.5MB (acceptable for mobile)

### Battery Impact
- **Per inference:** Negligible (<0.001% battery)
- **Per game:** ~30-40 inferences = ~0.03% battery
- **Minimal impact** for typical gameplay

---

## Comparison: DL vs Linear Value

| Metric | Linear Value | DQN (ONNX) |
|--------|-------------|-----------|
| **Model Size** | 1KB | 35KB (quantized) |
| **Runtime Dependency** | None | 10MB |
| **Inference Speed** | <1ms | 2-3ms |
| **Win Rate** | 42% | 31% |
| **Platform Support** | All (pure math) | All (needs ONNX) |
| **Learning Value** | Minimal | High (ML deployment) |
| **Player Experience** | Smart tactical AI | "Neural network AI" |

**Trade-off:** DQN is weaker but teaches valuable ML deployment skills

---

## Project Structure

```
mobile-ml-deployment/
├── 1_export/
│   ├── export_dqn_to_onnx.py
│   ├── test_onnx_inference.py
│   └── quantize_model.py
├── 2_android/
│   ├── DQNAgent.kt
│   ├── FeatureExtractor.kt
│   └── build.gradle
├── 3_ios/
│   ├── DQNAgent.swift
│   ├── FeatureExtractor.swift
│   └── convert_to_coreml.py
├── 4_cross_platform/
│   ├── react_native/
│   │   └── DQNAgent.js
│   └── flutter/
│       └── dqn_agent.dart
├── models/
│   ├── dqn_mobile.onnx
│   ├── dqn_mobile_quantized.onnx
│   └── DQNAgent.mlmodel
└── docs/
    ├── ONNX_GUIDE.md
    ├── ANDROID_GUIDE.md
    ├── IOS_GUIDE.md
    └── PERFORMANCE.md
```

---

## Success Criteria

### Phase 1: Export (Done)
- ✅ ONNX model exported
- ✅ Validation passes
- ✅ Inference matches PyTorch
- ✅ Quantized model works

### Phase 2: Android (Done)
- ✅ ONNX Runtime integrated
- ✅ DQN agent runs on device
- ✅ <5ms inference on mid-range phone
- ✅ Plays full game correctly

### Phase 3: iOS (Done)
- ✅ ONNX Runtime or CoreML working
- ✅ DQN agent runs on device
- ✅ <5ms inference
- ✅ App Store ready

### Phase 4: Polish (Optional)
- ✅ Cross-platform wrapper
- ✅ Documentation complete
- ✅ Demo app published
- ✅ Blog post / tutorial

---

## Resources

### ONNX
- Official docs: https://onnx.ai/
- PyTorch export: https://pytorch.org/docs/stable/onnx.html
- ONNX Runtime: https://onnxruntime.ai/

### Android
- ONNX Runtime Android: https://onnxruntime.ai/docs/tutorials/mobile/
- TensorFlow Lite: https://www.tensorflow.org/lite/android

### iOS
- ONNX Runtime iOS: https://onnxruntime.ai/docs/tutorials/mobile/
- CoreML Tools: https://coremltools.readme.io/

### Cross-Platform
- React Native ONNX: https://github.com/microsoft/onnxruntime/tree/main/js/react_native
- Flutter ONNX: https://pub.dev/packages/onnxruntime

---

## Learning Outcomes

By completing this plan, you'll learn:

1. **Model Export:** PyTorch → ONNX conversion
2. **Optimization:** Quantization, pruning
3. **Mobile ML:** ONNX Runtime, TFLite, CoreML
4. **Platform-Specific:** Android (Kotlin) and iOS (Swift) integration
5. **Cross-Platform:** React Native / Flutter ML
6. **Performance:** Profiling, optimization, battery impact
7. **Deployment:** App Store / Play Store ML guidelines
8. **Best Practices:** Model versioning, fallback strategies

---

## Timeline

### Minimal (2 weeks)
- Week 1: ONNX export + Android
- Week 2: iOS + testing

### Complete (4 weeks)
- Week 1: ONNX export + validation
- Week 2: Android implementation
- Week 3: iOS implementation
- Week 4: Cross-platform + polish

### With Game Integration (6 weeks)
- Week 1-4: Above
- Week 5: Full game integration
- Week 6: Testing + App Store submission

---

## Recommendation

**Start with Phase 1 (ONNX Export)**
1. Export DQN to ONNX (1 day)
2. Validate inference (1 day)
3. Quantize for mobile (1 day)

**Then choose platform:**
- Android-first → Phase 2
- iOS-first → Phase 3
- React Native/Flutter → Phase 4

**DQN Position in Game:**
- **Medium difficulty** (31% vs Heuristic)
- Between Easy (20%) and Hard (42%)
- Market as "Neural Network AI" / "Smart AI"
- Good learning project even though Linear Value is better

---

## Summary

**DQN Agent Position:** Medium AI (31% difficulty)

**Value Proposition:**
- ❌ Not the best performing (Linear Value 42% is better)
- ✅ Excellent learning opportunity (ML deployment)
- ✅ Modern tech showcase (neural networks)
- ✅ Marketing appeal ("AI-powered opponent")

**Best Format:** ONNX (cross-platform, open standard)

**Timeline:** 2-4 weeks depending on scope

**Outcome:** Deployable neural network AI on mobile + valuable ML deployment skills
