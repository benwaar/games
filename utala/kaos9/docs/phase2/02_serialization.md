# Model Serialization

## What We Did

Built a standard framework for saving and loading learned AI models to/from JSON format.

**File:** `src/utala/learning/serialization.py`

**Key components:**

1. **ModelSerializer:** Static methods for save/load operations
2. **SerializableAgent:** Base class for agents with built-in serialization
3. **Standard schema:** All models follow common JSON structure

**Standard model format:**
```json
{
  "agent_name": "AssociativeMemory-v1",
  "agent_type": "associative_memory",
  "version": "1.0",
  "timestamp": "2026-03-23T10:30:00Z",
  "model_data": { ... },
  "hyperparameters": { ... },
  "performance": { "vs_random": 0.89 },
  "metadata": { ... }
}
```

## What It Means

**Why serialization matters:**

For production game integration, AI models must be:
- **Portable:** Move between Python research → game engines (Flutter, Unity, etc.)
- **Versionable:** Track model evolution, A/B test different versions
- **Inspectable:** Debug and understand what's been learned
- **Shareable:** Distribute trained models without sharing code/training data

**JSON benefits:**

- Human-readable (can inspect with any text editor)
- Language-agnostic (every platform has JSON parsers)
- Git-friendly (text diff shows model changes)
- Compact enough for version control (<1MB typical)

**Alternative approaches (and why we didn't use them):**

- **Pickle (Python):** Not portable across languages
- **Binary formats (Protocol Buffers):** Not human-readable
- **Neural network formats (ONNX):** Overkill for simple models

## Further Reading

**Serialization concepts:**
- [JSON specification](https://www.json.org/) - Official JSON docs
- [Model serialization best practices](https://www.tensorflow.org/guide/saved_model) - TensorFlow approach (we use simpler version)

**ML model deployment:**
- [Deploying Machine Learning Models](https://christophergs.com/machine%20learning/2019/03/17/how-to-deploy-machine-learning-models/) - Overview of model serving
- [Model versioning](https://neptune.ai/blog/version-control-for-machine-learning) - Why version models

**Data interchange:**
- [Choosing a serialization format](https://developers.google.com/protocol-buffers/docs/techniques#streaming) - Google's comparison

## Video Resources

**JSON basics:**
- [JSON in 10 Minutes](https://www.youtube.com/watch?v=iiADhChRriM) (10 min) - Complete JSON overview
- [Working with JSON Data](https://www.youtube.com/watch?v=9RRvvTMWB0c) (15 min) - Python JSON tutorial

**Model deployment:**
- [Machine Learning Model Deployment](https://www.youtube.com/watch?v=nhKdCa0qByI) (12 min) - Intro to ML deployment
- [Saving and Loading Models](https://www.youtube.com/watch?v=6Ka3ZXnBZu0) (8 min) - scikit-learn example (similar concept)

## Code Example

```python
from utala.learning.serialization import SerializableAgent

class MyAgent(SerializableAgent):
    def __init__(self):
        self.weights = [0.5, 0.3, 0.8]
        self.memory = []

    def to_dict(self):
        """Export learned state to dictionary."""
        return {
            'weights': self.weights,
            'memory': self.memory
        }

    def from_dict(self, data):
        """Import learned state from dictionary."""
        self.weights = data['weights']
        self.memory = data['memory']

# Create and train agent
agent = MyAgent()
# ... training happens ...

# Save to JSON
agent.save(
    agent_name="MyAgent-v1",
    agent_type="custom",
    version="1.0",
    hyperparameters={'learning_rate': 0.01},
    performance={'win_rate': 0.75},
    filepath="models/my_agent.json"
)

# Load from JSON
agent2 = MyAgent()
agent2.load("models/my_agent.json")
# agent2.weights == agent.weights
```

## Design Decisions

**Why require to_dict/from_dict?**

Forces clean separation:
- `to_dict()`: Extract only learned state (not temporary variables)
- `from_dict()`: Restore learned state (not hyperparameters)

This makes models portable—agents can load weights without knowing training history.

**Why include metadata?**

Provenance matters for production:
- Which dataset trained this model?
- What hyperparameters were used?
- What was the performance at save time?

Metadata enables model governance: auditing, debugging, rollback.

**Why JSONL for training data?**

One example per line:
- Streaming: process examples without loading full file
- Append-friendly: add more training data without rewriting file
- Fault-tolerant: partial file is still valid

Standard in ML data pipelines (TensorFlow, PyTorch use similar).

## Integration Guide

**For game developers:**

1. **Load model:** Use any JSON parser in your language
2. **Extract learned state:** `model['model_data']`
3. **Implement inference:** Reimplement agent logic using learned weights/memory
4. **Feature extraction:** Use same feature calculation (port from Python or document algorithm)

**Example (Flutter/Dart):**

```dart
import 'dart:convert';

// Load model
final jsonString = await File('model.json').readAsString();
final model = jsonDecode(jsonString);

// Extract weights
final weights = (model['model_data']['weights'] as List)
    .map((w) => w as double)
    .toList();

// Use weights in game logic
double evaluateState(GameState state) {
  final features = extractFeatures(state);  // Implement feature extraction
  return dotProduct(weights, features);
}
```
