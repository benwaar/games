"""
Export DQN Agent to ONNX Format for Mobile Deployment

Converts trained PyTorch DQN to ONNX format with INT8 quantization
for deployment to Flutter (web, Android, iOS).
"""

import torch
import onnx
from pathlib import Path
from src.utala.deep_learning.dqn_agent import DQNAgent


def export_dqn():
    """Export trained DQN to ONNX format with quantization."""
    print("="*70)
    print("Exporting DQN to ONNX for Mobile Deployment")
    print("="*70)

    # Load trained DQN
    print("\n1. Loading trained DQN agent...")
    model_path = "results/dqn/dqn_final.pth"
    if not Path(model_path).exists():
        print(f"❌ Error: Model not found at {model_path}")
        print("   Run train_dqn_agent.py first to train the model.")
        return False

    agent = DQNAgent.load(model_path)
    agent.set_training(False)
    agent.q_network.eval()
    print(f"✓ Loaded model from {model_path}")
    print(f"  Parameters: {sum(p.numel() for p in agent.q_network.parameters()):,}")

    # Create dummy input
    print("\n2. Creating dummy input...")
    dummy_input = torch.randn(1, 53)  # Batch size 1, 53 features
    print(f"  Input shape: {dummy_input.shape}")

    # Export to ONNX
    print("\n3. Exporting to ONNX...")
    output_path = "models/dqn_mobile.onnx"
    Path("models").mkdir(exist_ok=True)

    torch.onnx.export(
        agent.q_network,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['state_features'],
        output_names=['q_values'],
        dynamic_axes={
            'state_features': {0: 'batch_size'},
            'q_values': {0: 'batch_size'}
        },
        verbose=False
    )
    print(f"✓ Exported to {output_path}")

    # Validate ONNX model
    print("\n4. Validating ONNX model...")
    try:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model is valid")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

    # Test inference with ONNX Runtime
    print("\n5. Testing ONNX inference...")
    try:
        import onnxruntime as ort
        import numpy as np

        session = ort.InferenceSession(output_path)
        test_input = np.random.randn(1, 53).astype(np.float32)
        q_values = session.run(['q_values'], {'state_features': test_input})[0]
        print(f"✓ Inference successful")
        print(f"  Output shape: {q_values.shape}")
        print(f"  Sample Q-values: {q_values[0][:5]}")
    except ImportError:
        print("⚠️  onnxruntime not installed, skipping inference test")
        print("   Install with: pip install onnxruntime")
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        return False

    # Quantize model (FP32 → INT8)
    print("\n6. Quantizing model for mobile...")
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        quantized_path = "models/dqn_mobile_quantized.onnx"
        quantize_dynamic(
            model_input=output_path,
            model_output=quantized_path,
            weight_type=QuantType.QInt8
        )
        print(f"✓ Quantized model saved to {quantized_path}")

        # Compare sizes
        import os
        original_size = os.path.getsize(output_path) / 1024
        quantized_size = os.path.getsize(quantized_path) / 1024
        reduction = (1 - quantized_size/original_size) * 100

        print(f"\n  Original (FP32):  {original_size:6.1f} KB")
        print(f"  Quantized (INT8): {quantized_size:6.1f} KB")
        print(f"  Reduction:        {reduction:6.1f}%")

    except ImportError:
        print("⚠️  onnxruntime quantization not available")
        print("   Install with: pip install onnxruntime")
    except Exception as e:
        print(f"⚠️  Quantization failed: {e}")
        print("   Proceeding with FP32 model only")

    # Summary
    print("\n" + "="*70)
    print("Export Complete!")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  1. {output_path} (FP32)")
    if Path(quantized_path).exists():
        print(f"  2. {quantized_path} (INT8, recommended)")
    print(f"\nNext steps:")
    print(f"  1. Copy quantized model to Flutter project:")
    print(f"     cp {quantized_path} <flutter_project>/assets/models/")
    print(f"  2. Update pubspec.yaml:")
    print(f"     flutter:")
    print(f"       assets:")
    print(f"         - assets/models/dqn_mobile_quantized.onnx")
    print(f"  3. Add dependency:")
    print(f"     flutter pub add onnxruntime")
    print(f"  4. Use DQNAgent class from FLUTTER_ONNX_GUIDE.md")
    print()

    return True


if __name__ == "__main__":
    success = export_dqn()
    exit(0 if success else 1)
