"""Tests for model serialization."""

import sys
sys.path.insert(0, 'src')

import os
import json
import tempfile
import shutil
from utala.learning.serialization import ModelSerializer, SerializableAgent


def test_save_and_load_model():
    """Test saving and loading a model."""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, "test_model.json")

    try:
        # Save a model
        model_data = {
            "weights": [0.1, 0.2, 0.3],
            "memory": ["state1", "state2"]
        }
        hyperparameters = {
            "learning_rate": 0.01,
            "k": 20
        }
        performance = {
            "vs_random": 0.85,
            "vs_heuristic": 0.60
        }

        filepath = ModelSerializer.save_model(
            agent_name="TestAgent",
            agent_type="test_type",
            version="1.0",
            model_data=model_data,
            hyperparameters=hyperparameters,
            performance=performance,
            filepath=temp_file
        )

        assert os.path.exists(filepath)

        # Load the model
        loaded = ModelSerializer.load_model(filepath)

        assert loaded["agent_name"] == "TestAgent"
        assert loaded["agent_type"] == "test_type"
        assert loaded["version"] == "1.0"
        assert loaded["model_data"] == model_data
        assert loaded["hyperparameters"] == hyperparameters
        assert loaded["performance"] == performance
        assert "timestamp" in loaded

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_serializable_agent_interface():
    """Test SerializableAgent base class."""

    class TestAgent(SerializableAgent):
        def __init__(self):
            self.weights = [1.0, 2.0, 3.0]

        def to_dict(self):
            return {"weights": self.weights}

        def from_dict(self, data):
            self.weights = data["weights"]

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Create and save an agent
        agent = TestAgent()
        filepath = agent.save(
            agent_name="TestAgent",
            agent_type="test",
            version="1.0",
            hyperparameters={"lr": 0.1},
            filepath=os.path.join(temp_dir, "agent.json")
        )

        # Load into a new agent
        agent2 = TestAgent()
        agent2.weights = []  # Clear weights
        agent2.load(filepath)

        assert agent2.weights == [1.0, 2.0, 3.0]

    finally:
        shutil.rmtree(temp_dir)


def test_list_models():
    """Test listing models in a directory."""
    temp_dir = tempfile.mkdtemp()

    try:
        # Create several test models
        for i in range(3):
            ModelSerializer.save_model(
                agent_name=f"Agent{i}",
                agent_type="type_a" if i < 2 else "type_b",
                version="1.0",
                model_data={"value": i},
                hyperparameters={},
                filepath=os.path.join(temp_dir, f"model{i}.json")
            )

        # List all models
        all_models = ModelSerializer.list_models(temp_dir)
        assert len(all_models) == 3

        # List filtered models
        type_a_models = ModelSerializer.list_models(temp_dir, agent_type="type_a")
        assert len(type_a_models) == 2

        type_b_models = ModelSerializer.list_models(temp_dir, agent_type="type_b")
        assert len(type_b_models) == 1

    finally:
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    test_save_and_load_model()
    test_serializable_agent_interface()
    test_list_models()
    print("All serialization tests passed!")
