"""
Model serialization framework for learned agents.

All learning agents must be able to save/load their learned state to JSON.
This enables:
- Portability across platforms
- Version control of trained models
- Easy integration into game implementations
"""

import json
import os
from datetime import datetime
from typing import Any, Dict
from pathlib import Path


class ModelSerializer:
    """
    Standard serialization interface for learned models.

    All learning agents should implement:
    - to_dict() -> Dict: Convert learned state to dictionary
    - from_dict(data: Dict) -> None: Restore learned state from dictionary
    """

    @staticmethod
    def save_model(
        agent_name: str,
        agent_type: str,
        version: str,
        model_data: Dict[str, Any],
        hyperparameters: Dict[str, Any],
        performance: Dict[str, float] | None = None,
        metadata: Dict[str, Any] | None = None,
        filepath: str | None = None
    ) -> str:
        """
        Save a learned model to JSON.

        Args:
            agent_name: Human-readable agent name (e.g., "AssociativeMemory-v1")
            agent_type: Type identifier (e.g., "associative_memory", "linear_value")
            version: Model version (e.g., "1.0")
            model_data: The learned state (weights, memory, etc.)
            hyperparameters: Training hyperparameters
            performance: Optional performance metrics (win rates, etc.)
            metadata: Optional additional metadata
            filepath: Optional custom save path (defaults to models/<agent_name>_<timestamp>.json)

        Returns:
            Path to saved file
        """
        # Build standard model schema
        model_dict = {
            "agent_name": agent_name,
            "agent_type": agent_type,
            "version": version,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model_data": model_data,
            "hyperparameters": hyperparameters,
        }

        if performance is not None:
            model_dict["performance"] = performance

        if metadata is not None:
            model_dict["metadata"] = metadata

        # Determine save path
        if filepath is None:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{agent_name}_{timestamp_str}.json"
            filepath = os.path.join("models", filename)

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save to JSON with pretty formatting
        with open(filepath, 'w') as f:
            json.dump(model_dict, f, indent=2, sort_keys=True)

        print(f"Model saved to: {filepath}")
        return filepath

    @staticmethod
    def load_model(filepath: str) -> Dict[str, Any]:
        """
        Load a learned model from JSON.

        Args:
            filepath: Path to saved model file

        Returns:
            Dictionary containing model data, hyperparameters, etc.

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, 'r') as f:
            model_dict = json.load(f)

        # Validate required fields
        required_fields = ["agent_name", "agent_type", "version", "model_data", "hyperparameters"]
        missing_fields = [f for f in required_fields if f not in model_dict]
        if missing_fields:
            raise ValueError(f"Invalid model file. Missing fields: {missing_fields}")

        print(f"Model loaded from: {filepath}")
        print(f"  Agent: {model_dict['agent_name']} ({model_dict['agent_type']})")
        print(f"  Version: {model_dict['version']}")
        if "timestamp" in model_dict:
            print(f"  Saved: {model_dict['timestamp']}")

        return model_dict

    @staticmethod
    def list_models(models_dir: str = "models", agent_type: str | None = None) -> list[str]:
        """
        List all saved models in a directory.

        Args:
            models_dir: Directory to search for models
            agent_type: Optional filter by agent type

        Returns:
            List of model filepaths
        """
        if not os.path.exists(models_dir):
            return []

        model_files = []
        for filename in os.listdir(models_dir):
            if not filename.endswith('.json'):
                continue

            filepath = os.path.join(models_dir, filename)

            # If filtering by type, check the file
            if agent_type is not None:
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    if data.get("agent_type") != agent_type:
                        continue
                except:
                    continue

            model_files.append(filepath)

        return sorted(model_files)


class SerializableAgent:
    """
    Base class for agents with serialization support.

    Subclasses should implement:
    - to_dict() -> Dict: Export learned state
    - from_dict(data: Dict) -> None: Import learned state
    """

    def to_dict(self) -> Dict[str, Any]:
        """
        Export agent's learned state to dictionary.
        Must be implemented by subclass.
        """
        raise NotImplementedError("Subclass must implement to_dict()")

    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        Import agent's learned state from dictionary.
        Must be implemented by subclass.
        """
        raise NotImplementedError("Subclass must implement from_dict()")

    def save(
        self,
        agent_name: str,
        agent_type: str,
        version: str,
        hyperparameters: Dict[str, Any],
        performance: Dict[str, float] | None = None,
        filepath: str | None = None
    ) -> str:
        """
        Save this agent to JSON.

        Args:
            agent_name: Human-readable agent name
            agent_type: Type identifier
            version: Model version
            hyperparameters: Training hyperparameters
            performance: Optional performance metrics
            filepath: Optional custom save path

        Returns:
            Path to saved file
        """
        model_data = self.to_dict()
        return ModelSerializer.save_model(
            agent_name=agent_name,
            agent_type=agent_type,
            version=version,
            model_data=model_data,
            hyperparameters=hyperparameters,
            performance=performance,
            filepath=filepath
        )

    def load(self, filepath: str) -> None:
        """
        Load this agent from JSON.

        Args:
            filepath: Path to saved model file
        """
        model_dict = ModelSerializer.load_model(filepath)
        self.from_dict(model_dict["model_data"])
