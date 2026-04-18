"""Tests for state feature extraction."""

import sys
sys.path.insert(0, 'src')

import numpy as np
from utala.state import GameState, Player, Phase
from utala.learning.features import StateFeatureExtractor, get_feature_extractor


def test_feature_extractor_shape():
    """Test that feature extraction produces correct shape."""
    extractor = StateFeatureExtractor()
    state = GameState()

    features = extractor.extract(state, Player.ONE)

    assert isinstance(features, np.ndarray)
    assert features.shape == (53,)
    assert extractor.feature_dim == 53


def test_feature_extractor_range():
    """Test that all features are normalized to [0, 1]."""
    extractor = StateFeatureExtractor()
    state = GameState()

    features = extractor.extract(state, Player.ONE)

    assert np.all(features >= 0.0)
    assert np.all(features <= 1.0)


def test_feature_names_length():
    """Test that feature names match feature dimension."""
    extractor = StateFeatureExtractor()
    names = extractor.feature_names()

    assert len(names) == extractor.feature_dim
    assert len(names) == 53


def test_singleton_pattern():
    """Test that get_feature_extractor returns singleton."""
    extractor1 = get_feature_extractor()
    extractor2 = get_feature_extractor()

    assert extractor1 is extractor2


def test_initial_state_features():
    """Test feature extraction from initial game state."""
    extractor = StateFeatureExtractor()
    state = GameState()

    features = extractor.extract(state, Player.ONE)

    # Initial state: all squares empty
    # Grid occupancy: first 27 features
    # Each square = [empty=1, P1=0, P2=0]
    for i in range(0, 27, 3):
        assert features[i] == 1.0  # empty
        assert features[i+1] == 0.0  # P1
        assert features[i+2] == 0.0  # P2

    # Resource counts: both players have full resources
    assert features[27] == 1.0  # my rocketmen (9/9)
    assert features[28] == 1.0  # my weapons (4/4)
    assert features[29] == 0.0  # my kaos (0/9 - not initialized yet)
    assert features[30] == 1.0  # opp rocketmen (9/9)
    assert features[31] == 1.0  # opp weapons (4/4)
    assert features[32] == 0.0  # opp kaos (0/9 - not initialized yet)

    # Material balance: equal resources
    assert features[33] == 0.5  # rocketmen advantage (0 diff)
    assert features[34] == 0.5  # weapons advantage (0 diff)
    assert features[35] == 0.5  # kaos advantage (0 diff)

    # Phase: placement (indices 42-43)
    assert features[42] == 1.0  # placement
    assert features[43] == 0.0  # dogfights


if __name__ == '__main__':
    test_feature_extractor_shape()
    test_feature_extractor_range()
    test_feature_names_length()
    test_singleton_pattern()
    test_initial_state_features()
    print("All feature extraction tests passed!")
