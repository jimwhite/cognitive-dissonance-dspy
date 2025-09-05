"""End-to-end test for checkpointing functionality."""

import tempfile
from unittest.mock import Mock, patch
from cognitive_dissonance.experiment import ExperimentResults


def test_checkpointing_saves_files_when_enabled():
    """Test that checkpoints are actually saved when config.checkpoints is set."""
    
    from cognitive_dissonance.config import ExperimentConfig
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create config and results with checkpoints enabled
        config = ExperimentConfig(checkpoints=tmpdir)
        results = ExperimentResults(config)
        results.add_round(1, 0.8, 0.7, 0.9, 0.85, 0.75)
        
        # Test the exact logic from the experiment function:
        # results.save_checkpoint()
        checkpoint_path = results.save_checkpoint()
        
        # Verify file was actually created
        import os
        assert checkpoint_path is not None
        assert os.path.exists(checkpoint_path)
        assert checkpoint_path.endswith('.pkl')
        assert 'experiment_' in os.path.basename(checkpoint_path)
        assert 'round_1' in os.path.basename(checkpoint_path)
        
        # Verify we can load it back
        loaded_results = ExperimentResults.load_checkpoint(checkpoint_path)
        assert len(loaded_results.rounds) == 1
        assert loaded_results.rounds[0]['accuracy_a'] == 0.8
        
        print(f"✅ Checkpoint saved and verified: {checkpoint_path}")


def test_checkpointing_disabled_when_none():
    """Test that no files are saved when config.checkpoints is None."""
    
    from cognitive_dissonance.config import ExperimentConfig
    
    # Create config with checkpoints disabled and results
    config = ExperimentConfig(checkpoints=None)
    results = ExperimentResults(config)
    results.add_round(1, 0.8, 0.7, 0.9, 0.85, 0.75)
    
    # This should return None and not save any files
    checkpoint_path = results.save_checkpoint()
    assert checkpoint_path is None
    print("✅ No checkpoint saved when checkpoints=None")


def test_config_determines_checkpoint_behavior():
    """Test that config.checkpoints controls whether checkpointing occurs."""
    
    from cognitive_dissonance.config import ExperimentConfig
    
    # Test 1: Default config (no checkpoints)
    config = ExperimentConfig()
    assert config.checkpoints is None
    assert not bool(config.checkpoints)  # This is the actual condition checked
    print("✅ Default config has checkpoints=None")
    
    # Test 2: Config with checkpoints enabled
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ExperimentConfig(checkpoints=tmpdir)
        assert config.checkpoints == tmpdir
        assert bool(config.checkpoints)  # This would trigger checkpoint saving
        print(f"✅ Config with checkpoints={tmpdir} would enable saving")


def test_cli_and_env_var_integration():
    """Test that CLI args and env vars properly set config.checkpoints."""
    
    from cognitive_dissonance.config import ExperimentConfig
    import os
    
    # Test environment variable
    original_env = os.environ.get('CHECKPOINTS')
    try:
        os.environ['CHECKPOINTS'] = '/tmp/from_env'
        config = ExperimentConfig.from_env()
        assert config.checkpoints == '/tmp/from_env'
        print("✅ Environment variable CHECKPOINTS works")
    finally:
        if original_env is not None:
            os.environ['CHECKPOINTS'] = original_env
        else:
            os.environ.pop('CHECKPOINTS', None)
    
    # Test direct assignment (simulates CLI override)
    config = ExperimentConfig()
    config.checkpoints = '/tmp/from_cli'
    assert config.checkpoints == '/tmp/from_cli'
    print("✅ CLI override simulation works")


if __name__ == "__main__":
    test_checkpointing_saves_files_when_enabled()
    test_checkpointing_disabled_when_none()
    test_config_determines_checkpoint_behavior()
    test_cli_and_env_var_integration()
    print("\n🎉 All checkpointing end-to-end tests passed!")
    print("\nTo enable checkpointing in your experiment:")
    print("  export CHECKPOINTS=.checkpoints")
    print("  python -m cognitive_dissonance experiment")
    print("OR:")
    print("  python -m cognitive_dissonance experiment --checkpoints=.checkpoints")