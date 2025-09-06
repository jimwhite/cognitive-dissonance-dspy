"""Debug checkpoint saving step by step."""

import os
import tempfile
from cognitive_dissonance.config import ExperimentConfig
from cognitive_dissonance.experiment import ExperimentResults


def test_step_by_step_checkpoint_creation():
    """Debug each step of checkpoint creation."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Using tmpdir: {tmpdir}")
        
        # Step 1: Create config with checkpoints
        config = ExperimentConfig(checkpoints=tmpdir)
        print(f"Config checkpoints: {config.checkpoints}")
        assert config.checkpoints == tmpdir
        
        # Step 2: Create results with config
        results = ExperimentResults(config)
        print(f"Results config: {results.config}")
        print(f"Results config checkpoints: {results.config.checkpoints if results.config else 'No config'}")
        
        # Step 3: Add a round
        results.add_round(1, 0.8, 0.7, 0.9, 0.85, 0.75)
        print(f"Rounds: {len(results.rounds)}")
        
        # Step 4: Call save_checkpoint
        print("Calling save_checkpoint()...")
        checkpoint_path = results.save_checkpoint()
        print(f"Returned checkpoint path: {checkpoint_path}")
        
        # Step 5: Check if file exists
        if checkpoint_path:
            exists = os.path.exists(checkpoint_path)
            print(f"File exists: {exists}")
            if exists:
                size = os.path.getsize(checkpoint_path)
                print(f"File size: {size} bytes")
            
            # List all files in directory
            files = os.listdir(tmpdir)
            print(f"Files in {tmpdir}: {files}")
        else:
            print("❌ save_checkpoint returned None")
            
        assert checkpoint_path is not None, "save_checkpoint should return a path"
        assert os.path.exists(checkpoint_path), f"Checkpoint file should exist: {checkpoint_path}"


def test_experiment_loop_checkpoint_call():
    """Test the actual checkpoint call that happens in experiment loop."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Testing experiment loop logic in: {tmpdir}")
        
        # Simulate the exact experiment setup
        config = ExperimentConfig(checkpoints=tmpdir)
        results = ExperimentResults(config)
        
        # Simulate experiment loop
        for round_num in [1, 2]:
            # Add round results
            results.add_round(round_num, 0.8, 0.7, 0.9, 0.85, 0.75)
            
            # Update agents (simulate experiment code)
            results.agent_a = f"mock_agent_a_round_{round_num}"
            results.agent_b = f"mock_agent_b_round_{round_num}"
            
            # This is the exact call from experiment.py
            print(f"Round {round_num}: calling save_checkpoint()")
            checkpoint_path = results.save_checkpoint()
            print(f"Round {round_num}: returned {checkpoint_path}")
            
            if checkpoint_path:
                assert os.path.exists(checkpoint_path), f"Round {round_num} checkpoint should exist"
                print(f"✅ Round {round_num} checkpoint created")
            else:
                print(f"❌ Round {round_num} checkpoint failed")
        
        # Check final state
        files = [f for f in os.listdir(tmpdir) if f.endswith('.pkl')]
        print(f"Final checkpoint files: {files}")
        assert len(files) == 2, f"Should have 2 checkpoint files, got {len(files)}: {files}"


def test_experiment_config_vs_results_config():
    """Test if there's a mismatch between experiment config and results config."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create config like CLI does
        config = ExperimentConfig.from_env()
        config.checkpoints = tmpdir  # Simulate CLI override
        
        print(f"Original config checkpoints: {config.checkpoints}")
        
        # Create results like experiment does
        results = ExperimentResults(config)
        
        print(f"Results config checkpoints: {results.config.checkpoints}")
        print(f"Configs are same object: {config is results.config}")
        
        # Test checkpoint saving
        results.add_round(1, 0.8, 0.7, 0.9, 0.85, 0.75)
        checkpoint_path = results.save_checkpoint()
        
        assert checkpoint_path is not None, "Checkpoint should be created"
        assert os.path.exists(checkpoint_path), "Checkpoint file should exist"
        print(f"✅ Config flow test passed: {checkpoint_path}")


if __name__ == "__main__":
    print("=== Step by step checkpoint test ===")
    test_step_by_step_checkpoint_creation()
    
    print("\n=== Experiment loop test ===") 
    test_experiment_loop_checkpoint_call()
    
    print("\n=== Config flow test ===")
    test_experiment_config_vs_results_config()
    
    print("\n🎉 All debug tests passed!")