"""Test the actual CLI checkpointing behavior that the user reported."""

import os
import tempfile
import sys
from unittest.mock import Mock, patch
import dspy

# Import the main function directly 
from cognitive_dissonance.main import main
from cognitive_dissonance.config import ExperimentConfig


def test_cli_experiment_creates_checkpoint_files():
    """Test that CLI experiment --checkpoints=.checkpoints actually creates files."""
    
    # with tempfile.TemporaryDirectory() as tmpdir:
    with open('/tmp/cp') as tmpdir:
        # checkpoint_dir = os.path.join(tmpdir, ".checkpoints")
        checkpoint_dir = os.path.join('/tmp/cp', ".checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Mock sys.argv to simulate the exact CLI command user ran
        original_argv = sys.argv
        try:
            sys.argv = [
                'cognitive_dissonance', 
                'experiment', 
                f'--checkpoints={checkpoint_dir}',
                '--rounds=2'
            ]
            
            # Mock all the expensive parts like in the integration test
            with patch('cognitive_dissonance.experiment.get_dev_labeled') as mock_dev, \
                 patch('cognitive_dissonance.experiment.get_train_unlabeled') as mock_train, \
                 patch('cognitive_dissonance.experiment.MIPROv2') as mock_mipro, \
                 patch('cognitive_dissonance.experiment.evaluate', return_value=0.8), \
                 patch('cognitive_dissonance.experiment.agreement_rate', return_value=0.9), \
                 patch('cognitive_dissonance.experiment.analyze_errors', return_value={}), \
                 patch('dspy.configure'), \
                 patch('dspy.LM'), \
                 patch('dspy.configure_cache'):
                
                # Setup mock data
                mock_dev.return_value = [
                    dspy.Example(
                        text1="t1", text2="t2",
                        has_dissonance="yes",
                        reconciled="r1"
                    ).with_inputs("text1", "text2"),
                    dspy.Example(
                        text1="t1b", text2="t2b", 
                        has_dissonance="no",
                        reconciled="r1b"
                    ).with_inputs("text1", "text2")
                ]
                mock_train.return_value = [
                    dspy.Example(text1="t3", text2="t4").with_inputs("text1", "text2"),
                    dspy.Example(text1="t5", text2="t6").with_inputs("text1", "text2")
                ]
                
                # Setup mock optimizer
                mock_optimizer = Mock()
                mock_optimizer.compile = Mock(return_value=Mock())
                mock_mipro.return_value = mock_optimizer
                
                # Run main() - this simulates exactly what happens when user runs CLI
                main()
                
                # Verify checkpoint files were created
                assert os.path.exists(checkpoint_dir), f"Checkpoint directory was not created: {checkpoint_dir}"
                
                checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pkl')]
                
                # Should have 2 checkpoint files (one after each round)
                assert len(checkpoint_files) == 2, f"Expected 2 checkpoint files, got {len(checkpoint_files)}: {checkpoint_files}"
                
                # Verify file naming pattern
                for filename in checkpoint_files:
                    assert filename.startswith('experiment_')
                    assert '.pkl' in filename
                    assert 'round_' in filename
                
                print(f"✅ CLI created checkpoint files: {checkpoint_files}")
                
                # Verify we can load the checkpoints
                from cognitive_dissonance.experiment import ExperimentResults
                for checkpoint_file in checkpoint_files:
                    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
                    loaded_results = ExperimentResults.load_checkpoint(checkpoint_path)
                    assert len(loaded_results.rounds) > 0
                    print(f"✅ Successfully loaded checkpoint: {checkpoint_file}")
        
        finally:
            sys.argv = original_argv


def test_cli_experiment_no_checkpoints_creates_no_files():
    """Test that CLI experiment without --checkpoints creates no files."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock sys.argv to simulate CLI command WITHOUT checkpoints
        original_argv = sys.argv
        try:
            sys.argv = [
                'cognitive_dissonance', 
                'experiment',
                '--rounds=1'
                # Note: NO --checkpoints argument
            ]
            
            # Mock all the expensive parts
            with patch('cognitive_dissonance.experiment.get_dev_labeled') as mock_dev, \
                 patch('cognitive_dissonance.experiment.get_train_unlabeled') as mock_train, \
                 patch('cognitive_dissonance.experiment.MIPROv2') as mock_mipro, \
                 patch('cognitive_dissonance.experiment.evaluate', return_value=0.8), \
                 patch('cognitive_dissonance.experiment.agreement_rate', return_value=0.9), \
                 patch('cognitive_dissonance.experiment.analyze_errors', return_value={}), \
                 patch('dspy.configure'), \
                 patch('dspy.LM'), \
                 patch('dspy.configure_cache'):
                
                # Setup mock data  
                mock_dev.return_value = [
                    dspy.Example(
                        text1="t1", text2="t2",
                        has_dissonance="yes",
                        reconciled="r1"
                    ).with_inputs("text1", "text2")
                ]
                mock_train.return_value = [
                    dspy.Example(text1="t3", text2="t4").with_inputs("text1", "text2"),
                    dspy.Example(text1="t5", text2="t6").with_inputs("text1", "text2")
                ]
                
                # Setup mock optimizer
                mock_optimizer = Mock()
                mock_optimizer.compile = Mock(return_value=Mock())
                mock_mipro.return_value = mock_optimizer
                
                # Run main()
                main()
                
                # Verify NO checkpoint files were created anywhere
                # Check if .checkpoints directory was even created
                default_checkpoint_dir = os.path.join(os.getcwd(), '.checkpoints')
                
                if os.path.exists(default_checkpoint_dir):
                    checkpoint_files = [f for f in os.listdir(default_checkpoint_dir) if f.endswith('.pkl')]
                    assert len(checkpoint_files) == 0, f"Unexpected checkpoint files found: {checkpoint_files}"
                
                print("✅ No checkpoint files created when --checkpoints not specified")
        
        finally:
            sys.argv = original_argv


def test_checkpoint_config_flow_in_cli():
    """Test that CLI argument properly flows through to config and results."""
    
    from cognitive_dissonance.config import ExperimentConfig
    from cognitive_dissonance.experiment import ExperimentResults
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = os.path.join(tmpdir, "test_checkpoints")
        
        # Test 1: Create config as CLI would
        config = ExperimentConfig.from_env()
        # Simulate CLI override  
        config.checkpoints = checkpoint_dir
        
        # Test 2: Create results as experiment would
        results = ExperimentResults(config)
        results.add_round(1, 0.8, 0.7, 0.9, 0.85, 0.75)
        
        # Test 3: Verify save_checkpoint works
        checkpoint_path = results.save_checkpoint()
        
        assert checkpoint_path is not None, "save_checkpoint should return path when checkpoints enabled"
        assert os.path.exists(checkpoint_path), f"Checkpoint file should exist: {checkpoint_path}"
        assert checkpoint_dir in checkpoint_path, f"Checkpoint should be in specified directory: {checkpoint_path}"
        
        print(f"✅ Config flow test: {checkpoint_path}")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])