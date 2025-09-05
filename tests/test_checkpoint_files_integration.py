"""Test that checkpoints actually save to files during experiment execution."""

import os
import tempfile
import dspy
from unittest.mock import Mock, patch
from cognitive_dissonance.experiment import cognitive_dissonance_experiment
from cognitive_dissonance.config import ExperimentConfig


def test_experiment_saves_checkpoint_files():
    """Test that experiment actually saves checkpoint files when enabled."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create config with checkpoints enabled
        config = ExperimentConfig(
            checkpoints=tmpdir,
            rounds=2,
            use_cot=False,
            alpha=0.0
        )
        
        # Mock all the expensive parts but let the checkpoint logic run
        with patch('cognitive_dissonance.experiment.get_dev_labeled') as mock_dev, \
             patch('cognitive_dissonance.experiment.get_train_unlabeled') as mock_train, \
             patch('cognitive_dissonance.experiment.MIPROv2') as mock_mipro, \
             patch('cognitive_dissonance.experiment.evaluate', return_value=0.8), \
             patch('cognitive_dissonance.experiment.agreement_rate', return_value=0.9), \
             patch('cognitive_dissonance.experiment.analyze_errors', return_value={}):
            
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
            
            # Don't mock config methods - let them run normally or patch the internals
            with patch('dspy.configure'), \
                 patch('dspy.LM'), \
                 patch('dspy.configure_cache'):
                
                # Run the experiment - this should save checkpoint files
                results = cognitive_dissonance_experiment(config)
                
                # Verify checkpoint files were created
                checkpoint_files = [f for f in os.listdir(tmpdir) if f.endswith('.pkl')]
                
                # Should have 2 checkpoint files (one after each round)
                assert len(checkpoint_files) == 2, f"Expected 2 checkpoint files, got {len(checkpoint_files)}: {checkpoint_files}"
                
                # Verify file naming pattern
                for filename in checkpoint_files:
                    assert filename.startswith('experiment_')
                    assert '.pkl' in filename
                    assert 'round_' in filename
                
                print(f"✅ Checkpoint files created: {checkpoint_files}")
                
                # Verify we can load the checkpoints
                from cognitive_dissonance.experiment import ExperimentResults
                for checkpoint_file in checkpoint_files:
                    checkpoint_path = os.path.join(tmpdir, checkpoint_file)
                    loaded_results = ExperimentResults.load_checkpoint(checkpoint_path)
                    assert len(loaded_results.rounds) > 0
                    print(f"✅ Successfully loaded checkpoint: {checkpoint_file}")


def test_experiment_no_files_when_checkpoints_disabled():
    """Test that no checkpoint files are saved when checkpoints are disabled."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create config with checkpoints DISABLED
        config = ExperimentConfig(
            checkpoints=None,  # Disabled!
            rounds=1,
            use_cot=False, 
            alpha=0.0
        )
        
        # Mock all the expensive parts
        with patch('cognitive_dissonance.experiment.get_dev_labeled') as mock_dev, \
             patch('cognitive_dissonance.experiment.get_train_unlabeled') as mock_train, \
             patch('cognitive_dissonance.experiment.MIPROv2') as mock_mipro, \
             patch('cognitive_dissonance.experiment.evaluate', return_value=0.8), \
             patch('cognitive_dissonance.experiment.agreement_rate', return_value=0.9), \
             patch('cognitive_dissonance.experiment.analyze_errors', return_value={}):
            
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
            
            config.validate = Mock()
            config.setup_dspy = Mock()
            
            # Run the experiment
            results = cognitive_dissonance_experiment(config)
            
            # Verify NO checkpoint files were created
            all_files = os.listdir(tmpdir) if os.path.exists(tmpdir) else []
            checkpoint_files = [f for f in all_files if f.endswith('.pkl')]
            
            assert len(checkpoint_files) == 0, f"Expected no checkpoint files, but found: {checkpoint_files}"
            print("✅ No checkpoint files created when checkpoints disabled")


def test_save_checkpoint_method_api():
    """Test the new save_checkpoint() API without arguments."""
    
    from cognitive_dissonance.experiment import ExperimentResults
    from cognitive_dissonance.config import ExperimentConfig
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test 1: With checkpoints enabled
        config = ExperimentConfig(checkpoints=tmpdir)
        results = ExperimentResults(config)
        results.add_round(1, 0.8, 0.7, 0.9, 0.85, 0.75)
        
        checkpoint_path = results.save_checkpoint()
        assert checkpoint_path is not None
        assert os.path.exists(checkpoint_path)
        print(f"✅ save_checkpoint() with config works: {checkpoint_path}")
        
        # Test 2: With checkpoints disabled
        config_disabled = ExperimentConfig(checkpoints=None)
        results_disabled = ExperimentResults(config_disabled)
        results_disabled.add_round(1, 0.8, 0.7, 0.9, 0.85, 0.75)
        
        checkpoint_path = results_disabled.save_checkpoint()
        assert checkpoint_path is None
        print("✅ save_checkpoint() returns None when checkpoints disabled")
        
        # Test 3: No config at all
        results_no_config = ExperimentResults()
        results_no_config.add_round(1, 0.8, 0.7, 0.9, 0.85, 0.75)
        
        checkpoint_path = results_no_config.save_checkpoint()
        assert checkpoint_path is None
        print("✅ save_checkpoint() returns None when no config")


if __name__ == "__main__":
    test_experiment_saves_checkpoint_files()
    test_experiment_no_files_when_checkpoints_disabled() 
    test_save_checkpoint_method_api()
    print("\n🎉 All checkpoint file integration tests passed!")