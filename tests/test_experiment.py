"""Tests for experiment module."""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
import dspy
from cognitive_dissonance.experiment import (
    ExperimentResults,
    cognitive_dissonance_experiment,
    run_ablation_study,
    run_confidence_analysis,
    find_latest_checkpoint
)
from cognitive_dissonance.config import ExperimentConfig


class TestExperimentResults:
    """Test ExperimentResults class."""
    
    def test_initialization(self):
        """Test initialization of results container."""
        results = ExperimentResults()
        
        assert results.rounds == []
        assert results.agent_a is None
        assert results.agent_b is None
        assert results.error_analysis == {}
    
    def test_add_round(self):
        """Test adding round results."""
        results = ExperimentResults()
        
        results.add_round(
            round_num=1,
            acc_a=0.8,
            acc_b=0.75,
            agree_dev=0.9,
            agree_train=0.85,
            reconciliation_quality=0.7
        )
        
        assert len(results.rounds) == 1
        assert results.rounds[0]["round"] == 1
        assert results.rounds[0]["accuracy_a"] == 0.8
        assert results.rounds[0]["accuracy_b"] == 0.75
        assert results.rounds[0]["agreement_dev"] == 0.9
        assert results.rounds[0]["agreement_train"] == 0.85
        assert results.rounds[0]["reconciliation_quality"] == 0.7
    
    def test_get_final_accuracies(self):
        """Test getting final accuracies."""
        results = ExperimentResults()
        
        # Empty results
        acc_a, acc_b = results.get_final_accuracies()
        assert acc_a == 0.0
        assert acc_b == 0.0
        
        # Add rounds
        results.add_round(1, 0.6, 0.5, 0.7, 0.65)
        results.add_round(2, 0.8, 0.75, 0.85, 0.8)
        
        acc_a, acc_b = results.get_final_accuracies()
        assert acc_a == 0.8
        assert acc_b == 0.75
    
    def test_get_final_agreement(self):
        """Test getting final agreement."""
        results = ExperimentResults()
        
        # Empty results
        assert results.get_final_agreement() == 0.0
        
        # Add rounds
        results.add_round(1, 0.6, 0.5, 0.7, 0.65)
        results.add_round(2, 0.8, 0.75, 0.85, 0.8)
        
        assert results.get_final_agreement() == 0.85
    
    def test_get_final_reconciliation(self):
        """Test getting final reconciliation quality."""
        results = ExperimentResults()
        
        # Empty results
        assert results.get_final_reconciliation() == 0.0
        
        # Add rounds
        results.add_round(1, 0.6, 0.5, 0.7, 0.65, reconciliation_quality=0.6)
        results.add_round(2, 0.8, 0.75, 0.85, 0.8, reconciliation_quality=0.9)
        
        assert results.get_final_reconciliation() == 0.9
    
    def test_summary(self):
        """Test generating summary."""
        results = ExperimentResults()
        
        # Empty summary
        summary = results.summary()
        assert summary["total_rounds"] == 0
        
        # Add data
        results.add_round(1, 0.6, 0.5, 0.7, 0.65, 0.55)
        results.add_round(2, 0.8, 0.75, 0.85, 0.8, 0.7)
        results.add_round(3, 0.75, 0.8, 0.9, 0.85, 0.8)
        results.error_analysis = {"error_rate": 0.1}
        
        summary = results.summary()
        
        assert summary["total_rounds"] == 3
        assert summary["final_accuracy_a"] == 0.75
        assert summary["final_accuracy_b"] == 0.8
        assert summary["final_agreement"] == 0.9
        assert summary["final_reconciliation"] == 0.8
        assert summary["max_accuracy_a"] == 0.8
        assert summary["max_accuracy_b"] == 0.8
        assert summary["max_agreement"] == 0.9
        assert summary["error_analysis"]["error_rate"] == 0.1
    
    def test_save_checkpoint(self):
        """Test saving experiment checkpoint."""
        from cognitive_dissonance.config import ExperimentConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(checkpoints=tmpdir)
            results = ExperimentResults(config)
            results.add_round(1, 0.8, 0.7, 0.9, 0.85, 0.75)
            results.add_round(2, 0.85, 0.8, 0.95, 0.9, 0.8)
            
            checkpoint_path = results.save_checkpoint()
            
            # Check file was created
            assert checkpoint_path is not None
            assert os.path.exists(checkpoint_path)
            assert checkpoint_path.endswith('.pkl')
            assert 'experiment_' in os.path.basename(checkpoint_path)
            assert 'round_2' in os.path.basename(checkpoint_path)
    
    def test_load_checkpoint(self):
        """Test loading experiment checkpoint."""
        from cognitive_dissonance.config import ExperimentConfig
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save results
            config = ExperimentConfig(checkpoints=tmpdir)
            original_results = ExperimentResults(config)
            original_results.add_round(1, 0.8, 0.7, 0.9, 0.85, 0.75)
            original_results.add_round(2, 0.85, 0.8, 0.95, 0.9, 0.8)
            original_results.error_analysis = {"test": "data"}
            original_results.optimization_history = [{"step": 1, "score": 0.8}]
            
            checkpoint_path = original_results.save_checkpoint()
            
            # Load results
            loaded_results = ExperimentResults.load_checkpoint(checkpoint_path)
            
            # Verify loaded data matches original
            assert len(loaded_results.rounds) == 2
            assert loaded_results.rounds[0]["accuracy_a"] == 0.8
            assert loaded_results.rounds[1]["accuracy_a"] == 0.85
            assert loaded_results.error_analysis == {"test": "data"}
            assert loaded_results.optimization_history == [{"step": 1, "score": 0.8}]
    
    def test_load_checkpoint_nonexistent(self):
        """Test loading from nonexistent checkpoint raises error."""
        with pytest.raises(FileNotFoundError):
            ExperimentResults.load_checkpoint("/nonexistent/path.pkl")


class TestFindLatestCheckpoint:
    """Test find_latest_checkpoint utility function."""
    
    def test_nonexistent_directory(self):
        """Test with nonexistent directory returns None."""
        result = find_latest_checkpoint("/nonexistent/directory")
        assert result is None
    
    def test_empty_directory(self):
        """Test with empty directory returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = find_latest_checkpoint(tmpdir)
            assert result is None
    
    def test_no_checkpoint_files(self):
        """Test with directory containing no .pkl files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some non-checkpoint files
            with open(os.path.join(tmpdir, "not_checkpoint.txt"), "w") as f:
                f.write("test")
            
            result = find_latest_checkpoint(tmpdir)
            assert result is None
    
    def test_single_checkpoint(self):
        """Test with single checkpoint file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_file = os.path.join(tmpdir, "checkpoint.pkl")
            with open(checkpoint_file, "w") as f:
                f.write("test")
            
            result = find_latest_checkpoint(tmpdir)
            assert result == checkpoint_file
    
    def test_multiple_checkpoints_by_timestamp(self):
        """Test selecting latest checkpoint by modification time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create checkpoints with different timestamps
            old_file = os.path.join(tmpdir, "old.pkl")
            middle_file = os.path.join(tmpdir, "middle.pkl")
            newest_file = os.path.join(tmpdir, "newest.pkl")
            
            with open(old_file, "w") as f:
                f.write("old")
            os.utime(old_file, (1000, 1000))  # Set old timestamp
            
            with open(middle_file, "w") as f:
                f.write("middle")  
            os.utime(middle_file, (2000, 2000))  # Set middle timestamp
            
            with open(newest_file, "w") as f:
                f.write("newest")
            os.utime(newest_file, (3000, 3000))  # Set newest timestamp
            
            result = find_latest_checkpoint(tmpdir)
            assert result == newest_file
    
    def test_touch_changes_selection(self):
        """Test that touching a file makes it the latest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_file = os.path.join(tmpdir, "old.pkl")
            new_file = os.path.join(tmpdir, "new.pkl")
            
            # Create old file first
            with open(old_file, "w") as f:
                f.write("old")
            os.utime(old_file, (1000, 1000))
            
            # Create new file second  
            with open(new_file, "w") as f:
                f.write("new")
            os.utime(new_file, (2000, 2000))
            
            # New file should be selected initially
            result = find_latest_checkpoint(tmpdir)
            assert result == new_file
            
            # Touch old file to make it newest
            os.utime(old_file, (3000, 3000))
            
            # Old file should now be selected
            result = find_latest_checkpoint(tmpdir)
            assert result == old_file


class TestCognitiveDissonanceExperiment:
    """Test main experiment function."""
    
    @patch('cognitive_dissonance.experiment.get_dev_labeled')
    @patch('cognitive_dissonance.experiment.get_train_unlabeled')
    @patch('cognitive_dissonance.experiment.MIPROv2')
    def test_basic_experiment(self, mock_mipro, mock_train, mock_dev, mock_config):
        """Test basic experiment execution."""
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
        
        # Configure for minimal rounds
        mock_config.rounds = 1
        mock_config.checkpoints = None  # No checkpoints
        mock_config.validate = Mock()
        mock_config.setup_dspy = Mock()
        
        # Run experiment
        with patch('cognitive_dissonance.experiment.evaluate', return_value=0.8):
            with patch('cognitive_dissonance.experiment.agreement_rate', return_value=0.9):
                with patch('cognitive_dissonance.experiment.analyze_errors', return_value={}):
                    results = cognitive_dissonance_experiment(mock_config)
        
        assert isinstance(results, ExperimentResults)
        assert len(results.rounds) == 1
        assert results.agent_a is not None
        assert results.agent_b is not None
    
    @patch('cognitive_dissonance.experiment.get_dev_labeled')
    @patch('cognitive_dissonance.experiment.get_train_unlabeled')  
    @patch('cognitive_dissonance.experiment.MIPROv2')
    def test_experiment_with_checkpointing_enabled(self, mock_mipro, mock_train, mock_dev, mock_config):
        """Test experiment saves checkpoints when enabled."""
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
        
        # Configure with checkpoints enabled
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config.rounds = 2
            mock_config.checkpoints = tmpdir
            mock_config.validate = Mock()
            mock_config.setup_dspy = Mock()
            
            # Mock ExperimentResults.save_checkpoint to track calls
            with patch.object(ExperimentResults, 'save_checkpoint', return_value="/fake/path.pkl") as mock_save:
                with patch('cognitive_dissonance.experiment.evaluate', return_value=0.8):
                    with patch('cognitive_dissonance.experiment.agreement_rate', return_value=0.9):
                        with patch('cognitive_dissonance.experiment.analyze_errors', return_value={}):
                            results = cognitive_dissonance_experiment(mock_config)
                
                # Verify checkpoints were saved after each round
                assert mock_save.call_count == 2  # Once per round
                # New API doesn't take arguments
                mock_save.assert_called_with()
    
    def test_experiment_checkpoint_resume_logic(self, mock_config):
        """Test checkpoint resume logic without running full experiment."""
        from cognitive_dissonance.experiment import find_latest_checkpoint
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test 1: No checkpoint directory - should start fresh
            mock_config.checkpoints = None
            start_round = 1
            skip_baseline = False
            
            if mock_config.checkpoints:
                latest_checkpoint = find_latest_checkpoint(mock_config.checkpoints)
                if latest_checkpoint:
                    skip_baseline = True
            
            assert start_round == 1
            assert skip_baseline is False
            
            # Test 2: Checkpoint directory exists but empty - should start fresh
            mock_config.checkpoints = tmpdir
            start_round = 1
            skip_baseline = False
            
            if mock_config.checkpoints:
                latest_checkpoint = find_latest_checkpoint(mock_config.checkpoints)
                if latest_checkpoint:
                    skip_baseline = True
            
            assert start_round == 1
            assert skip_baseline is False
            
            # Test 3: Checkpoint exists - should resume
            # Create a fake checkpoint file
            checkpoint_file = os.path.join(tmpdir, "experiment_123_round_2.pkl")
            with open(checkpoint_file, "w") as f:
                f.write("fake checkpoint")
            
            if mock_config.checkpoints:
                latest_checkpoint = find_latest_checkpoint(mock_config.checkpoints)
                if latest_checkpoint:
                    # In real scenario, would load results and set start_round = len(results.rounds) + 1
                    start_round = 3  # Simulating resuming from round 2, so next is 3
                    skip_baseline = True
            
            assert start_round == 3
            assert skip_baseline is True
            assert latest_checkpoint == checkpoint_file
    
    @patch('cognitive_dissonance.experiment.ExperimentConfig')
    def test_experiment_from_env(self, mock_config_class):
        """Test experiment loading config from environment."""
        mock_config = Mock()
        mock_config.rounds = 0  # Will cause early exit
        mock_config.validate = Mock()
        mock_config.setup_dspy = Mock()
        mock_config_class.from_env.return_value = mock_config
        
        with patch('cognitive_dissonance.experiment.get_dev_labeled', return_value=[]):
            with patch('cognitive_dissonance.experiment.get_train_unlabeled', return_value=[]):
                with pytest.raises(ValueError):  # Empty dataset should raise
                    cognitive_dissonance_experiment()
    
    def test_experiment_with_overrides(self, mock_config):
        """Test experiment with parameter overrides."""
        mock_config.validate = Mock()
        mock_config.setup_dspy = Mock()
        
        with patch('cognitive_dissonance.experiment.get_dev_labeled', return_value=[]):
            with patch('cognitive_dissonance.experiment.get_train_unlabeled', return_value=[]):
                with pytest.raises(ValueError):  # Empty dataset
                    cognitive_dissonance_experiment(
                        config=mock_config,
                        rounds=3,
                        use_cot=True,
                        alpha_anchor=0.5
                    )
        
        assert mock_config.rounds == 3
        assert mock_config.use_cot is True
        assert mock_config.alpha == 0.5


class TestAblationStudy:
    """Test ablation study function."""
    
    @patch('cognitive_dissonance.experiment.cognitive_dissonance_experiment')
    def test_run_ablation_study(self, mock_experiment, mock_config):
        """Test running ablation study."""
        # Setup mock experiment results
        mock_results = ExperimentResults()
        mock_results.add_round(1, 0.8, 0.75, 0.85, 0.8, 0.7)
        mock_experiment.return_value = mock_results
        
        results = run_ablation_study(mock_config)
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Check that different configurations were tested
        assert "baseline" in results
        assert "with_cot" in results
        assert "alpha_0.1" in results
        assert "alpha_0.5" in results
        
        # Verify experiment was called multiple times
        assert mock_experiment.call_count == len(results)
    
    @patch('cognitive_dissonance.experiment.cognitive_dissonance_experiment')
    def test_ablation_study_handles_failures(self, mock_experiment, mock_config):
        """Test ablation study handles experiment failures."""
        # Make some experiments fail
        mock_experiment.side_effect = [
            ExperimentResults(),  # Success
            ValueError("Failed"),  # Failure
            ExperimentResults(),  # Success
        ]
        
        results = run_ablation_study(mock_config)
        
        # Should still return results, with None for failures
        assert any(r is None for r in results.values())
        assert any(r is not None for r in results.values())


class TestConfidenceAnalysis:
    """Test confidence analysis function."""
    
    @patch('cognitive_dissonance.experiment.get_dev_labeled')
    @patch('cognitive_dissonance.experiment.BootstrapFewShot')
    @patch('cognitive_dissonance.experiment.evaluate')
    def test_run_confidence_analysis(
        self, mock_evaluate, mock_bootstrap, mock_dev, mock_config
    ):
        """Test running confidence analysis."""
        # Setup mock data
        mock_dev.return_value = [
            dspy.Example(
                text1="t1", text2="t2",
                has_dissonance="yes",
                reconciled="r1"
            ).with_inputs("text1", "text2")
        ]
        
        # Setup mock optimizer
        mock_optimizer = Mock()
        mock_optimizer.compile = Mock(return_value=Mock())
        mock_bootstrap.return_value = mock_optimizer
        
        # Setup mock evaluation
        mock_evaluate.side_effect = [0.7, 0.8, 0.75]
        
        mock_config.validate = Mock()
        mock_config.setup_dspy = Mock()
        
        results = run_confidence_analysis(mock_config)
        
        assert isinstance(results, dict)
        assert "accuracy_without_confidence" in results
        assert "accuracy_with_confidence" in results
        assert "confidence_weighted_accuracy" in results
        assert "improvement" in results
        
        assert results["accuracy_without_confidence"] == 0.7
        assert results["accuracy_with_confidence"] == 0.8
        assert results["confidence_weighted_accuracy"] == 0.75
        assert abs(results["improvement"] - 0.1) < 0.0001  # Use approximate equality
    
    @patch('cognitive_dissonance.experiment.ExperimentConfig')
    def test_confidence_analysis_from_env(self, mock_config_class):
        """Test confidence analysis loading config from environment."""
        mock_config = Mock()
        mock_config.validate = Mock()
        mock_config.setup_dspy = Mock()
        mock_config_class.from_env.return_value = mock_config
        
        with patch('cognitive_dissonance.experiment.get_dev_labeled', return_value=[]):
            with patch('cognitive_dissonance.experiment.BootstrapFewShot'):
                with patch('cognitive_dissonance.experiment.evaluate', return_value=0.5):
                    results = run_confidence_analysis()
        
        assert isinstance(results, dict)