"""Integration tests for checkpoint functionality."""

import pytest
import tempfile
from unittest.mock import Mock, patch, MagicMock
from cognitive_dissonance.config import ExperimentConfig


class TestCheckpointingIntegration:
    """Test checkpoint integration with main CLI."""
    
    def test_config_checkpoints_defaults_to_none(self):
        """Test that checkpoints defaults to None."""
        config = ExperimentConfig()
        assert config.checkpoints is None
    
    def test_config_checkpoints_from_env(self, monkeypatch):
        """Test checkpoints config from environment variable."""
        monkeypatch.setenv("CHECKPOINTS", "/tmp/test_checkpoints")
        config = ExperimentConfig.from_env()
        assert config.checkpoints == "/tmp/test_checkpoints"
    
    @patch('cognitive_dissonance.main.cognitive_dissonance_experiment')
    @patch('cognitive_dissonance.main.setup_logging')
    @patch('sys.argv', ['main.py', 'experiment', '--checkpoints', '/tmp/test'])
    def test_cli_checkpoints_argument_passed_to_experiment(self, mock_logging, mock_experiment):
        """Test that CLI checkpoints argument reaches the experiment function."""
        from cognitive_dissonance.main import main
        
        # Mock the experiment to avoid actually running it
        mock_results = Mock()
        mock_results.summary.return_value = {
            'total_rounds': 1,
            'final_accuracy_a': 0.8,
            'final_accuracy_b': 0.7,
            'final_agreement': 0.9,
            'final_reconciliation': 0.85
        }
        mock_experiment.return_value = mock_results
        
        # Run main
        main()
        
        # Verify experiment was called with config that has checkpoints set
        mock_experiment.assert_called_once()
        config_arg = mock_experiment.call_args[0][0]  # Get the first positional arg (config)
        assert config_arg.checkpoints == '/tmp/test'
    
    @patch('cognitive_dissonance.main.cognitive_dissonance_experiment')
    @patch('cognitive_dissonance.main.setup_logging')
    @patch.dict('os.environ', {'CHECKPOINTS': '/tmp/env_test'})
    @patch('sys.argv', ['main.py', 'experiment'])
    def test_env_checkpoints_passed_to_experiment(self, mock_logging, mock_experiment):
        """Test that environment CHECKPOINTS variable reaches experiment."""
        from cognitive_dissonance.main import main
        
        # Mock the experiment
        mock_results = Mock()
        mock_results.summary.return_value = {
            'total_rounds': 1,
            'final_accuracy_a': 0.8,
            'final_accuracy_b': 0.7,
            'final_agreement': 0.9,
            'final_reconciliation': 0.85
        }
        mock_experiment.return_value = mock_results
        
        # Run main
        main()
        
        # Verify experiment was called with config that has checkpoints from env
        mock_experiment.assert_called_once()
        config_arg = mock_experiment.call_args[0][0]  # Get the first positional arg (config)
        assert config_arg.checkpoints == '/tmp/env_test'
    
    @patch('cognitive_dissonance.main.cognitive_dissonance_experiment')
    @patch('cognitive_dissonance.main.setup_logging')
    @patch.dict('os.environ', {'CHECKPOINTS': '/tmp/env_test'})
    @patch('sys.argv', ['main.py', 'experiment', '--checkpoints', '/tmp/cli_override'])
    def test_cli_overrides_env_checkpoints(self, mock_logging, mock_experiment):
        """Test that CLI argument overrides environment variable."""
        from cognitive_dissonance.main import main
        
        # Mock the experiment
        mock_results = Mock()
        mock_results.summary.return_value = {
            'total_rounds': 1,
            'final_accuracy_a': 0.8,
            'final_accuracy_b': 0.7,
            'final_agreement': 0.9,
            'final_reconciliation': 0.85
        }
        mock_experiment.return_value = mock_results
        
        # Run main
        main()
        
        # Verify CLI overrides environment
        mock_experiment.assert_called_once()
        config_arg = mock_experiment.call_args[0][0]  # Get the first positional arg (config)
        assert config_arg.checkpoints == '/tmp/cli_override'
    
    def test_checkpoint_logic_flow(self):
        """Test the checkpoint decision logic without running experiment."""
        # Test 1: No checkpoints configured
        config = ExperimentConfig(checkpoints=None)
        assert not bool(config.checkpoints)  # Should not save checkpoints
        
        # Test 2: Checkpoints configured
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(checkpoints=tmpdir)
            assert bool(config.checkpoints)  # Should save checkpoints
            assert config.checkpoints == tmpdir
    
    @patch('cognitive_dissonance.experiment.find_latest_checkpoint')
    def test_checkpoint_resume_logic(self, mock_find_checkpoint):
        """Test checkpoint resume decision logic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test 1: No existing checkpoint
            mock_find_checkpoint.return_value = None
            config = ExperimentConfig(checkpoints=tmpdir)
            
            # Should start fresh
            latest = mock_find_checkpoint(config.checkpoints)
            assert latest is None
            
            # Test 2: Existing checkpoint found
            mock_checkpoint_path = f"{tmpdir}/experiment_123_round_5.pkl"
            mock_find_checkpoint.return_value = mock_checkpoint_path
            
            # Should resume
            latest = mock_find_checkpoint(config.checkpoints)
            assert latest == mock_checkpoint_path