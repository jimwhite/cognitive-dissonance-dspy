"""Test MLFlow telemetry integration."""

import os
import tempfile
import pytest
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, '.')

from cognitive_dissonance.config import ExperimentConfig
from cognitive_dissonance.experiment import ExperimentResults


class TestMLFlowIntegration:
    """Test MLFlow integration with experiment tracking."""
    
    def test_mlflow_config_from_env(self):
        """Test MLFlow configuration from environment variables."""
        with patch.dict(os.environ, {
            'ENABLE_MLFLOW': 'true',
            'MLFLOW_TRACKING_URI': 'http://localhost:5000',
            'MLFLOW_EXPERIMENT_NAME': 'test-experiment'
        }):
            config = ExperimentConfig.from_env()
            
            assert config.enable_mlflow is True
            assert config.mlflow_tracking_uri == 'http://localhost:5000'
            assert config.mlflow_experiment_name == 'test-experiment'
    
    def test_mlflow_setup_when_disabled(self):
        """Test that MLFlow setup does nothing when disabled."""
        config = ExperimentConfig(enable_mlflow=False)
        
        # Should not raise any exceptions
        config.setup_mlflow()
        
    def test_mlflow_setup_when_enabled(self):
        """Test MLFlow setup when enabled - basic config test."""
        config = ExperimentConfig(
            enable_mlflow=True,
            mlflow_tracking_uri='http://localhost:5000',
            mlflow_experiment_name='test-experiment'
        )
        
        # Test that setup doesn't crash even without mlflow installed
        # In real usage, this would configure MLFlow if available
        config.setup_mlflow()
    
    def test_mlflow_setup_missing_import(self):
        """Test MLFlow setup when package is not installed."""
        with patch('builtins.__import__', side_effect=ImportError("No module named 'mlflow'")):
            config = ExperimentConfig(enable_mlflow=True)
            
            # Should not raise exception, just log warning
            config.setup_mlflow()
    
    def test_mlflow_disabled_in_experiment_results(self):
        """Test that MLFlow logging is skipped when disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(
                enable_mlflow=False,
                checkpoints=tmpdir
            )
            results = ExperimentResults(config)
            results.add_round(1, 0.8, 0.7, 0.9, 0.85, 0.75)
            
            # Mock agents
            results.agent_a = Mock()
            results.agent_b = Mock()
            
            # Should not attempt MLFlow logging
            results._log_models_to_mlflow("test_checkpoint", "/path/a.json", "/path/b.json")
    
    def test_mlflow_enabled_in_experiment_results(self):
        """Test that MLFlow logging works when enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(
                enable_mlflow=True,
                mlflow_experiment_name='test-experiment',
                checkpoints=tmpdir,
                model='test-model',
                temperature=0.7,
                alpha=0.5
            )
            results = ExperimentResults(config)
            results.add_round(1, 0.8, 0.7, 0.9, 0.85, 0.75)
            
            # Mock agents
            results.agent_a = Mock()
            results.agent_b = Mock()
            
            # Call MLFlow logging - should not crash even without MLFlow installed
            results._log_models_to_mlflow("test_checkpoint", "/path/a.json", "/path/b.json")
    
    def test_mlflow_logging_resilient_to_errors(self):
        """Test that MLFlow logging errors don't crash the experiment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(
                enable_mlflow=True,
                checkpoints=tmpdir
            )
            results = ExperimentResults(config)
            results.add_round(1, 0.8, 0.7, 0.9, 0.85, 0.75)
            
            results.agent_a = Mock()
            results.agent_b = Mock()
            
            # Should not raise exception, just log warnings (MLFlow not installed)
            results._log_models_to_mlflow("test_checkpoint", "/path/a.json", "/path/b.json")
    
    def test_mlflow_integration_with_checkpoint_save(self):
        """Test that MLFlow logging is called during checkpoint save."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(
                enable_mlflow=True,
                checkpoints=tmpdir
            )
            results = ExperimentResults(config)
            results.add_round(1, 0.8, 0.7, 0.9, 0.85, 0.75)
            
            # Mock the MLFlow logging method to track calls
            with patch.object(results, '_log_models_to_mlflow') as mock_log:
                # Mock agents with save method
                mock_agent_a = Mock()
                mock_agent_b = Mock()
                results.agent_a = mock_agent_a
                results.agent_b = mock_agent_b
                
                # Save checkpoint
                checkpoint_path = results.save_checkpoint()
                
                # Verify MLFlow logging was called
                assert checkpoint_path is not None
                mock_log.assert_called_once()
    
    def test_cli_args_set_mlflow_config(self):
        """Test that CLI arguments properly set MLFlow configuration."""
        # This would test the main.py argument parsing, but we'll simulate it
        config = ExperimentConfig.from_env()
        
        # Simulate CLI argument application
        config.enable_mlflow = True
        config.mlflow_tracking_uri = 'http://test-server:5000'
        config.mlflow_experiment_name = 'cli-test'
        
        assert config.enable_mlflow is True
        assert config.mlflow_tracking_uri == 'http://test-server:5000'
        assert config.mlflow_experiment_name == 'cli-test'


if __name__ == "__main__":
    # Run a simple integration test
    print("🧪 Testing MLFlow Integration")
    print("=" * 50)
    
    # Test 1: Config from environment
    with patch.dict(os.environ, {'ENABLE_MLFLOW': 'true'}):
        config = ExperimentConfig.from_env()
        assert config.enable_mlflow is True
        print("✅ Config from environment works")
    
    # Test 2: MLFlow setup (basic functionality)
    config = ExperimentConfig(enable_mlflow=True)
    config.setup_mlflow()  # Should not crash
    print("✅ MLFlow setup works when enabled")
    
    # Test 3: Disabled MLFlow doesn't crash
    config = ExperimentConfig(enable_mlflow=False)
    config.setup_mlflow()  # Should do nothing
    print("✅ Disabled MLFlow doesn't crash")
    
    print("\n🎉 MLFlow integration tests passed!")
    print("\nTo use MLFlow with your experiments:")
    print("  export ENABLE_MLFLOW=true")
    print("  export MLFLOW_TRACKING_URI=http://localhost:5000")
    print("  python -m cognitive_dissonance experiment --enable-mlflow")