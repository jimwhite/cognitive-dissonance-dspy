#!/usr/bin/env python3
"""
Standalone test to verify checkpoint files are actually created.
Run this directly: python test_checkpoint_standalone.py
"""

import os
import sys
import tempfile

# Add current directory to path so we can import our modules
sys.path.insert(0, '.')

from cognitive_dissonance.config import ExperimentConfig
from cognitive_dissonance.experiment import ExperimentResults


def test_checkpoint_creation():
    """Test that actually creates checkpoint files you can see."""
    
    # Use a real directory so you can check it
    checkpoint_dir = "/tmp/test_checkpoints_real"
    
    print(f"Creating checkpoints in: {checkpoint_dir}")
    
    # Clean up any existing files first
    if os.path.exists(checkpoint_dir):
        import shutil
        shutil.rmtree(checkpoint_dir)
    
    # Create the directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Created directory: {os.path.exists(checkpoint_dir)}")
    
    # Test 1: Basic checkpoint creation
    print("\n=== Test 1: Basic checkpoint creation ===")
    config = ExperimentConfig(checkpoints=checkpoint_dir)
    print(f"Config checkpoints: {config.checkpoints}")
    
    results = ExperimentResults(config)
    print(f"Results config: {results.config is not None}")
    print(f"Results config checkpoints: {results.config.checkpoints if results.config else 'None'}")
    
    # Add some data
    results.add_round(1, 0.8, 0.7, 0.9, 0.85, 0.75)
    print(f"Added round, now have {len(results.rounds)} rounds")
    
    # Try to save
    print("Calling save_checkpoint()...")
    try:
        checkpoint_path = results.save_checkpoint()
        print(f"save_checkpoint() returned: {checkpoint_path}")
        
        if checkpoint_path:
            exists = os.path.exists(checkpoint_path)
            print(f"File exists: {exists}")
            if exists:
                size = os.path.getsize(checkpoint_path)
                print(f"File size: {size} bytes")
        
        # List directory contents
        files = os.listdir(checkpoint_dir)
        print(f"Files in directory: {files}")
        
        return len(files) > 0
        
    except Exception as e:
        print(f"Error during save_checkpoint(): {e}")
        import traceback
        traceback.print_exc()
        return False


def test_experiment_simulation():
    """Simulate the experiment loop that should create checkpoints."""
    
    checkpoint_dir = "/tmp/test_experiment_checkpoints"
    
    print(f"\n=== Test 2: Experiment simulation ===")
    print(f"Using directory: {checkpoint_dir}")
    
    # Clean up
    if os.path.exists(checkpoint_dir):
        import shutil
        shutil.rmtree(checkpoint_dir)
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Create config and results like the experiment does
    config = ExperimentConfig(checkpoints=checkpoint_dir)
    results = ExperimentResults(config)
    
    print("Simulating experiment rounds...")
    
    files_created = []
    
    # Simulate multiple rounds
    for round_num in [1, 2, 3]:
        print(f"\n--- Round {round_num} ---")
        
        # Add round data
        results.add_round(round_num, 0.8, 0.7, 0.9, 0.85, 0.75)
        
        # Simulate setting agents (like in experiment)
        results.agent_a = f"mock_agent_a_{round_num}"
        results.agent_b = f"mock_agent_b_{round_num}"
        
        print(f"Added round {round_num}, total rounds: {len(results.rounds)}")
        
        # This is the exact call from cognitive_dissonance_experiment()
        checkpoint_path = results.save_checkpoint()
        print(f"save_checkpoint() returned: {checkpoint_path}")
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            size = os.path.getsize(checkpoint_path)
            print(f"✅ Created checkpoint: {os.path.basename(checkpoint_path)} ({size} bytes)")
            files_created.append(os.path.basename(checkpoint_path))
        else:
            print(f"❌ No checkpoint created for round {round_num}")
    
    # Final check
    all_files = os.listdir(checkpoint_dir)
    pkl_files = [f for f in all_files if f.endswith('.pkl')]
    
    print(f"\nFinal directory contents: {all_files}")
    print(f"Checkpoint files (.pkl): {pkl_files}")
    print(f"Expected 3 files, got: {len(pkl_files)}")
    
    return len(pkl_files) == 3


def main():
    """Run all tests and report results."""
    
    print("🔍 Testing checkpoint file creation...")
    print("=" * 50)
    
    # Test 1
    test1_success = test_checkpoint_creation()
    print(f"\nTest 1 result: {'✅ PASS' if test1_success else '❌ FAIL'}")
    
    # Test 2  
    test2_success = test_experiment_simulation()
    print(f"Test 2 result: {'✅ PASS' if test2_success else '❌ FAIL'}")
    
    # Overall result
    if test1_success and test2_success:
        print("\n🎉 ALL TESTS PASSED - Checkpoints are being created!")
        print("\nCheck these directories for the files:")
        print("  /tmp/test_checkpoints_real")
        print("  /tmp/test_experiment_checkpoints")
    else:
        print("\n💥 TESTS FAILED - Checkpoints are NOT being created!")
        print("\nThis explains why your CLI command isn't creating checkpoint files.")
    
    return test1_success and test2_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)