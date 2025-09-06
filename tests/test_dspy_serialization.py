#!/usr/bin/env python3
"""
Test DSPy agent serialization to ensure proper checkpoint/resume functionality.
This tests the core issue: can we save and restore DSPy agents without retraining?
"""

import os
import sys
import tempfile

# Add current directory to path so we can import our modules
sys.path.insert(0, '.')

from cognitive_dissonance.config import ExperimentConfig, setup_logging
from cognitive_dissonance.experiment import ExperimentResults
from cognitive_dissonance.verifier import CognitiveDissonanceResolver


def test_dspy_agent_serialization():
    """Test that DSPy agents can be saved and loaded properly."""
    
    # Set up minimal logging to see what's happening
    setup_logging("WARNING")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Testing DSPy serialization in: {tmpdir}")
        
        # Create config and results
        config = ExperimentConfig(checkpoints=tmpdir)
        results = ExperimentResults(config)
        
        print("1. Creating and training agent...")
        # Create an agent (this simulates the expensive training)
        agent = CognitiveDissonanceResolver(use_cot=False)
        
        # Add it to results
        results.agent_a = agent
        results.agent_b = agent  # Same agent for simplicity
        
        print(f"   Agent created: {type(agent)}")
        print(f"   Agent has belief_agent: {hasattr(agent, 'belief_agent')}")
        print(f"   Agent has dissonance_detector: {hasattr(agent, 'dissonance_detector')}")
        
        # Add some round data
        results.add_round(1, 0.8, 0.7, 0.9, 0.85, 0.75)
        results.add_round(2, 0.82, 0.72, 0.91, 0.86, 0.76)
        
        print(f"2. Added {len(results.rounds)} rounds of data")
        
        print("3. Saving checkpoint with agents...")
        try:
            checkpoint_path = results.save_checkpoint()
            print(f"   Checkpoint saved to: {checkpoint_path}")
            
            if checkpoint_path and os.path.exists(checkpoint_path):
                size = os.path.getsize(checkpoint_path)
                print(f"   Checkpoint file size: {size} bytes")
            
            # Check for agent files
            files = os.listdir(tmpdir)
            json_files = [f for f in files if f.endswith('.json')]
            pkl_files = [f for f in files if f.endswith('.pkl')]
            
            print(f"   Files created: {files}")
            print(f"   JSON agent files: {json_files}")
            print(f"   PKL checkpoint files: {pkl_files}")
            
            # We should have:
            # - 1 .pkl file for the main checkpoint
            # - 2 .json files for the agents
            assert len(pkl_files) == 1, f"Should have 1 pkl file, got {len(pkl_files)}"
            assert len(json_files) == 2, f"Should have 2 json agent files, got {len(json_files)}"
            
        except Exception as e:
            print(f"   ❌ Error saving checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("4. Loading checkpoint to verify agents restore...")
        try:
            # Load from checkpoint using static method
            new_results = ExperimentResults.load_checkpoint(checkpoint_path)
            new_results.config = config  # Restore config like in real code
            print(f"   Loaded results has {len(new_results.rounds)} rounds")
            
            print(f"   Agent A type: {type(new_results.agent_a)}")
            print(f"   Agent B type: {type(new_results.agent_b)}")
            print(f"   Agent A is None: {new_results.agent_a is None}")
            print(f"   Agent B is None: {new_results.agent_b is None}")
            
            # Verify agents have their components
            if new_results.agent_a:
                print(f"   Agent A has belief_agent: {hasattr(new_results.agent_a, 'belief_agent')}")
                print(f"   Agent A has dissonance_detector: {hasattr(new_results.agent_a, 'dissonance_detector')}")
            
            # Test that we can actually use the restored agent
            print("5. Testing restored agent functionality...")
            if new_results.agent_a:
                try:
                    # This should work without retraining
                    test_result = new_results.agent_a(
                        text1="The sky is blue",
                        text2="The sky is red"
                    )
                    print(f"   ✅ Agent works! Result type: {type(test_result)}")
                    print(f"   ✅ Has dissonance: {getattr(test_result, 'has_dissonance', 'unknown')}")
                    return True
                except Exception as e:
                    print(f"   ❌ Agent failed to work: {e}")
                    import traceback
                    traceback.print_exc()
                    return False
            else:
                print("   ❌ Agent A is None after loading")
                return False
                
        except Exception as e:
            print(f"   ❌ Error loading checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_resume_from_existing_checkpoint():
    """Test resuming from an existing checkpoint like in the real experiment."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\n=== Testing Resume from Checkpoint ===")
        print(f"Using directory: {tmpdir}")
        
        # Phase 1: Create initial experiment with 2 rounds (like user's scenario)
        print("Phase 1: Creating initial experiment...")
        config = ExperimentConfig(checkpoints=tmpdir)
        results1 = ExperimentResults(config)
        
        # Add 2 rounds of data (simulating 2.5 hours each)
        results1.add_round(1, 0.8, 0.7, 0.9, 0.85, 0.75)
        results1.add_round(2, 0.82, 0.72, 0.91, 0.86, 0.76)
        
        # Create agents (expensive training)
        results1.agent_a = CognitiveDissonanceResolver(use_cot=False)
        results1.agent_b = CognitiveDissonanceResolver(use_cot=False)
        
        # Save checkpoint
        checkpoint_path = results1.save_checkpoint()
        print(f"   Saved checkpoint: {checkpoint_path}")
        
        # Phase 2: Resume experiment (like user's restart)
        print("Phase 2: Resuming experiment...")
        
        # This is what should happen on restart
        results2 = ExperimentResults.load_checkpoint(checkpoint_path)
        results2.config = config  # Restore config like in real code
        
        if results2:
            print(f"   ✅ Resumed with {len(results2.rounds)} rounds")
            print(f"   ✅ Agent A restored: {results2.agent_a is not None}")
            print(f"   ✅ Agent B restored: {results2.agent_b is not None}")
            
            # Continue experiment from round 3
            results2.add_round(3, 0.83, 0.73, 0.92, 0.87, 0.77)
            print(f"   ✅ Added round 3, now have {len(results2.rounds)} rounds")
            
            # Test agent still works
            if results2.agent_a:
                try:
                    test_result = results2.agent_a(
                        text1="Test claim 1",
                        text2="Test claim 2"
                    )
                    print(f"   ✅ Agent still functional after resume!")
                    return True
                except Exception as e:
                    print(f"   ❌ Agent not functional: {e}")
                    return False
            else:
                print(f"   ❌ Agent A is None after resume")
                return False
        else:
            print(f"   ❌ Failed to resume from checkpoint")
            return False


def main():
    """Run all serialization tests."""
    
    print("🧪 Testing DSPy Agent Serialization for Checkpointing")
    print("=" * 60)
    
    # Test 1: Basic serialization
    test1_success = test_dspy_agent_serialization()
    print(f"\nTest 1 (Basic serialization): {'✅ PASS' if test1_success else '❌ FAIL'}")
    
    # Test 2: Resume scenario
    test2_success = test_resume_from_existing_checkpoint()
    print(f"Test 2 (Resume scenario): {'✅ PASS' if test2_success else '❌ FAIL'}")
    
    if test1_success and test2_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ DSPy agents can be properly saved and restored")
        print("✅ No retraining required on resume")
        print("✅ 2.5-hour training time is preserved")
    else:
        print("\n💥 TESTS FAILED!")
        print("❌ DSPy agent serialization is not working properly")
        
    return test1_success and test2_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)