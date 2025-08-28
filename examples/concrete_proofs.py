#!/usr/bin/env python3
"""
Example: Concrete mathematical proofs that actually succeed in Coq.

This demonstrates working formal verification with concrete, provable theorems
rather than abstract function specifications that require implementation details.
"""

import time
import logging
from typing import List

from formal_verification import (
    FormalVerificationConflictDetector, 
    Claim, 
    PropertyType
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_provable_claims() -> List[Claim]:
    """Create mathematical claims that can be formally proven."""
    return [
        # Basic arithmetic - these should work
        Claim(
            agent_id="alice",
            claim_text="3 + 5 = 8",
            property_type=PropertyType.CORRECTNESS,
            confidence=0.99,
            timestamp=time.time()
        ),
        
        # Conflicting arithmetic
        Claim(
            agent_id="bob", 
            claim_text="3 + 5 = 9",
            property_type=PropertyType.CORRECTNESS,
            confidence=0.85,
            timestamp=time.time()
        ),
        
        # Factorial claims
        Claim(
            agent_id="charlie",
            claim_text="factorial 4 = 24",
            property_type=PropertyType.CORRECTNESS,
            confidence=0.95,
            timestamp=time.time()
        ),
        
        Claim(
            agent_id="dave",
            claim_text="factorial 4 = 20",
            property_type=PropertyType.CORRECTNESS,
            confidence=0.70,
            timestamp=time.time()
        ),
        
        # More complex arithmetic
        Claim(
            agent_id="eve",
            claim_text="7 + 13 = 20",
            property_type=PropertyType.CORRECTNESS,
            confidence=0.92,
            timestamp=time.time()
        ),
        
        Claim(
            agent_id="frank",
            claim_text="12 + 8 = 20", 
            property_type=PropertyType.CORRECTNESS,
            confidence=0.88,
            timestamp=time.time()
        )
    ]


def test_concrete_mathematical_proofs():
    """Test concrete mathematical proofs that should succeed."""
    print("🧮 Testing Concrete Mathematical Formal Proofs...\n")
    
    detector = FormalVerificationConflictDetector(timeout_seconds=10)
    claims = create_provable_claims()
    
    try:
        results = detector.analyze_claims(claims, code="/* mathematical claims */")
        
        print("🔍 CONCRETE MATHEMATICAL PROOF VERIFICATION")
        print("=" * 55)
        print(f"Total Claims: {results['summary']['total_claims']}")
        print(f"Conflicts Detected: {results['summary']['conflicts_detected']}")
        print()
        
        print("📊 MATHEMATICAL PROOF RESULTS:")
        success_count = 0
        for result in results['proof_results']:
            status = "✅ MATHEMATICALLY PROVEN" if result.proven else "❌ PROOF FAILED"
            agent = result.spec.claim.agent_id
            claim = result.spec.claim.claim_text
            confidence = result.spec.claim.confidence
            
            if result.proven:
                success_count += 1
            
            print(f"{status}")
            print(f"  Agent: {agent} | Confidence: {confidence:.0%}")
            print(f"  Claim: '{claim}'")
            print(f"  Verification time: {result.proof_time_ms:.1f}ms")
            
            if not result.proven and result.error_message:
                # Extract key error information
                error_lines = result.error_message.split('\n')
                key_error = None
                for line in error_lines:
                    if 'Error:' in line or 'Unable to unify' in line:
                        key_error = line.strip()
                        break
                if key_error:
                    print(f"  Mathematical error: {key_error}")
                else:
                    print(f"  Error: Proof verification failed")
            
            print()
        
        if results['conflicts']:
            print("⚔️  MATHEMATICAL CONFLICTS DETECTED:")
            for i, (spec1, spec2) in enumerate(results['conflicts']):
                print(f"  {i+1}. '{spec1.claim.claim_text}' vs '{spec2.claim.claim_text}'")
            print()
        
        print("🏆 AGENT MATHEMATICAL ACCURACY RANKINGS:")
        for agent, accuracy in results['resolution']['agent_rankings'].items():
            print(f"  {agent}: {accuracy:.1%} mathematical accuracy")
        
        print()
        print("📋 CONCRETE PROOF VERIFICATION SUMMARY:")
        summary = results['summary']
        print(f"  • {summary['mathematically_proven']}/{len(results['proof_results'])} claims formally proven")
        print(f"  • {summary['mathematically_disproven']} claims mathematically disproven") 
        print(f"  • {summary['conflicts_detected']} mathematical conflicts resolved")
        print(f"  • Average proof time: {summary['average_proof_time_ms']:.1f}ms")
        print(f"  • Mathematical ground truth established: {summary['has_ground_truth']}")
        print(f"  • Success rate: {success_count}/{len(results['proof_results'])} = {success_count/len(results['proof_results']):.1%}")
        
        return results, success_count
        
    except Exception as e:
        logger.error(f"Concrete proof verification failed: {e}")
        print(f"❌ Concrete verification failed: {e}")
        return None, 0


def main():
    """Main concrete proof verification demo."""
    print("Starting Concrete Mathematical Proof Verification Demo...\n")
    
    results, success_count = test_concrete_mathematical_proofs()
    
    if results:
        print(f"\n🎯 CONCRETE THEOREM PROVING DEMONSTRATION:")
        print("=" * 50)
        print(f"Concrete Mathematical Claims: {len(results['proof_results'])}")
        print(f"Successfully Proven: {success_count}")
        print(f"Proof Success Rate: {success_count/len(results['proof_results']):.1%}")
        
        if success_count > 0:
            print("\n✅ Demonstrated working formal verification:")
            print("  • Mathematical theorem proving using Coq")
            print("  • Automatic translation of claims to formal specs")
            print("  • Conflict detection between contradictory claims") 
            print("  • Agent ranking based on mathematical correctness")
            print("  • Ground truth establishment through formal proof")
        else:
            print("\n⚠️  All proofs failed - investigating Coq specification issues")
    
    print(f"\n{'✅' if success_count > 0 else '⚠️'} Concrete mathematical proof verification complete!")
    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    exit(main())