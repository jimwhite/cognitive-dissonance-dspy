#!/usr/bin/env python3
"""
Example: Formal verification of mathematical claims with cognitive dissonance detection.

This example demonstrates the formal verification approach by analyzing
conflicting mathematical claims from different agents and using Coq theorem
proving to establish ground truth.
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


def create_mathematical_claims() -> List[Claim]:
    """Create a set of conflicting mathematical claims for testing."""
    return [
        Claim(
            agent_id="alice",
            claim_text="2 + 2 = 4",
            property_type=PropertyType.CORRECTNESS,
            confidence=0.95,
            timestamp=time.time()
        ),
        Claim(
            agent_id="bob", 
            claim_text="2 + 2 = 5",
            property_type=PropertyType.CORRECTNESS,
            confidence=0.80,
            timestamp=time.time()
        ),
        Claim(
            agent_id="charlie",
            claim_text="factorial 3 = 6", 
            property_type=PropertyType.CORRECTNESS,
            confidence=0.90,
            timestamp=time.time()
        ),
        Claim(
            agent_id="dave",
            claim_text="factorial 3 = 8",
            property_type=PropertyType.CORRECTNESS,
            confidence=0.70,
            timestamp=time.time()
        )
    ]


def print_analysis_results(results: dict):
    """Print formatted analysis results."""
    print("🔍 FORMAL VERIFICATION COGNITIVE DISSONANCE ANALYSIS")
    print("=" * 60)
    print(f"Domain: Mathematical Properties")
    print(f"Total Claims: {results['summary']['total_claims']}")
    print(f"Conflicts Detected: {results['summary']['conflicts_detected']}")
    print()
    
    print("📊 FORMAL PROOF RESULTS:")
    for result in results['proof_results']:
        status = "✅ MATHEMATICALLY PROVEN" if result.proven else "❌ MATHEMATICALLY DISPROVEN"
        agent = result.spec.claim.agent_id
        claim = result.spec.claim.claim_text
        confidence = result.spec.claim.confidence
        
        print(f"{status}")
        print(f"  Agent: {agent} | Confidence: {confidence:.0%}")
        print(f"  Claim: '{claim}'")
        print(f"  Verification time: {result.proof_time_ms:.1f}ms")
        
        if not result.proven and result.error_message:
            # Show relevant part of error message
            error_lines = result.error_message.split('\n')
            relevant_error = next((line for line in error_lines if 'Error:' in line), "Unknown error")
            print(f"  Mathematical error: {relevant_error[:80]}...")
        
        print()
    
    if results['conflicts']:
        print("🔍 CONFLICTS DETECTED:")
        for i, (spec1, spec2) in enumerate(results['conflicts']):
            print(f"  {i+1}. '{spec1.claim.claim_text}' vs '{spec2.claim.claim_text}'")
        print()
    
    print("🏆 AGENT MATHEMATICAL ACCURACY RANKINGS:")
    for agent, accuracy in results['resolution']['agent_rankings'].items():
        print(f"  {agent}: {accuracy:.1%} mathematical accuracy")
    
    print()
    print("📋 RESOLUTION SUMMARY:")
    summary = results['summary']
    print(f"  • {summary['mathematically_proven']} claims mathematically proven correct")
    print(f"  • {summary['mathematically_disproven']} claims mathematically disproven") 
    print(f"  • {summary['conflicts_detected']} conflicts detected and resolved")
    print(f"  • Average proof time: {summary['average_proof_time_ms']:.1f}ms")
    print(f"  • Ground truth established: {summary['has_ground_truth']}")


def main():
    """Main demonstration function."""
    print("Starting mathematical claims formal verification demo...\n")
    
    # Create detector
    detector = FormalVerificationConflictDetector(timeout_seconds=10)
    
    # Create conflicting mathematical claims
    claims = create_mathematical_claims()
    
    # Analyze claims using formal verification
    try:
        results = detector.analyze_claims(claims)
        print_analysis_results(results)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"❌ Analysis failed: {e}")
        return 1
    
    print("\n✅ Formal verification cognitive dissonance analysis complete!")
    return 0


if __name__ == "__main__":
    exit(main())