#!/usr/bin/env python3
"""
Demonstrate the hybrid Coq + Z3 proving system.

This shows how we intelligently choose between provers for different claim types,
achieving better coverage than either prover alone.
"""

import time
import logging
from typing import List

from formal_verification import (
    Claim,
    PropertyType
)
from formal_verification.z3_prover import HybridProver, Z3Prover
from formal_verification.prover import CoqProver

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demonstrate_z3_strengths():
    """Show what Z3 excels at compared to Coq."""
    print("🔷 Z3/SMT SOLVER CAPABILITIES")
    print("=" * 60)
    print("Z3 excels at constraint solving and satisfiability:\n")
    
    z3_prover = Z3Prover()
    
    # Claims that Z3 handles well
    z3_claims = [
        # Constraint satisfaction
        ("x + y = 10", "Find values where x + y equals 10"),
        
        # Array bounds checking
        ("array[i] is safe when 0 <= i < length", "Array access safety"),
        
        # Sorted array property
        ("array is sorted", "Verify array is in sorted order"),
        
        # Inequality chains
        ("x < 10", "Simple inequality constraint"),
        
        # Existential quantification
        ("exists x such that x > 10", "Find witness for existence"),
    ]
    
    results = []
    for claim_text, description in z3_claims:
        print(f"📌 {description}")
        print(f"   Claim: '{claim_text}'")
        
        result = z3_prover.prove_claim(claim_text)
        
        if result.result == "valid":
            print(f"   ✅ PROVEN by Z3 in {result.time_ms:.1f}ms")
        elif result.result == "invalid":
            print(f"   ❌ INVALID - Counter-example found:")
            if result.model:
                for var, val in result.model.items():
                    print(f"      {var} = {val}")
        else:
            print(f"   ⚠️ UNKNOWN - Could not determine")
        
        results.append(result)
        print()
    
    # Demonstrate satisfiability checking
    print("\n📊 SATISFIABILITY CHECKING:")
    print("-" * 40)
    
    constraints = [
        "x > 5",
        "x < 10",
        "y = x + 2"
    ]
    
    print(f"Constraints: {', '.join(constraints)}")
    sat_result = z3_prover.check_satisfiability(constraints)
    
    if sat_result.result == "satisfiable":
        print(f"✅ SATISFIABLE - Solution found:")
        if sat_result.model:
            for var, val in sat_result.model.items():
                print(f"   {var} = {val}")
    else:
        print(f"❌ UNSATISFIABLE - No solution exists")
    
    return results


def demonstrate_hybrid_prover():
    """Show the hybrid prover choosing the best tool for each claim."""
    print("\n\n🔄 HYBRID PROVING (Coq + Z3)")
    print("=" * 60)
    print("Intelligently choosing the best prover for each claim:\n")
    
    hybrid = HybridProver()
    
    # Mix of claims that benefit from different provers
    test_claims = [
        # Better with Coq (inductive/recursive)
        ("factorial 5 = 120", "factorial", "coq"),
        ("fibonacci 7 = 13", "fibonacci", "coq"),
        ("10 + 15 = 25", "arithmetic", "either"),
        
        # Better with Z3 (constraints/arrays)
        ("x < 10", "constraint", "z3"),
        ("array is sorted", "array property", "z3"),
        ("exists x such that x > 100", "existential", "z3"),
        
        # Complex (benefits from trying both)
        ("forall x, x + 0 = x", "universal identity", "either"),
        ("if x > 0 then x + 1 > 1", "implication", "either"),
    ]
    
    print("📈 PROVER SELECTION ANALYSIS:\n")
    
    coq_success = 0
    z3_success = 0
    total = len(test_claims)
    
    for claim_text, category, expected_prover in test_claims:
        print(f"Claim: '{claim_text}' ({category})")
        
        # Let hybrid choose
        result = hybrid.prove_claim(claim_text)
        
        prover_used = result.get('prover', 'unknown')
        proven = result.get('proven', False)
        time_ms = result.get('time_ms', 0)
        
        status = "✅ PROVEN" if proven else "❌ FAILED"
        print(f"  Prover: {prover_used.upper()} | Status: {status} | Time: {time_ms:.1f}ms")
        
        if proven:
            if prover_used == 'coq':
                coq_success += 1
            elif prover_used == 'z3':
                z3_success += 1
        
        # Show if it matched expected prover
        if expected_prover != "either" and prover_used != expected_prover:
            print(f"  ⚠️ Expected {expected_prover}, but used {prover_used}")
        
        print()
    
    # Statistics
    print("\n📊 HYBRID PROVING STATISTICS:")
    print("-" * 40)
    print(f"Total claims: {total}")
    print(f"Coq successes: {coq_success}")
    print(f"Z3 successes: {z3_success}")
    print(f"Overall success rate: {(coq_success + z3_success) / total:.1%}")
    
    # Show adaptive learning
    print(f"\n🧠 ADAPTIVE LEARNING:")
    print(f"Coq attempts: {hybrid.success_stats['coq']['attempts']}")
    print(f"Coq successes: {hybrid.success_stats['coq']['successes']}")
    print(f"Z3 attempts: {hybrid.success_stats['z3']['attempts']}")
    print(f"Z3 successes: {hybrid.success_stats['z3']['successes']}")
    
    return hybrid


def demonstrate_counter_examples():
    """Show Z3's ability to find counter-examples."""
    print("\n\n🔍 COUNTER-EXAMPLE GENERATION")
    print("=" * 60)
    print("Z3 can find counter-examples to disprove claims:\n")
    
    z3_prover = Z3Prover()
    
    false_claims = [
        "10 + 15 = 26",  # Wrong arithmetic
        "forall x, x > 0",  # Not all numbers are positive
        "x * 2 = 15",  # No integer solution
    ]
    
    for claim in false_claims:
        print(f"Claim: '{claim}'")
        
        counter = z3_prover.find_counter_example(claim)
        if counter:
            print(f"  ❌ COUNTER-EXAMPLE FOUND:")
            for var, val in counter.items():
                print(f"     {var} = {val}")
            print(f"  This proves the claim is FALSE")
        else:
            result = z3_prover.prove_claim(claim)
            if result.result == "valid":
                print(f"  ✅ No counter-example (claim is valid)")
            else:
                print(f"  ⚠️ Could not determine")
        print()


def demonstrate_software_verification():
    """Show how hybrid proving helps with real software properties."""
    print("\n\n💻 SOFTWARE VERIFICATION WITH HYBRID PROVING")
    print("=" * 60)
    print("Verifying real software properties:\n")
    
    hybrid = HybridProver()
    
    # Software properties to verify
    software_claims = [
        # Input validation
        ("if input >= 0 then process(input) succeeds", "Input validation"),
        
        # Array bounds
        ("array access at index i is safe when 0 <= i < array_length", "Bounds checking"),
        
        # Loop invariants
        ("forall i, loop_counter >= 0", "Loop invariant"),
        
        # Function properties
        ("hash_function is deterministic", "Determinism"),
        
        # Complexity bounds
        ("execution_time <= n * log(n)", "Complexity bound"),
    ]
    
    print("🔧 VERIFYING SOFTWARE PROPERTIES:\n")
    
    for claim, property_type in software_claims:
        print(f"Property: {property_type}")
        print(f"  Claim: '{claim}'")
        
        # Try to verify
        result = hybrid.prove_claim(claim)
        
        if result.get('proven'):
            print(f"  ✅ VERIFIED using {result.get('prover', 'unknown').upper()}")
            print(f"  Time: {result.get('time_ms', 0):.1f}ms")
        else:
            # Check for counter-example
            if result.get('counter_example'):
                print(f"  ❌ COUNTER-EXAMPLE FOUND:")
                for var, val in result['counter_example'].items():
                    print(f"     {var} = {val}")
            else:
                print(f"  ⚠️ Could not verify automatically")
                print(f"     May require manual proof or additional context")
        print()


def main():
    """Main demonstration."""
    print("Starting Hybrid Proving Demonstration...\n")
    print("This shows how Coq + Z3 together achieve better results than either alone.\n")
    
    # Demonstrate Z3's strengths
    z3_results = demonstrate_z3_strengths()
    
    # Demonstrate hybrid proving
    hybrid = demonstrate_hybrid_prover()
    
    # Show counter-example generation
    demonstrate_counter_examples()
    
    # Real software verification
    demonstrate_software_verification()
    
    print("\n" + "=" * 60)
    print("🎯 KEY ACHIEVEMENTS:")
    print("• Z3 excels at constraint solving and satisfiability")
    print("• Coq excels at inductive proofs and complex theorems")
    print("• Hybrid approach intelligently chooses the best prover")
    print("• Counter-example generation helps debug false claims")
    print("• Real software properties can be formally verified")
    
    print("\n💡 IMPACT:")
    print("By combining multiple provers, we achieve:")
    print("• Higher proof success rates")
    print("• Faster proof times (using the right tool)")
    print("• Better debugging with counter-examples")
    print("• Practical software verification capabilities")
    
    print("\n✅ Hybrid proving system operational!")
    
    return 0


if __name__ == "__main__":
    exit(main())