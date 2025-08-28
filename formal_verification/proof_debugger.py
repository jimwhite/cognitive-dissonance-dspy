"""Proof debugging and repair system for failed proofs."""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from .types import FormalSpec, ProofResult

logger = logging.getLogger(__name__)


class ProofErrorType(Enum):
    """Types of proof errors."""
    UNIFICATION_ERROR = "unification"  # Unable to unify X with Y
    TACTIC_FAILURE = "tactic_failure"  # Tactic failed to make progress
    TYPE_ERROR = "type_error"  # Type mismatch
    UNDEFINED_REFERENCE = "undefined"  # Variable or lemma not found
    TIMEOUT = "timeout"  # Proof took too long
    SYNTAX_ERROR = "syntax"  # Coq syntax error
    ADMITTED = "admitted"  # Proof uses Admitted
    UNKNOWN = "unknown"


@dataclass
class ProofDiagnosis:
    """Diagnosis of a proof failure."""
    error_type: ProofErrorType
    error_location: Optional[str]  # Line number or tactic
    error_details: str
    suggested_fixes: List[str]
    confidence: float  # How confident we are in the diagnosis


@dataclass
class ProofRepair:
    """A suggested repair for a failed proof."""
    description: str
    modified_coq: str
    repair_type: str  # tactic_replacement, lemma_addition, type_fix, etc.
    confidence: float


class ProofDebugger:
    """
    Analyzes failed proofs to provide actionable feedback.
    
    This system understands common proof failures and suggests fixes,
    making formal verification more accessible to developers.
    """
    
    def __init__(self):
        """Initialize the proof debugger."""
        self.error_patterns = self._build_error_patterns()
        self.repair_strategies = self._build_repair_strategies()
    
    def diagnose_failure(self, spec: FormalSpec, result: ProofResult) -> ProofDiagnosis:
        """
        Diagnose why a proof failed.
        
        Args:
            spec: The specification that failed
            result: The proof result with error message
            
        Returns:
            Diagnosis with error type and suggested fixes
        """
        if not result.error_message:
            return ProofDiagnosis(
                error_type=ProofErrorType.UNKNOWN,
                error_location=None,
                error_details="No error message provided",
                suggested_fixes=[],
                confidence=0.0
            )
        
        error_msg = result.error_message
        
        # Identify error type
        error_type = self._identify_error_type(error_msg)
        
        # Extract error location
        location = self._extract_error_location(error_msg)
        
        # Get error details
        details = self._extract_error_details(error_msg, error_type)
        
        # Generate suggested fixes
        fixes = self._suggest_fixes(spec, error_type, details)
        
        # Calculate confidence
        confidence = self._calculate_confidence(error_type, fixes)
        
        return ProofDiagnosis(
            error_type=error_type,
            error_location=location,
            error_details=details,
            suggested_fixes=fixes,
            confidence=confidence
        )
    
    def suggest_repair(self, spec: FormalSpec, diagnosis: ProofDiagnosis) -> Optional[ProofRepair]:
        """
        Suggest a concrete repair for the failed proof.
        
        Args:
            spec: The failed specification
            diagnosis: The diagnosis of the failure
            
        Returns:
            Suggested repair if possible
        """
        if diagnosis.error_type == ProofErrorType.UNIFICATION_ERROR:
            return self._repair_unification_error(spec, diagnosis)
        elif diagnosis.error_type == ProofErrorType.TACTIC_FAILURE:
            return self._repair_tactic_failure(spec, diagnosis)
        elif diagnosis.error_type == ProofErrorType.TYPE_ERROR:
            return self._repair_type_error(spec, diagnosis)
        elif diagnosis.error_type == ProofErrorType.UNDEFINED_REFERENCE:
            return self._repair_undefined_reference(spec, diagnosis)
        else:
            return None
    
    def _identify_error_type(self, error_msg: str) -> ProofErrorType:
        """Identify the type of error from the message."""
        error_lower = error_msg.lower()
        
        if "unable to unify" in error_lower:
            return ProofErrorType.UNIFICATION_ERROR
        elif "tactic failure" in error_lower:
            return ProofErrorType.TACTIC_FAILURE
        elif "type" in error_lower and "expected" in error_lower:
            return ProofErrorType.TYPE_ERROR
        elif "not found" in error_lower or "undefined" in error_lower:
            return ProofErrorType.UNDEFINED_REFERENCE
        elif "timeout" in error_lower:
            return ProofErrorType.TIMEOUT
        elif "syntax error" in error_lower or "parse" in error_lower:
            return ProofErrorType.SYNTAX_ERROR
        elif "admitted" in error_lower:
            return ProofErrorType.ADMITTED
        else:
            return ProofErrorType.UNKNOWN
    
    def _extract_error_location(self, error_msg: str) -> Optional[str]:
        """Extract the location of the error."""
        # Look for line numbers
        line_match = re.search(r'line (\d+)', error_msg)
        if line_match:
            return f"Line {line_match.group(1)}"
        
        # Look for character positions
        char_match = re.search(r'characters? (\d+)-(\d+)', error_msg)
        if char_match:
            return f"Characters {char_match.group(1)}-{char_match.group(2)}"
        
        return None
    
    def _extract_error_details(self, error_msg: str, error_type: ProofErrorType) -> str:
        """Extract detailed error information."""
        if error_type == ProofErrorType.UNIFICATION_ERROR:
            # Extract what couldn't be unified
            match = re.search(r'Unable to unify "([^"]+)" with "([^"]+)"', error_msg)
            if match:
                return f"Cannot unify '{match.group(1)}' with '{match.group(2)}'"
        
        elif error_type == ProofErrorType.TACTIC_FAILURE:
            # Extract which tactic failed
            match = re.search(r'Tactic failure[:\s]+(.+?)(?:\n|$)', error_msg)
            if match:
                return f"Tactic failed: {match.group(1)}"
        
        elif error_type == ProofErrorType.UNDEFINED_REFERENCE:
            # Extract what's undefined
            match = re.search(r'(?:reference|variable|identifier)\s+(\w+)\s+(?:was\s+)?not found', error_msg, re.IGNORECASE)
            if match:
                return f"Undefined reference: {match.group(1)}"
        
        # Default: return first non-empty line
        lines = error_msg.split('\n')
        for line in lines:
            if line.strip() and not line.startswith('File'):
                return line.strip()[:200]
        
        return "Error details unavailable"
    
    def _suggest_fixes(self, spec: FormalSpec, error_type: ProofErrorType, details: str) -> List[str]:
        """Generate suggested fixes based on error type."""
        fixes = []
        
        if error_type == ProofErrorType.UNIFICATION_ERROR:
            fixes.append("Check that the values match the expected result")
            fixes.append("Verify the computation is correct")
            fixes.append("Try using 'compute' or 'simpl' before reflexivity")
            
            # Specific fix for arithmetic
            if "=" in spec.claim.claim_text:
                fixes.append("Ensure the arithmetic is actually correct")
        
        elif error_type == ProofErrorType.TACTIC_FAILURE:
            if "lia" in details.lower():
                fixes.append("Try 'omega' instead of 'lia'")
                fixes.append("Ensure all variables are of type nat or Z")
                fixes.append("Check that the goal is a linear arithmetic formula")
            elif "reflexivity" in details.lower():
                fixes.append("Try 'simpl' or 'compute' first")
                fixes.append("Check if the terms are actually equal")
            else:
                fixes.append("Try a different tactic")
                fixes.append("Break down the proof into smaller steps")
        
        elif error_type == ProofErrorType.TYPE_ERROR:
            fixes.append("Check type annotations")
            fixes.append("Ensure all terms have compatible types")
            fixes.append("Add type conversions if needed")
        
        elif error_type == ProofErrorType.UNDEFINED_REFERENCE:
            if "x" in details or "y" in details or "n" in details:
                fixes.append("Add 'intros' to introduce variables")
                fixes.append("Check variable names match the theorem statement")
            else:
                fixes.append("Import required libraries")
                fixes.append("Define missing functions or lemmas")
        
        elif error_type == ProofErrorType.TIMEOUT:
            fixes.append("Simplify the proof")
            fixes.append("Use more efficient tactics")
            fixes.append("Break into smaller lemmas")
        
        return fixes
    
    def _calculate_confidence(self, error_type: ProofErrorType, fixes: List[str]) -> float:
        """Calculate confidence in the diagnosis."""
        if error_type == ProofErrorType.UNKNOWN:
            return 0.1
        elif error_type == ProofErrorType.UNIFICATION_ERROR:
            return 0.9  # Very clear error type
        elif error_type == ProofErrorType.TACTIC_FAILURE:
            return 0.8
        elif len(fixes) > 2:
            return 0.7
        else:
            return 0.5
    
    def _repair_unification_error(self, spec: FormalSpec, diagnosis: ProofDiagnosis) -> Optional[ProofRepair]:
        """Repair a unification error."""
        # Extract the mismatched values
        match = re.search(r"Cannot unify '(\d+)' with '(\d+)'", diagnosis.error_details)
        if match:
            wrong_value = match.group(1)
            correct_value = match.group(2)
            
            # Fix the claim
            fixed_claim = spec.claim.claim_text.replace(wrong_value, correct_value)
            
            # Update the Coq code
            modified_coq = spec.coq_code.replace(wrong_value, correct_value)
            
            return ProofRepair(
                description=f"Fix incorrect value: {wrong_value} should be {correct_value}",
                modified_coq=modified_coq,
                repair_type="value_correction",
                confidence=0.9
            )
        
        return None
    
    def _repair_tactic_failure(self, spec: FormalSpec, diagnosis: ProofDiagnosis) -> Optional[ProofRepair]:
        """Repair a tactic failure."""
        # Try alternative tactics
        tactic_alternatives = {
            "lia": ["omega", "ring", "auto with arith"],
            "reflexivity": ["simpl; reflexivity", "compute; reflexivity", "auto"],
            "auto": ["auto with arith", "intuition", "tauto"],
            "omega": ["lia", "ring", "auto with arith"]
        }
        
        # Find which tactic failed
        for tactic, alternatives in tactic_alternatives.items():
            if tactic in spec.coq_code:
                # Try first alternative
                if alternatives:
                    modified_coq = spec.coq_code.replace(tactic, alternatives[0])
                    
                    return ProofRepair(
                        description=f"Replace '{tactic}' with '{alternatives[0]}'",
                        modified_coq=modified_coq,
                        repair_type="tactic_replacement",
                        confidence=0.7
                    )
        
        return None
    
    def _repair_type_error(self, spec: FormalSpec, diagnosis: ProofDiagnosis) -> Optional[ProofRepair]:
        """Repair a type error."""
        # Add type annotations or conversions
        if "nat" in diagnosis.error_details:
            # Add nat type annotations
            modified_coq = re.sub(r'forall (\w+)', r'forall \1 : nat', spec.coq_code)
            
            return ProofRepair(
                description="Add type annotations for natural numbers",
                modified_coq=modified_coq,
                repair_type="type_annotation",
                confidence=0.6
            )
        
        return None
    
    def _repair_undefined_reference(self, spec: FormalSpec, diagnosis: ProofDiagnosis) -> Optional[ProofRepair]:
        """Repair an undefined reference error."""
        # Extract the undefined variable
        match = re.search(r"Undefined reference: (\w+)", diagnosis.error_details)
        if match:
            var_name = match.group(1)
            
            # Add intros if it's a variable
            if len(var_name) == 1:  # Single letter, likely a variable
                # Find the Proof line
                proof_match = re.search(r'Proof\.\n', spec.coq_code)
                if proof_match:
                    # Add intros after Proof
                    modified_coq = spec.coq_code.replace(
                        'Proof.\n',
                        'Proof.\n  intros.\n'
                    )
                    
                    return ProofRepair(
                        description=f"Introduce variable '{var_name}' with 'intros'",
                        modified_coq=modified_coq,
                        repair_type="add_intros",
                        confidence=0.8
                    )
        
        return None
    
    def _build_error_patterns(self) -> Dict[str, str]:
        """Build patterns for common error messages."""
        return {
            "unification": r"Unable to unify",
            "tactic": r"Tactic failure|Error: Cannot",
            "type": r"Type error|Expected type|has type",
            "undefined": r"not found|undefined|Unknown",
            "timeout": r"Timeout|time limit",
            "syntax": r"Syntax error|Parse error"
        }
    
    def _build_repair_strategies(self) -> Dict[ProofErrorType, List[str]]:
        """Build repair strategies for each error type."""
        return {
            ProofErrorType.UNIFICATION_ERROR: [
                "correct_value",
                "add_computation",
                "change_equality"
            ],
            ProofErrorType.TACTIC_FAILURE: [
                "try_alternative_tactic",
                "add_intermediate_step",
                "simplify_goal"
            ],
            ProofErrorType.TYPE_ERROR: [
                "add_type_annotation",
                "add_type_conversion",
                "fix_type_mismatch"
            ],
            ProofErrorType.UNDEFINED_REFERENCE: [
                "add_intros",
                "import_module",
                "define_missing"
            ]
        }


class InteractiveProofAssistant:
    """
    Interactive assistant for developing proofs.
    
    This provides a conversational interface for proof development,
    helping users understand and fix proof failures.
    """
    
    def __init__(self):
        """Initialize the interactive assistant."""
        self.debugger = ProofDebugger()
        self.history: List[Tuple[FormalSpec, ProofResult, ProofDiagnosis]] = []
    
    def explain_failure(self, spec: FormalSpec, result: ProofResult) -> str:
        """
        Explain a proof failure in user-friendly terms.
        
        Args:
            spec: The failed specification
            result: The proof result
            
        Returns:
            Human-readable explanation
        """
        diagnosis = self.debugger.diagnose_failure(spec, result)
        self.history.append((spec, result, diagnosis))
        
        explanation = []
        explanation.append(f"🔍 PROOF FAILURE ANALYSIS")
        explanation.append(f"Claim: '{spec.claim.claim_text}'")
        explanation.append("")
        
        # Explain the error type
        error_explanations = {
            ProofErrorType.UNIFICATION_ERROR: "The proof failed because the computed value doesn't match the expected value.",
            ProofErrorType.TACTIC_FAILURE: "The proof tactic couldn't make progress on the goal.",
            ProofErrorType.TYPE_ERROR: "There's a type mismatch in the proof.",
            ProofErrorType.UNDEFINED_REFERENCE: "The proof references something that isn't defined.",
            ProofErrorType.TIMEOUT: "The proof took too long to complete.",
            ProofErrorType.SYNTAX_ERROR: "There's a syntax error in the proof code.",
            ProofErrorType.UNKNOWN: "The proof failed for an unknown reason."
        }
        
        explanation.append(f"❌ Problem: {error_explanations.get(diagnosis.error_type, 'Unknown error')}")
        
        if diagnosis.error_location:
            explanation.append(f"📍 Location: {diagnosis.error_location}")
        
        explanation.append(f"📝 Details: {diagnosis.error_details}")
        
        if diagnosis.suggested_fixes:
            explanation.append("")
            explanation.append("💡 Suggested fixes:")
            for i, fix in enumerate(diagnosis.suggested_fixes, 1):
                explanation.append(f"   {i}. {fix}")
        
        # Suggest a repair if possible
        repair = self.debugger.suggest_repair(spec, diagnosis)
        if repair:
            explanation.append("")
            explanation.append(f"🔧 Automated repair available: {repair.description}")
            explanation.append(f"   Confidence: {repair.confidence:.0%}")
        
        return "\n".join(explanation)
    
    def get_step_by_step_guidance(self, spec: FormalSpec) -> List[str]:
        """
        Provide step-by-step guidance for proving a specification.
        
        Args:
            spec: The specification to prove
            
        Returns:
            List of steps to follow
        """
        steps = []
        claim_lower = spec.claim.claim_text.lower()
        
        # Analyze the claim to determine proof strategy
        if any(op in claim_lower for op in ['+', '-', '*', '/']):
            # Arithmetic proof
            steps.append("1. Start with 'Proof.'")
            steps.append("2. Use 'simpl.' to simplify the arithmetic")
            steps.append("3. Use 'reflexivity.' to show both sides are equal")
            steps.append("4. End with 'Qed.'")
        
        elif any(op in claim_lower for op in ['<', '>', '<=', '>=']):
            # Inequality proof
            steps.append("1. Start with 'Proof.'")
            steps.append("2. Use 'lia.' for linear arithmetic")
            steps.append("3. End with 'Qed.'")
        
        elif 'forall' in claim_lower:
            # Universal quantification
            steps.append("1. Start with 'Proof.'")
            steps.append("2. Use 'intros' to introduce variables")
            steps.append("3. Simplify or compute if needed")
            steps.append("4. Apply appropriate tactics")
            steps.append("5. End with 'Qed.'")
        
        elif 'if' in claim_lower and 'then' in claim_lower:
            # Implication
            steps.append("1. Start with 'Proof.'")
            steps.append("2. Use 'intros H' to introduce hypothesis")
            steps.append("3. Use the hypothesis to prove the conclusion")
            steps.append("4. End with 'Qed.'")
        
        else:
            # Generic proof
            steps.append("1. Start with 'Proof.'")
            steps.append("2. Try 'auto.' for automatic proving")
            steps.append("3. If that fails, break down the problem")
            steps.append("4. End with 'Qed.'")
        
        return steps