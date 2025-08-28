"""Proof strategy learning and synthesis system."""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import re

from .types import FormalSpec, ProofResult

logger = logging.getLogger(__name__)


@dataclass
class ProofStrategy:
    """A proof strategy that worked for a particular pattern."""
    pattern_type: str  # arithmetic, inequality, logic, etc.
    pattern_regex: str
    tactics_sequence: List[str]
    success_rate: float
    avg_proof_time_ms: float
    examples: List[str]


class ProofStrategyLearner:
    """
    Learns successful proof strategies from completed proofs.
    
    This system observes what tactics work for different types of claims
    and builds a database of proof strategies that can be reused.
    """
    
    def __init__(self, strategies_file: str = "proof_strategies.json"):
        """Initialize the proof strategy learner."""
        self.strategies_file = Path(strategies_file)
        self.strategies: Dict[str, ProofStrategy] = {}
        self.load_strategies()
    
    def load_strategies(self):
        """Load learned strategies from file."""
        if self.strategies_file.exists():
            try:
                with open(self.strategies_file, 'r') as f:
                    data = json.load(f)
                    for key, strategy_data in data.items():
                        self.strategies[key] = ProofStrategy(**strategy_data)
                logger.info(f"Loaded {len(self.strategies)} proof strategies")
            except Exception as e:
                logger.error(f"Failed to load strategies: {e}")
    
    def save_strategies(self):
        """Save learned strategies to file."""
        try:
            data = {key: asdict(strategy) for key, strategy in self.strategies.items()}
            with open(self.strategies_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved {len(self.strategies)} proof strategies")
        except Exception as e:
            logger.error(f"Failed to save strategies: {e}")
    
    def learn_from_proof(self, spec: FormalSpec, result: ProofResult):
        """
        Learn from a successful proof.
        
        Args:
            spec: The specification that was proved
            result: The proof result
        """
        if not result.proven:
            return  # Only learn from successful proofs
        
        # Extract pattern type and tactics
        pattern_type = self._identify_pattern_type(spec)
        tactics = self._extract_tactics(spec.coq_code)
        
        if not pattern_type or not tactics:
            return
        
        # Create or update strategy
        strategy_key = f"{pattern_type}_{len(tactics)}"
        
        if strategy_key in self.strategies:
            # Update existing strategy
            strategy = self.strategies[strategy_key]
            strategy.examples.append(spec.claim.claim_text)
            strategy.success_rate = (strategy.success_rate * (len(strategy.examples) - 1) + 1.0) / len(strategy.examples)
            strategy.avg_proof_time_ms = (strategy.avg_proof_time_ms * (len(strategy.examples) - 1) + result.proof_time_ms) / len(strategy.examples)
        else:
            # Create new strategy
            pattern_regex = self._generate_pattern_regex(spec.claim.claim_text, pattern_type)
            strategy = ProofStrategy(
                pattern_type=pattern_type,
                pattern_regex=pattern_regex,
                tactics_sequence=tactics,
                success_rate=1.0,
                avg_proof_time_ms=result.proof_time_ms,
                examples=[spec.claim.claim_text]
            )
            self.strategies[strategy_key] = strategy
        
        self.save_strategies()
    
    def suggest_strategy(self, spec: FormalSpec) -> Optional[ProofStrategy]:
        """
        Suggest a proof strategy for a specification.
        
        Args:
            spec: The specification to prove
            
        Returns:
            Suggested proof strategy if found
        """
        claim_text = spec.claim.claim_text.lower()
        
        # Find matching strategies
        matches = []
        for strategy in self.strategies.values():
            if re.search(strategy.pattern_regex, claim_text):
                matches.append(strategy)
        
        if not matches:
            return None
        
        # Return strategy with highest success rate
        return max(matches, key=lambda s: s.success_rate)
    
    def _identify_pattern_type(self, spec: FormalSpec) -> Optional[str]:
        """Identify the pattern type of a specification."""
        claim_lower = spec.claim.claim_text.lower()
        
        if any(op in claim_lower for op in ['+', '-', '*', '/', 'factorial', 'fibonacci']):
            return "arithmetic"
        elif any(op in claim_lower for op in ['<', '>', '<=', '>=']):
            return "inequality"
        elif 'if' in claim_lower and 'then' in claim_lower:
            return "implication"
        elif 'forall' in claim_lower or 'exists' in claim_lower:
            return "quantifier"
        elif 'sort' in claim_lower:
            return "sorting"
        elif 'permutation' in claim_lower:
            return "permutation"
        else:
            return "general"
    
    def _extract_tactics(self, coq_code: str) -> List[str]:
        """Extract proof tactics from Coq code."""
        tactics = []
        
        # Find proof section
        proof_match = re.search(r'Proof\.(.*?)(?:Qed|Admitted)', coq_code, re.DOTALL)
        if not proof_match:
            return tactics
        
        proof_body = proof_match.group(1)
        
        # Extract individual tactics
        tactic_pattern = r'\b(reflexivity|simpl|auto|lia|omega|ring|intros?|apply|rewrite|unfold|compute|induction|destruct|exists|split|trivial)\b'
        found_tactics = re.findall(tactic_pattern, proof_body)
        
        return found_tactics
    
    def _generate_pattern_regex(self, claim_text: str, pattern_type: str) -> str:
        """Generate a regex pattern for matching similar claims."""
        if pattern_type == "arithmetic":
            # Replace numbers with \d+ pattern
            pattern = re.sub(r'\d+', r'\\d+', claim_text.lower())
            return pattern
        elif pattern_type == "inequality":
            return r'\d+\s*[<>]=?\s*\d+'
        elif pattern_type == "implication":
            return r'if\s+.+\s+then\s+.+'
        elif pattern_type == "quantifier":
            return r'(forall|exists)\s+\w+.*'
        else:
            return claim_text.lower()


class ProofSynthesizer:
    """
    Synthesizes new proofs based on learned strategies.
    
    This is where the magic happens - using learned patterns to
    automatically generate proofs for new claims.
    """
    
    def __init__(self):
        """Initialize the proof synthesizer."""
        self.learner = ProofStrategyLearner()
        self.tactic_templates = self._build_tactic_templates()
    
    def synthesize_proof(self, spec: FormalSpec) -> Optional[str]:
        """
        Synthesize a proof for a specification.
        
        Args:
            spec: The specification to prove
            
        Returns:
            Synthesized Coq proof code
        """
        # Try to find a matching strategy
        strategy = self.learner.suggest_strategy(spec)
        
        if strategy:
            return self._apply_strategy(spec, strategy)
        else:
            return self._synthesize_default_proof(spec)
    
    def _apply_strategy(self, spec: FormalSpec, strategy: ProofStrategy) -> str:
        """Apply a learned strategy to generate a proof."""
        # Extract the theorem statement
        theorem_match = re.search(r'Theorem\s+(\w+)\s*:(.*?)\.', spec.coq_code, re.DOTALL)
        if not theorem_match:
            return self._synthesize_default_proof(spec)
        
        theorem_name = theorem_match.group(1)
        theorem_statement = theorem_match.group(2)
        
        # Build proof using learned tactics
        proof_lines = ["Proof."]
        
        for tactic in strategy.tactics_sequence:
            if tactic == "intros":
                # Check if we need to introduce variables
                if "forall" in theorem_statement:
                    proof_lines.append("  intros.")
            elif tactic == "lia":
                proof_lines.append("  lia.")
            elif tactic == "reflexivity":
                proof_lines.append("  reflexivity.")
            elif tactic == "simpl":
                proof_lines.append("  simpl.")
            elif tactic == "auto":
                proof_lines.append("  auto.")
            else:
                proof_lines.append(f"  {tactic}.")
        
        proof_lines.append("Qed.")
        
        # Replace the proof in the original code
        original_proof = re.search(r'Proof\..*?(?:Qed|Admitted)\.', spec.coq_code, re.DOTALL)
        if original_proof:
            new_proof = '\n'.join(proof_lines)
            return spec.coq_code.replace(original_proof.group(0), new_proof)
        
        return spec.coq_code
    
    def _synthesize_default_proof(self, spec: FormalSpec) -> str:
        """Synthesize a default proof when no strategy matches."""
        # Analyze the theorem to determine proof approach
        theorem_match = re.search(r'Theorem\s+\w+\s*:(.*?)\.', spec.coq_code, re.DOTALL)
        if not theorem_match:
            return spec.coq_code
        
        theorem_statement = theorem_match.group(1)
        
        proof_lines = ["Proof."]
        
        # Choose tactics based on theorem structure
        if "forall" in theorem_statement:
            proof_lines.append("  intros.")
        
        if "->" in theorem_statement:
            proof_lines.append("  intros H.")
        
        if any(op in theorem_statement for op in ['<', '>', '<=', '>=']):
            proof_lines.append("  lia.")
        elif "=" in theorem_statement:
            proof_lines.append("  try reflexivity.")
            proof_lines.append("  try ring.")
            proof_lines.append("  try lia.")
        else:
            proof_lines.append("  auto.")
        
        proof_lines.append("Qed.")
        
        # Replace proof
        original_proof = re.search(r'Proof\..*?(?:Qed|Admitted)\.', spec.coq_code, re.DOTALL)
        if original_proof:
            new_proof = '\n'.join(proof_lines)
            return spec.coq_code.replace(original_proof.group(0), new_proof)
        
        return spec.coq_code
    
    def _build_tactic_templates(self) -> Dict[str, List[str]]:
        """Build templates for common proof patterns."""
        return {
            "arithmetic": ["simpl", "reflexivity"],
            "inequality": ["lia"],
            "implication": ["intros", "lia"],
            "quantifier": ["intros", "auto"],
            "induction": ["induction", "simpl", "auto"],
            "contradiction": ["intros", "destruct", "discriminate"],
            "existence": ["exists", "auto"],
        }
    
    def learn_from_failure(self, spec: FormalSpec, error_message: str):
        """
        Learn from proof failures to improve synthesis.
        
        Args:
            spec: The specification that failed
            error_message: The error from the proof attempt
        """
        # Analyze error to understand what went wrong
        if "Unable to unify" in error_message:
            # Type mismatch or wrong value
            logger.debug(f"Unification error for: {spec.claim.claim_text}")
        elif "Cannot find" in error_message:
            # Missing lemma or definition
            logger.debug(f"Missing dependency for: {spec.claim.claim_text}")
        elif "Tactic failure" in error_message:
            # Wrong tactic choice
            logger.debug(f"Tactic failure for: {spec.claim.claim_text}")
        
        # TODO: Implement learning from failures to avoid similar mistakes


class AdaptiveProver:
    """
    Adaptive prover that learns and improves over time.
    
    This combines proof synthesis with learning to create a system
    that gets better at proving theorems as it sees more examples.
    """
    
    def __init__(self):
        """Initialize the adaptive prover."""
        self.synthesizer = ProofSynthesizer()
        self.success_count = 0
        self.failure_count = 0
    
    def prove_adaptive(self, spec: FormalSpec) -> ProofResult:
        """
        Attempt to prove using learned strategies.
        
        Args:
            spec: The specification to prove
            
        Returns:
            Proof result
        """
        # First, try to synthesize an improved proof
        improved_coq = self.synthesizer.synthesize_proof(spec)
        
        # Create modified spec with improved proof
        improved_spec = FormalSpec(
            claim=spec.claim,
            spec_text=spec.spec_text,
            coq_code=improved_coq,
            variables=spec.variables
        )
        
        # Attempt proof with improved spec
        from .prover import CoqProver
        prover = CoqProver(timeout_seconds=10)
        result = prover.prove_specification(improved_spec)
        
        # Learn from the result
        if result.proven:
            self.success_count += 1
            self.synthesizer.learner.learn_from_proof(improved_spec, result)
            logger.info(f"Adaptive proof successful for: {spec.claim.claim_text}")
        else:
            self.failure_count += 1
            self.synthesizer.learn_from_failure(improved_spec, result.error_message or "")
            logger.debug(f"Adaptive proof failed for: {spec.claim.claim_text}")
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get adaptive prover statistics."""
        total = self.success_count + self.failure_count
        success_rate = self.success_count / total if total > 0 else 0.0
        
        return {
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": success_rate,
            "total_attempts": total,
            "strategies_learned": len(self.synthesizer.learner.strategies)
        }