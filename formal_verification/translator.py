"""Natural language claim to formal specification translator."""

import re
import logging
from typing import Optional

from .types import Claim, FormalSpec, PropertyType

logger = logging.getLogger(__name__)


class ClaimTranslator:
    """Translates natural language claims to formal Coq specifications."""
    
    def __init__(self):
        """Initialize the claim translator with pattern matching rules."""
        self.memory_patterns = [
            (r"memory safe|no buffer overflow|no use.after.free", self._memory_safety_spec),
            (r"buffer overflow|memory corruption|segfault", self._memory_safety_spec),
        ]
        
        self.complexity_patterns = [
            (r"O\((\w+)\)|time complexity.*O\((\w+)\)", self._complexity_spec),
            (r"linear time|O\(n\)", self._complexity_spec),
            (r"constant time|O\(1\)", self._complexity_spec),
        ]
        
        self.correctness_patterns = [
            (r"sorts? the array|sorting|sorted|correctly sorts", self._sorting_correctness_spec),
            (r"returns? the (maximum|minimum)|finds the (maximum|minimum)", self._extremum_correctness_spec),
            (r"computes? the sum", self._sum_correctness_spec),
            (r"binary.?search.*returns.*correct.*index|binary.?search.*finds.*element", self._binary_search_correctness_spec),
            (r"preserves all elements|permutation", self._permutation_correctness_spec),
        ]
        
        self.mathematical_patterns = [
            (r"(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)", self._arithmetic_spec),
            (r"factorial\s+(\d+)\s*=\s*(\d+)", self._factorial_spec),
        ]
    
    def translate(self, claim: Claim, code: str) -> Optional[FormalSpec]:
        """Convert a claim to formal specification.
        
        Args:
            claim: The claim to translate
            code: The code being analyzed
            
        Returns:
            FormalSpec if translation successful, None otherwise
        """
        claim_lower = claim.claim_text.lower()
        
        # Try memory safety patterns
        for pattern, spec_generator in self.memory_patterns:
            if re.search(pattern, claim_lower):
                return spec_generator(claim, code)
        
        # Try complexity patterns  
        for pattern, spec_generator in self.complexity_patterns:
            match = re.search(pattern, claim_lower)
            if match:
                return spec_generator(claim, code, match)
        
        # Try correctness patterns
        for pattern, spec_generator in self.correctness_patterns:
            if re.search(pattern, claim_lower):
                return spec_generator(claim, code)
        
        # Try mathematical patterns
        for pattern, spec_generator in self.mathematical_patterns:
            match = re.search(pattern, claim_lower)
            if match:
                return spec_generator(claim, code, match)
        
        logger.warning(f"Could not translate claim: {claim.claim_text}")
        return None
    
    def _extract_function_name(self, code: str) -> str:
        """Extract function name from code."""
        func_match = re.search(r'fn\s+(\w+)', code)
        return func_match.group(1) if func_match else "function"
    
    def _memory_safety_spec(self, claim: Claim, code: str) -> FormalSpec:
        """Generate memory safety specification."""
        func_name = self._extract_function_name(code)
        
        spec_text = f"Function {func_name} does not cause buffer overflows or use-after-free"
        coq_code = f"""
Require Import List.
Require Import Arith.

Definition {func_name}_safe (input: list nat) : Prop :=
  forall i, i < length input -> 
    exists result, {func_name} input = Some result /\\
    (forall j, j < length result -> nth j result 0 < max_val).

Theorem {func_name}_memory_safe : 
  forall input, {func_name}_safe input.
Proof.
  (* Proof would establish memory safety *)
Admitted.
"""
        
        return FormalSpec(
            claim=claim,
            spec_text=spec_text,
            coq_code=coq_code,
            variables={"func_name": func_name}
        )
    
    def _extremum_correctness_spec(self, claim: Claim, code: str) -> FormalSpec:
        """Generate extremum finding correctness specification."""
        func_name = self._extract_function_name(code)
        
        spec_text = f"Function {func_name} correctly finds the extremum value"
        coq_code = f"""
Require Import List.
Require Import Arith.

Definition {func_name}_correct (input: list nat) (output: nat) : Prop :=
  In output input /\\ (forall x, In x input -> x <= output).

Theorem {func_name}_finds_extremum :
  forall input, input <> nil ->
    exists output, {func_name} input = Some output /\\ {func_name}_correct input output.
Proof.
  (* Proof would verify extremum finding correctness *)
Admitted.
"""
        
        return FormalSpec(
            claim=claim,
            spec_text=spec_text,
            coq_code=coq_code,
            variables={"func_name": func_name}
        )
    
    def _sum_correctness_spec(self, claim: Claim, code: str) -> FormalSpec:
        """Generate sum computation correctness specification."""
        func_name = self._extract_function_name(code)
        
        spec_text = f"Function {func_name} correctly computes the sum"
        coq_code = f"""
Require Import List.
Require Import Arith.

Fixpoint list_sum (l: list nat) : nat :=
  match l with
  | nil => 0
  | h :: t => h + list_sum t
  end.

Definition {func_name}_correct (input: list nat) (output: nat) : Prop :=
  output = list_sum input.

Theorem {func_name}_computes_sum :
  forall input, exists output,
    {func_name} input = output /\\ {func_name}_correct input output.
Proof.
  (* Proof would verify sum computation correctness *)
Admitted.
"""
        
        return FormalSpec(
            claim=claim,
            spec_text=spec_text,
            coq_code=coq_code,
            variables={"func_name": func_name}
        )
    
    def _arithmetic_spec(self, claim: Claim, code: str, match) -> FormalSpec:
        """Generate arithmetic correctness specification."""
        left = int(match.group(1))
        middle = int(match.group(2))  
        right = int(match.group(3))
        
        spec_text = f"Arithmetic claim: {left} + {middle} = {right}"
        coq_code = f"""
Require Import Arith.

Theorem arithmetic_claim : {left} + {middle} = {right}.
Proof.
  reflexivity.
Qed.
"""
        
        return FormalSpec(
            claim=claim,
            spec_text=spec_text,
            coq_code=coq_code,
            variables={"left": str(left), "middle": str(middle), "right": str(right)}
        )
    
    def _factorial_spec(self, claim: Claim, code: str, match) -> FormalSpec:
        """Generate factorial correctness specification."""
        input_val = int(match.group(1))
        output_val = int(match.group(2))
        
        spec_text = f"Factorial claim: factorial {input_val} = {output_val}"
        coq_code = f"""
Require Import Arith.

Fixpoint factorial (n : nat) : nat :=
  match n with
  | 0 => 1
  | S n' => n * factorial n'
  end.

Theorem factorial_claim : factorial {input_val} = {output_val}.
Proof.
  simpl.
  reflexivity.
Qed.
"""
        
        return FormalSpec(
            claim=claim,
            spec_text=spec_text,
            coq_code=coq_code,
            variables={"input": str(input_val), "output": str(output_val)}
        )
    
    def _complexity_spec(self, claim: Claim, code: str, match=None) -> FormalSpec:
        """Generate time complexity specification."""
        complexity = "n"  # Default
        if match:
            complexity = match.group(1) or match.group(2) or "n"
        
        func_name = self._extract_function_name(code)
        
        spec_text = f"Function {func_name} has time complexity O({complexity})"
        coq_code = f"""
Require Import Omega.

Definition time_complexity_{func_name} (n: nat) : nat := 
  (* Time function would be derived from code analysis *)
  n. (* Simplified for example *)

Theorem {func_name}_complexity :
  exists c k, forall n, n >= k -> 
    time_complexity_{func_name} n <= c * ({complexity}).
Proof.
  (* Proof would establish the complexity bound *)
Admitted.
"""
        
        return FormalSpec(
            claim=claim,
            spec_text=spec_text,
            coq_code=coq_code,
            variables={"func_name": func_name, "complexity": complexity}
        )
    
    def _sorting_correctness_spec(self, claim: Claim, code: str) -> FormalSpec:
        """Generate sorting correctness specification."""
        func_name = self._extract_function_name(code)
        
        spec_text = f"Function {func_name} correctly sorts its input"
        coq_code = f"""
Require Import List.
Require Import Sorted.
Require Import Permutation.

Definition {func_name}_correct (input output: list nat) : Prop :=
  Permutation input output /\\ LocallySorted le output.

Theorem {func_name}_sorts_correctly :
  forall input, exists output,
    {func_name} input = output /\\ {func_name}_correct input output.
Proof.
  (* Proof would verify sorting correctness *)
Admitted.
"""
        
        return FormalSpec(
            claim=claim,
            spec_text=spec_text,
            coq_code=coq_code,
            variables={"func_name": func_name}
        )
    
    def _extremum_correctness_spec(self, claim: Claim, code: str) -> FormalSpec:
        """Generate extremum finding correctness specification."""
        func_name = self._extract_function_name(code)
        
        spec_text = f"Function {func_name} correctly finds the extremum value"
        coq_code = f"""
Require Import List.
Require Import Arith.

Definition {func_name}_correct (input: list nat) (output: nat) : Prop :=
  In output input /\\ (forall x, In x input -> x <= output).

Theorem {func_name}_finds_extremum :
  forall input, input <> nil ->
    exists output, {func_name} input = Some output /\\ {func_name}_correct input output.
Proof.
  (* Proof would verify extremum finding correctness *)
Admitted.
"""
        
        return FormalSpec(
            claim=claim,
            spec_text=spec_text,
            coq_code=coq_code,
            variables={"func_name": func_name}
        )
    
    def _sum_correctness_spec(self, claim: Claim, code: str) -> FormalSpec:
        """Generate sum computation correctness specification."""
        func_name = self._extract_function_name(code)
        
        spec_text = f"Function {func_name} correctly computes the sum"
        coq_code = f"""
Require Import List.
Require Import Arith.

Fixpoint list_sum (l: list nat) : nat :=
  match l with
  | nil => 0
  | h :: t => h + list_sum t
  end.

Definition {func_name}_correct (input: list nat) (output: nat) : Prop :=
  output = list_sum input.

Theorem {func_name}_computes_sum :
  forall input, exists output,
    {func_name} input = output /\\ {func_name}_correct input output.
Proof.
  (* Proof would verify sum computation correctness *)
Admitted.
"""
        
        return FormalSpec(
            claim=claim,
            spec_text=spec_text,
            coq_code=coq_code,
            variables={"func_name": func_name}
        )
    
    def _binary_search_correctness_spec(self, claim: Claim, code: str) -> FormalSpec:
        """Generate binary search correctness specification."""
        func_name = self._extract_function_name(code)
        
        spec_text = f"Function {func_name} correctly implements binary search"
        coq_code = f"""
Require Import List.
Require Import Arith.
Require Import Sorted.

Definition {func_name}_correct (input: list nat) (target: nat) (result: option nat) : Prop :=
  match result with
  | Some idx => idx < length input /\\ nth idx input 0 = target
  | None => ~In target input
  end.

Theorem {func_name}_search_correct :
  forall input target, LocallySorted le input ->
    exists result, {func_name} input target = result /\\ {func_name}_correct input target result.
Proof.
  (* Proof would verify binary search correctness *)
Admitted.
"""
        
        return FormalSpec(
            claim=claim,
            spec_text=spec_text,
            coq_code=coq_code,
            variables={"func_name": func_name}
        )
    
    def _extremum_correctness_spec(self, claim: Claim, code: str) -> FormalSpec:
        """Generate extremum finding correctness specification."""
        func_name = self._extract_function_name(code)
        
        spec_text = f"Function {func_name} correctly finds the extremum value"
        coq_code = f"""
Require Import List.
Require Import Arith.

Definition {func_name}_correct (input: list nat) (output: nat) : Prop :=
  In output input /\\ (forall x, In x input -> x <= output).

Theorem {func_name}_finds_extremum :
  forall input, input <> nil ->
    exists output, {func_name} input = Some output /\\ {func_name}_correct input output.
Proof.
  (* Proof would verify extremum finding correctness *)
Admitted.
"""
        
        return FormalSpec(
            claim=claim,
            spec_text=spec_text,
            coq_code=coq_code,
            variables={"func_name": func_name}
        )
    
    def _sum_correctness_spec(self, claim: Claim, code: str) -> FormalSpec:
        """Generate sum computation correctness specification."""
        func_name = self._extract_function_name(code)
        
        spec_text = f"Function {func_name} correctly computes the sum"
        coq_code = f"""
Require Import List.
Require Import Arith.

Fixpoint list_sum (l: list nat) : nat :=
  match l with
  | nil => 0
  | h :: t => h + list_sum t
  end.

Definition {func_name}_correct (input: list nat) (output: nat) : Prop :=
  output = list_sum input.

Theorem {func_name}_computes_sum :
  forall input, exists output,
    {func_name} input = output /\\ {func_name}_correct input output.
Proof.
  (* Proof would verify sum computation correctness *)
Admitted.
"""
        
        return FormalSpec(
            claim=claim,
            spec_text=spec_text,
            coq_code=coq_code,
            variables={"func_name": func_name}
        )
    
    def _permutation_correctness_spec(self, claim: Claim, code: str) -> FormalSpec:
        """Generate permutation preservation specification."""
        func_name = self._extract_function_name(code)
        
        spec_text = f"Function {func_name} preserves all elements (permutation)"
        coq_code = f"""
Require Import List.
Require Import Permutation.

Definition {func_name}_preserves_elements (input output: list nat) : Prop :=
  Permutation input output.

Theorem {func_name}_permutation_correct :
  forall input, exists output,
    {func_name} input = output /\\ {func_name}_preserves_elements input output.
Proof.
  (* Proof would verify permutation preservation *)
Admitted.
"""
        
        return FormalSpec(
            claim=claim,
            spec_text=spec_text,
            coq_code=coq_code,
            variables={"func_name": func_name}
        )
    
    def _extremum_correctness_spec(self, claim: Claim, code: str) -> FormalSpec:
        """Generate extremum finding correctness specification."""
        func_name = self._extract_function_name(code)
        
        spec_text = f"Function {func_name} correctly finds the extremum value"
        coq_code = f"""
Require Import List.
Require Import Arith.

Definition {func_name}_correct (input: list nat) (output: nat) : Prop :=
  In output input /\\ (forall x, In x input -> x <= output).

Theorem {func_name}_finds_extremum :
  forall input, input <> nil ->
    exists output, {func_name} input = Some output /\\ {func_name}_correct input output.
Proof.
  (* Proof would verify extremum finding correctness *)
Admitted.
"""
        
        return FormalSpec(
            claim=claim,
            spec_text=spec_text,
            coq_code=coq_code,
            variables={"func_name": func_name}
        )
    
    def _sum_correctness_spec(self, claim: Claim, code: str) -> FormalSpec:
        """Generate sum computation correctness specification."""
        func_name = self._extract_function_name(code)
        
        spec_text = f"Function {func_name} correctly computes the sum"
        coq_code = f"""
Require Import List.
Require Import Arith.

Fixpoint list_sum (l: list nat) : nat :=
  match l with
  | nil => 0
  | h :: t => h + list_sum t
  end.

Definition {func_name}_correct (input: list nat) (output: nat) : Prop :=
  output = list_sum input.

Theorem {func_name}_computes_sum :
  forall input, exists output,
    {func_name} input = output /\\ {func_name}_correct input output.
Proof.
  (* Proof would verify sum computation correctness *)
Admitted.
"""
        
        return FormalSpec(
            claim=claim,
            spec_text=spec_text,
            coq_code=coq_code,
            variables={"func_name": func_name}
        )