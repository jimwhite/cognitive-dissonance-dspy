"""Incremental verification system that tracks changes and re-verifies only what's necessary."""

import hashlib
import json
import time
import difflib
import ast
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
import logging

from .types import FormalSpec, ProofResult, Claim, PropertyType
from .code_analyzer import CodeAnalyzer, CodeSpecification
from .proof_strategies import AdaptiveProver

logger = logging.getLogger(__name__)


@dataclass
class FunctionSignature:
    """Signature of a function for tracking changes."""
    name: str
    parameters: List[Tuple[str, str]]
    return_type: Optional[str]
    body_hash: str
    dependencies: Set[str] = field(default_factory=set)
    last_verified: Optional[float] = None
    proof_status: str = "unverified"  # unverified, proven, failed


@dataclass
class VerificationState:
    """State of verification for a codebase."""
    functions: Dict[str, FunctionSignature] = field(default_factory=dict)
    proofs: Dict[str, ProofResult] = field(default_factory=dict)
    dependencies: Dict[str, Set[str]] = field(default_factory=dict)
    last_full_verification: Optional[float] = None
    version: int = 0


class DependencyAnalyzer:
    """
    Analyzes code dependencies to determine what needs re-verification.
    
    This is crucial for incremental verification - when function A changes,
    we need to re-verify A and all functions that depend on A.
    """
    
    def __init__(self):
        """Initialize the dependency analyzer."""
        self.call_graph: Dict[str, Set[str]] = {}
        self.reverse_deps: Dict[str, Set[str]] = {}
    
    def analyze_dependencies(self, code: str) -> Dict[str, Set[str]]:
        """
        Analyze function dependencies in code.
        
        Args:
            code: Source code to analyze
            
        Returns:
            Dictionary mapping functions to their dependencies
        """
        try:
            tree = ast.parse(code)
            
            # First pass: identify all functions
            functions = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.add(node.name)
            
            # Second pass: analyze each function's dependencies
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    deps = self._analyze_function_deps(node, functions)
                    self.call_graph[node.name] = deps
            
            # Build reverse dependency graph
            self._build_reverse_deps()
            
            return self.call_graph
        
        except SyntaxError as e:
            logger.error(f"Failed to analyze dependencies: {e}")
            return {}
    
    def _analyze_function_deps(self, func_node: ast.FunctionDef, known_functions: Set[str]) -> Set[str]:
        """Analyze dependencies of a single function."""
        deps = set()
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in known_functions:
                        deps.add(func_name)
        
        return deps
    
    def _build_reverse_deps(self):
        """Build reverse dependency graph."""
        self.reverse_deps = {}
        
        for func, deps in self.call_graph.items():
            for dep in deps:
                if dep not in self.reverse_deps:
                    self.reverse_deps[dep] = set()
                self.reverse_deps[dep].add(func)
    
    def get_affected_functions(self, changed_function: str) -> Set[str]:
        """
        Get all functions affected by a change.
        
        Args:
            changed_function: Name of the changed function
            
        Returns:
            Set of functions that need re-verification
        """
        affected = {changed_function}
        
        # Add all functions that depend on the changed function
        if changed_function in self.reverse_deps:
            to_check = list(self.reverse_deps[changed_function])
            
            while to_check:
                func = to_check.pop()
                if func not in affected:
                    affected.add(func)
                    # Add functions that depend on this one
                    if func in self.reverse_deps:
                        to_check.extend(self.reverse_deps[func])
        
        return affected


class ChangeDetector:
    """
    Detects changes in code that require re-verification.
    
    This uses AST comparison and hashing to efficiently detect
    semantic changes while ignoring cosmetic changes.
    """
    
    def __init__(self):
        """Initialize the change detector."""
        self.analyzer = CodeAnalyzer()
    
    def detect_changes(self, old_code: str, new_code: str) -> Dict[str, str]:
        """
        Detect function-level changes between code versions.
        
        Args:
            old_code: Previous version of code
            new_code: New version of code
            
        Returns:
            Dictionary of changed functions and change types
        """
        changes = {}
        
        # Parse both versions
        old_specs = self._extract_function_specs(old_code)
        new_specs = self._extract_function_specs(new_code)
        
        # Find added functions
        for func_name in new_specs:
            if func_name not in old_specs:
                changes[func_name] = "added"
        
        # Find removed functions
        for func_name in old_specs:
            if func_name not in new_specs:
                changes[func_name] = "removed"
        
        # Find modified functions
        for func_name in new_specs:
            if func_name in old_specs:
                if self._has_semantic_change(old_specs[func_name], new_specs[func_name]):
                    changes[func_name] = "modified"
        
        return changes
    
    def _extract_function_specs(self, code: str) -> Dict[str, FunctionSignature]:
        """Extract function signatures from code."""
        specs = {}
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    sig = self._create_signature(node)
                    specs[node.name] = sig
        
        except SyntaxError:
            pass
        
        return specs
    
    def _create_signature(self, func_node: ast.FunctionDef) -> FunctionSignature:
        """Create a function signature from AST node."""
        # Extract parameters
        params = []
        for arg in func_node.args.args:
            param_name = arg.arg
            param_type = "Any"  # Simplified
            params.append((param_name, param_type))
        
        # Hash the function body
        body_str = ast.unparse(func_node) if hasattr(ast, 'unparse') else str(func_node)
        body_hash = hashlib.sha256(body_str.encode()).hexdigest()[:16]
        
        return FunctionSignature(
            name=func_node.name,
            parameters=params,
            return_type="Any",
            body_hash=body_hash
        )
    
    def _has_semantic_change(self, old_sig: FunctionSignature, new_sig: FunctionSignature) -> bool:
        """Check if there's a semantic change between signatures."""
        # Check if parameters changed
        if old_sig.parameters != new_sig.parameters:
            return True
        
        # Check if body changed (using hash)
        if old_sig.body_hash != new_sig.body_hash:
            return True
        
        return False


class IncrementalVerifier:
    """
    Main incremental verification system.
    
    This orchestrates change detection, dependency analysis, and
    selective re-verification to minimize proof overhead.
    """
    
    def __init__(self, state_file: str = "verification_state.json"):
        """Initialize the incremental verifier."""
        self.state_file = Path(state_file)
        self.state = self._load_state()
        self.change_detector = ChangeDetector()
        self.dep_analyzer = DependencyAnalyzer()
        self.code_analyzer = CodeAnalyzer()
        self.prover = AdaptiveProver()
    
    def _load_state(self) -> VerificationState:
        """Load verification state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    state = VerificationState()
                    
                    # Reconstruct function signatures
                    for name, sig_data in data.get('functions', {}).items():
                        sig = FunctionSignature(
                            name=sig_data['name'],
                            parameters=sig_data['parameters'],
                            return_type=sig_data.get('return_type'),
                            body_hash=sig_data['body_hash'],
                            dependencies=set(sig_data.get('dependencies', [])),
                            last_verified=sig_data.get('last_verified'),
                            proof_status=sig_data.get('proof_status', 'unverified')
                        )
                        state.functions[name] = sig
                    
                    state.last_full_verification = data.get('last_full_verification')
                    state.version = data.get('version', 0)
                    
                    return state
            
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
        
        return VerificationState()
    
    def _save_state(self):
        """Save verification state to file."""
        try:
            data = {
                'functions': {},
                'last_full_verification': self.state.last_full_verification,
                'version': self.state.version
            }
            
            # Serialize function signatures
            for name, sig in self.state.functions.items():
                data['functions'][name] = {
                    'name': sig.name,
                    'parameters': sig.parameters,
                    'return_type': sig.return_type,
                    'body_hash': sig.body_hash,
                    'dependencies': list(sig.dependencies),
                    'last_verified': sig.last_verified,
                    'proof_status': sig.proof_status
                }
            
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
        
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def verify_incremental(self, code: str, force_full: bool = False) -> Dict[str, Any]:
        """
        Perform incremental verification on code.
        
        Args:
            code: Current code to verify
            force_full: Force full re-verification
            
        Returns:
            Verification results
        """
        start_time = time.time()
        
        # Analyze dependencies
        self.dep_analyzer.analyze_dependencies(code)
        
        # Detect changes if we have previous state
        if self.state.functions and not force_full:
            # Create previous code representation
            old_code = self._reconstruct_old_code()
            changes = self.change_detector.detect_changes(old_code, code)
            
            # Determine what needs re-verification
            functions_to_verify = set()
            for changed_func, change_type in changes.items():
                if change_type != "removed":
                    affected = self.dep_analyzer.get_affected_functions(changed_func)
                    functions_to_verify.update(affected)
            
            logger.info(f"Incremental verification: {len(functions_to_verify)} functions to verify")
        else:
            # Full verification
            specs = self.code_analyzer.analyze(code)
            functions_to_verify = {spec.function_name for spec in specs}
            logger.info(f"Full verification: {len(functions_to_verify)} functions")
        
        # Perform verification
        results = self._verify_functions(code, functions_to_verify)
        
        # Update state
        self._update_state(code, results)
        self.state.version += 1
        self._save_state()
        
        # Calculate statistics
        total_time = (time.time() - start_time) * 1000
        verified_count = len([r for r in results['proofs'] if r.proven])
        
        return {
            'incremental': not force_full,
            'functions_verified': len(functions_to_verify),
            'functions_proven': verified_count,
            'total_functions': len(self.state.functions),
            'verification_time_ms': total_time,
            'version': self.state.version,
            'proofs': results['proofs']
        }
    
    def _reconstruct_old_code(self) -> str:
        """Reconstruct approximate old code from state."""
        # This is simplified - in reality, we'd store the actual old code
        return ""
    
    def _verify_functions(self, code: str, function_names: Set[str]) -> Dict[str, Any]:
        """Verify specific functions."""
        proofs = []
        
        # Extract specifications for functions to verify
        specs = self.code_analyzer.analyze(code)
        
        for spec in specs:
            if spec.function_name in function_names:
                # Create claims from specification
                claims = self._create_claims_from_spec(spec)
                
                # Verify claims
                for claim in claims:
                    from .detector import FormalVerificationConflictDetector
                    detector = FormalVerificationConflictDetector()
                    results = detector.analyze_claims([claim])
                    
                    if results['proof_results']:
                        proofs.extend(results['proof_results'])
        
        return {'proofs': proofs}
    
    def _create_claims_from_spec(self, spec: CodeSpecification) -> List[Claim]:
        """Create formal claims from code specification."""
        claims = []
        
        # Create claims for preconditions
        for precond in spec.preconditions:
            claim = Claim(
                agent_id=f"{spec.function_name}_precond",
                claim_text=precond,
                property_type=PropertyType.CORRECTNESS,
                confidence=0.9,
                timestamp=time.time()
            )
            claims.append(claim)
        
        # Create claims for postconditions
        for postcond in spec.postconditions:
            claim = Claim(
                agent_id=f"{spec.function_name}_postcond",
                claim_text=postcond,
                property_type=PropertyType.CORRECTNESS,
                confidence=0.9,
                timestamp=time.time()
            )
            claims.append(claim)
        
        return claims
    
    def _update_state(self, code: str, results: Dict[str, Any]):
        """Update verification state with results."""
        # Update function signatures
        specs = self.code_analyzer.analyze(code)
        
        for spec in specs:
            sig = FunctionSignature(
                name=spec.function_name,
                parameters=spec.parameters,
                return_type=spec.return_type,
                body_hash=hashlib.sha256(spec.source_code.encode()).hexdigest()[:16],
                dependencies=self.dep_analyzer.call_graph.get(spec.function_name, set()),
                last_verified=time.time(),
                proof_status="proven" if any(p.proven for p in results['proofs'] if spec.function_name in str(p.spec.claim.agent_id)) else "failed"
            )
            self.state.functions[spec.function_name] = sig
    
    def get_verification_status(self) -> Dict[str, Any]:
        """Get current verification status."""
        proven_count = len([f for f in self.state.functions.values() if f.proof_status == "proven"])
        failed_count = len([f for f in self.state.functions.values() if f.proof_status == "failed"])
        unverified_count = len([f for f in self.state.functions.values() if f.proof_status == "unverified"])
        
        return {
            'total_functions': len(self.state.functions),
            'proven': proven_count,
            'failed': failed_count,
            'unverified': unverified_count,
            'coverage': proven_count / len(self.state.functions) if self.state.functions else 0,
            'version': self.state.version,
            'last_verification': self.state.last_full_verification
        }
    
    def get_function_status(self, function_name: str) -> Optional[Dict[str, Any]]:
        """Get verification status of a specific function."""
        if function_name in self.state.functions:
            sig = self.state.functions[function_name]
            return {
                'name': sig.name,
                'status': sig.proof_status,
                'last_verified': sig.last_verified,
                'dependencies': list(sig.dependencies),
                'dependents': list(self.dep_analyzer.reverse_deps.get(function_name, []))
            }
        return None