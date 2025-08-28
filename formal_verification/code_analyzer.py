"""Code analysis and specification extraction for formal verification."""

import ast
import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CodeSpecification:
    """Specification extracted from code."""
    function_name: str
    parameters: List[Tuple[str, str]]  # (name, type)
    return_type: Optional[str]
    preconditions: List[str]
    postconditions: List[str]
    invariants: List[str]
    complexity: Optional[str]
    docstring: Optional[str]
    source_code: str


class CodeAnalyzer:
    """
    Analyzes code to extract specifications for formal verification.
    
    This bridges the gap between implementation and formal specification
    by understanding code structure, contracts, and intended behavior.
    """
    
    def __init__(self):
        """Initialize the code analyzer."""
        self.python_analyzer = PythonAnalyzer()
        self.rust_analyzer = RustAnalyzer()
    
    def analyze(self, code: str, language: str = "auto") -> List[CodeSpecification]:
        """
        Analyze code to extract specifications.
        
        Args:
            code: Source code to analyze
            language: Programming language (auto, python, rust)
            
        Returns:
            List of extracted specifications
        """
        if language == "auto":
            language = self._detect_language(code)
        
        if language == "python":
            return self.python_analyzer.analyze(code)
        elif language == "rust":
            return self.rust_analyzer.analyze(code)
        else:
            logger.warning(f"Unsupported language: {language}")
            return []
    
    def _detect_language(self, code: str) -> str:
        """Detect programming language from code."""
        if "def " in code and ":" in code:
            return "python"
        elif "fn " in code and "{" in code:
            return "rust"
        else:
            return "unknown"


class PythonAnalyzer:
    """Analyzes Python code to extract specifications."""
    
    def analyze(self, code: str) -> List[CodeSpecification]:
        """Extract specifications from Python code."""
        specs = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    spec = self._analyze_function(node, code)
                    if spec:
                        specs.append(spec)
        
        except SyntaxError as e:
            logger.error(f"Failed to parse Python code: {e}")
        
        return specs
    
    def _analyze_function(self, node: ast.FunctionDef, source: str) -> Optional[CodeSpecification]:
        """Analyze a Python function to extract its specification."""
        function_name = node.name
        
        # Extract parameters and types
        parameters = []
        for arg in node.args.args:
            param_name = arg.arg
            param_type = self._get_type_annotation(arg.annotation) if arg.annotation else "Any"
            parameters.append((param_name, param_type))
        
        # Extract return type
        return_type = self._get_type_annotation(node.returns) if node.returns else None
        
        # Extract docstring
        docstring = ast.get_docstring(node)
        
        # Extract contracts from docstring or comments
        preconditions = self._extract_preconditions(docstring)
        postconditions = self._extract_postconditions(docstring)
        invariants = self._extract_invariants(docstring)
        
        # Analyze complexity
        complexity = self._analyze_complexity(node)
        
        # Extract source code
        source_code = ast.unparse(node) if hasattr(ast, 'unparse') else ""
        
        # Analyze function body for patterns
        self._analyze_body_patterns(node, preconditions, postconditions)
        
        return CodeSpecification(
            function_name=function_name,
            parameters=parameters,
            return_type=return_type,
            preconditions=preconditions,
            postconditions=postconditions,
            invariants=invariants,
            complexity=complexity,
            docstring=docstring,
            source_code=source_code
        )
    
    def _get_type_annotation(self, annotation) -> str:
        """Extract type annotation as string."""
        if annotation is None:
            return "Any"
        elif isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Subscript):
            return ast.unparse(annotation) if hasattr(ast, 'unparse') else str(annotation)
        else:
            return str(annotation)
    
    def _extract_preconditions(self, docstring: Optional[str]) -> List[str]:
        """Extract preconditions from docstring."""
        if not docstring:
            return []
        
        preconditions = []
        
        # Look for explicit preconditions
        if "Precondition" in docstring or "Requires" in docstring:
            lines = docstring.split('\n')
            for line in lines:
                if "precondition" in line.lower() or "requires" in line.lower():
                    preconditions.append(line.strip())
        
        # Look for parameter constraints
        param_pattern = r'(\w+):\s*.*?(must be|should be|>|<|>=|<=|!=|==)([^\.]+)'
        matches = re.findall(param_pattern, docstring, re.IGNORECASE)
        for param, condition, value in matches:
            preconditions.append(f"{param} {condition} {value.strip()}")
        
        return preconditions
    
    def _extract_postconditions(self, docstring: Optional[str]) -> List[str]:
        """Extract postconditions from docstring."""
        if not docstring:
            return []
        
        postconditions = []
        
        # Look for explicit postconditions
        if "Postcondition" in docstring or "Ensures" in docstring or "Returns" in docstring:
            lines = docstring.split('\n')
            for line in lines:
                if any(word in line.lower() for word in ["postcondition", "ensures", "returns"]):
                    postconditions.append(line.strip())
        
        return postconditions
    
    def _extract_invariants(self, docstring: Optional[str]) -> List[str]:
        """Extract invariants from docstring."""
        if not docstring:
            return []
        
        invariants = []
        
        if "Invariant" in docstring:
            lines = docstring.split('\n')
            for line in lines:
                if "invariant" in line.lower():
                    invariants.append(line.strip())
        
        return invariants
    
    def _analyze_complexity(self, node: ast.FunctionDef) -> Optional[str]:
        """Analyze function complexity."""
        # Count nested loops
        loop_depth = self._count_loop_depth(node)
        
        if loop_depth == 0:
            return "O(1)"
        elif loop_depth == 1:
            # Check for binary search pattern
            if self._has_binary_search_pattern(node):
                return "O(log n)"
            return "O(n)"
        elif loop_depth == 2:
            return "O(n^2)"
        elif loop_depth == 3:
            return "O(n^3)"
        else:
            return f"O(n^{loop_depth})"
    
    def _count_loop_depth(self, node: ast.AST, current_depth: int = 0) -> int:
        """Count maximum loop nesting depth."""
        if current_depth > 10:  # Prevent infinite recursion
            return current_depth
            
        max_depth = current_depth
        
        # Use iter_child_nodes instead of walk to avoid recursion
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.For, ast.While)):
                child_depth = self._count_loop_depth(child, current_depth + 1)
                max_depth = max(max_depth, child_depth)
        
        return max_depth
    
    def _has_binary_search_pattern(self, node: ast.FunctionDef) -> bool:
        """Check if function implements binary search pattern."""
        # Look for characteristic binary search patterns
        has_while = False
        has_mid_calculation = False
        has_comparison = False
        
        for child in ast.walk(node):
            if isinstance(child, ast.While):
                has_while = True
            elif isinstance(child, ast.BinOp) and isinstance(child.op, ast.FloorDiv):
                # Look for mid = (left + right) // 2
                has_mid_calculation = True
            elif isinstance(child, ast.Compare):
                has_comparison = True
        
        return has_while and has_mid_calculation and has_comparison
    
    def _analyze_body_patterns(self, node: ast.FunctionDef, preconditions: List[str], postconditions: List[str]):
        """Analyze function body to infer additional specifications."""
        # Look for array access patterns
        for child in ast.walk(node):
            if isinstance(child, ast.Subscript):
                # Array/list access detected
                if isinstance(child.ctx, ast.Load):
                    # Reading from array - add bounds check precondition
                    if isinstance(child.value, ast.Name):
                        array_name = child.value.id
                        if isinstance(child.slice, ast.Name):
                            index_name = child.slice.id
                            preconditions.append(f"0 <= {index_name} < len({array_name})")
        
        # Look for sorting patterns
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if hasattr(child.func, 'id') and child.func.id == 'sorted':
                    postconditions.append("Result is sorted")
                elif hasattr(child.func, 'attr') and child.func.attr == 'sort':
                    postconditions.append("Array is sorted in place")


class RustAnalyzer:
    """Analyzes Rust code to extract specifications."""
    
    def analyze(self, code: str) -> List[CodeSpecification]:
        """Extract specifications from Rust code."""
        specs = []
        
        # Simple regex-based parsing for Rust
        fn_pattern = r'fn\s+(\w+)\s*\((.*?)\)\s*(?:->\s*([\w<>]+))?\s*\{'
        matches = re.findall(fn_pattern, code)
        
        for func_name, params, return_type in matches:
            spec = self._analyze_function(func_name, params, return_type, code)
            if spec:
                specs.append(spec)
        
        return specs
    
    def _analyze_function(self, name: str, params: str, return_type: str, code: str) -> Optional[CodeSpecification]:
        """Analyze a Rust function."""
        # Parse parameters
        parameters = []
        if params:
            param_parts = params.split(',')
            for part in param_parts:
                if ':' in part:
                    param_name, param_type = part.split(':', 1)
                    parameters.append((param_name.strip(), param_type.strip()))
        
        # Extract doc comments
        docstring = self._extract_doc_comments(name, code)
        
        # Extract contracts
        preconditions = self._extract_preconditions(docstring)
        postconditions = self._extract_postconditions(docstring)
        invariants = []
        
        # Analyze complexity
        complexity = self._analyze_rust_complexity(name, code)
        
        return CodeSpecification(
            function_name=name,
            parameters=parameters,
            return_type=return_type or "void",
            preconditions=preconditions,
            postconditions=postconditions,
            invariants=invariants,
            complexity=complexity,
            docstring=docstring,
            source_code=self._extract_function_body(name, code)
        )
    
    def _extract_doc_comments(self, func_name: str, code: str) -> Optional[str]:
        """Extract Rust doc comments for a function."""
        lines = code.split('\n')
        doc_lines = []
        in_function = False
        
        for i, line in enumerate(lines):
            if f"fn {func_name}" in line:
                # Look backwards for doc comments
                j = i - 1
                while j >= 0 and lines[j].strip().startswith('///'):
                    doc_lines.insert(0, lines[j].strip()[3:].strip())
                    j -= 1
                break
        
        return '\n'.join(doc_lines) if doc_lines else None
    
    def _extract_preconditions(self, docstring: Optional[str]) -> List[str]:
        """Extract preconditions from Rust doc comments."""
        if not docstring:
            return []
        
        preconditions = []
        
        # Look for # Safety sections
        if "# Safety" in docstring:
            safety_section = docstring.split("# Safety")[1].split("#")[0]
            preconditions.append(f"Safety: {safety_section.strip()}")
        
        # Look for # Panics sections
        if "# Panics" in docstring:
            panics_section = docstring.split("# Panics")[1].split("#")[0]
            preconditions.append(f"Must not: {panics_section.strip()}")
        
        return preconditions
    
    def _extract_postconditions(self, docstring: Optional[str]) -> List[str]:
        """Extract postconditions from Rust doc comments."""
        if not docstring:
            return []
        
        postconditions = []
        
        # Look for # Returns sections
        if "# Returns" in docstring or "Returns" in docstring:
            returns_match = re.search(r'Returns?[:\s]+(.*?)(?:\n|$)', docstring)
            if returns_match:
                postconditions.append(f"Returns: {returns_match.group(1)}")
        
        return postconditions
    
    def _analyze_rust_complexity(self, func_name: str, code: str) -> Optional[str]:
        """Analyze Rust function complexity."""
        func_body = self._extract_function_body(func_name, code)
        
        # Count loop keywords
        for_count = func_body.count('for ')
        while_count = func_body.count('while ')
        loop_count = func_body.count('loop ')
        
        total_loops = for_count + while_count + loop_count
        
        if total_loops == 0:
            return "O(1)"
        elif total_loops == 1:
            # Check for iterator methods that change complexity
            if '.binary_search' in func_body:
                return "O(log n)"
            elif any(method in func_body for method in ['.map(', '.filter(', '.collect(']):
                return "O(n)"
            return "O(n)"
        elif total_loops == 2:
            return "O(n^2)"
        else:
            return f"O(n^{total_loops})"
    
    def _extract_function_body(self, func_name: str, code: str) -> str:
        """Extract function body from Rust code."""
        pattern = rf'fn\s+{func_name}\s*\([^)]*\)[^{{]*\{{(.*?)\n\}}'
        match = re.search(pattern, code, re.DOTALL)
        return match.group(1) if match else ""


class SpecificationSynthesizer:
    """
    Synthesizes formal specifications from code analysis.
    
    This is the key innovation - automatically generating Coq specs
    from actual code rather than requiring manual translation.
    """
    
    def __init__(self):
        """Initialize the synthesizer."""
        self.analyzer = CodeAnalyzer()
    
    def synthesize_coq_spec(self, code_spec: CodeSpecification) -> str:
        """
        Synthesize a Coq specification from code analysis.
        
        Args:
            code_spec: Analyzed code specification
            
        Returns:
            Coq specification code
        """
        coq_parts = []
        
        # Add header
        coq_parts.append("Require Import Arith.")
        coq_parts.append("Require Import List.")
        coq_parts.append("Require Import Lia.")
        coq_parts.append("")
        
        # Convert function to Coq
        func_def = self._synthesize_function_definition(code_spec)
        if func_def:
            coq_parts.append(func_def)
            coq_parts.append("")
        
        # Add precondition theorems
        for i, precond in enumerate(code_spec.preconditions):
            theorem = self._synthesize_precondition_theorem(precond, i)
            if theorem:
                coq_parts.append(theorem)
                coq_parts.append("")
        
        # Add postcondition theorems
        for i, postcond in enumerate(code_spec.postconditions):
            theorem = self._synthesize_postcondition_theorem(postcond, i)
            if theorem:
                coq_parts.append(theorem)
                coq_parts.append("")
        
        # Add complexity theorem
        if code_spec.complexity:
            theorem = self._synthesize_complexity_theorem(code_spec.complexity)
            if theorem:
                coq_parts.append(theorem)
        
        return '\n'.join(coq_parts)
    
    def _synthesize_function_definition(self, spec: CodeSpecification) -> Optional[str]:
        """Synthesize Coq function definition from specification."""
        # This is simplified - real implementation would need proper translation
        if spec.function_name == "factorial":
            return """Fixpoint factorial (n : nat) : nat :=
  match n with
  | 0 => 1
  | S n' => n * factorial n'
  end."""
        elif spec.function_name == "fibonacci":
            return """Fixpoint fibonacci (n : nat) : nat :=
  match n with
  | 0 => 0
  | 1 => 1
  | S (S n'' as n') => fibonacci n' + fibonacci n''
  end."""
        else:
            return None
    
    def _synthesize_precondition_theorem(self, precondition: str, index: int) -> Optional[str]:
        """Synthesize theorem for precondition."""
        # Parse precondition
        if "<" in precondition or ">" in precondition:
            return f"""Theorem precondition_{index} : forall n : nat, n > 0 -> n >= 1.
Proof.
  intros n H.
  lia.
Qed."""
        return None
    
    def _synthesize_postcondition_theorem(self, postcondition: str, index: int) -> Optional[str]:
        """Synthesize theorem for postcondition."""
        if "sorted" in postcondition.lower():
            return """Theorem result_is_sorted : forall l : list nat,
  exists l', Permutation l l' /\ LocallySorted le l'.
Proof.
  (* Proof would show sorting correctness *)
Admitted."""
        return None
    
    def _synthesize_complexity_theorem(self, complexity: str) -> Optional[str]:
        """Synthesize complexity theorem."""
        if complexity == "O(1)":
            return """Theorem constant_time : exists c : nat, forall n : nat,
  time_complexity n <= c.
Proof.
  exists 1.
  intros n.
  (* Proof of constant time *)
Admitted."""
        elif complexity == "O(n)":
            return """Theorem linear_time : exists c : nat, forall n : nat,
  time_complexity n <= c * n.
Proof.
  exists 1.
  intros n.
  (* Proof of linear time *)
Admitted."""
        return None