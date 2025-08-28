"""Specification inference from tests and code patterns."""

import ast
import re
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TestExample:
    """A test example with input and expected output."""
    function_name: str
    inputs: List[Any]
    expected_output: Any
    test_name: Optional[str] = None
    passed: bool = True


@dataclass
class InferredSpecification:
    """A specification inferred from tests."""
    function_name: str
    preconditions: Set[str] = field(default_factory=set)
    postconditions: Set[str] = field(default_factory=set)
    invariants: Set[str] = field(default_factory=set)
    examples: List[TestExample] = field(default_factory=list)
    patterns: Set[str] = field(default_factory=set)
    confidence: float = 0.0


class TestAnalyzer:
    """
    Analyzes test cases to extract specifications.
    
    This is key for converting existing test suites into formal specs,
    allowing gradual transition from testing to proving.
    """
    
    def __init__(self):
        """Initialize the test analyzer."""
        self.specifications: Dict[str, InferredSpecification] = {}
    
    def analyze_test_file(self, test_code: str) -> List[InferredSpecification]:
        """
        Analyze a test file to extract specifications.
        
        Args:
            test_code: Python test code (pytest or unittest style)
            
        Returns:
            List of inferred specifications
        """
        try:
            tree = ast.parse(test_code)
            
            # Find all test functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                    self._analyze_test_function(node)
            
            return list(self.specifications.values())
        
        except SyntaxError as e:
            logger.error(f"Failed to parse test code: {e}")
            return []
    
    def _analyze_test_function(self, node: ast.FunctionDef):
        """Analyze a single test function."""
        test_name = node.name
        
        # Extract assertions and function calls
        for child in ast.walk(node):
            if isinstance(child, ast.Assert):
                self._analyze_assertion(child, test_name)
            elif isinstance(child, ast.Call):
                if hasattr(child.func, 'attr') and child.func.attr == 'assertEqual':
                    self._analyze_assertEqual(child, test_name)
    
    def _analyze_assertion(self, assert_node: ast.Assert, test_name: str):
        """Analyze an assertion to extract specifications."""
        test_expr = assert_node.test
        
        if isinstance(test_expr, ast.Compare):
            # Extract comparison assertions
            left = self._extract_value(test_expr.left)
            comparators = [self._extract_value(c) for c in test_expr.comparators]
            ops = [type(op).__name__ for op in test_expr.ops]
            
            # Identify function calls
            if isinstance(test_expr.left, ast.Call):
                func_name = self._extract_function_name(test_expr.left)
                inputs = self._extract_arguments(test_expr.left)
                expected = comparators[0] if comparators else None
                
                if func_name:
                    self._add_example(func_name, inputs, expected, test_name)
    
    def _analyze_assertEqual(self, call_node: ast.Call, test_name: str):
        """Analyze assertEqual assertions."""
        if len(call_node.args) >= 2:
            actual = call_node.args[0]
            expected = call_node.args[1]
            
            # Check if actual is a function call
            if isinstance(actual, ast.Call):
                func_name = self._extract_function_name(actual)
                inputs = self._extract_arguments(actual)
                expected_value = self._extract_value(expected)
                
                if func_name:
                    self._add_example(func_name, inputs, expected_value, test_name)
    
    def _add_example(self, func_name: str, inputs: List[Any], expected: Any, test_name: str):
        """Add a test example to the specification."""
        if func_name not in self.specifications:
            self.specifications[func_name] = InferredSpecification(function_name=func_name)
        
        spec = self.specifications[func_name]
        
        # Add the example
        example = TestExample(
            function_name=func_name,
            inputs=inputs,
            expected_output=expected,
            test_name=test_name
        )
        spec.examples.append(example)
        
        # Infer patterns from examples
        self._infer_patterns(spec)
    
    def _infer_patterns(self, spec: InferredSpecification):
        """Infer patterns from test examples."""
        if not spec.examples:
            return
        
        # Analyze input/output relationships
        for example in spec.examples:
            # Check for identity function
            if len(example.inputs) == 1 and example.inputs[0] == example.expected_output:
                spec.patterns.add("identity")
            
            # Check for constant function
            if all(e.expected_output == spec.examples[0].expected_output for e in spec.examples):
                spec.patterns.add("constant")
            
            # Check for arithmetic patterns
            if len(example.inputs) == 2 and isinstance(example.expected_output, (int, float)):
                a, b = example.inputs[0], example.inputs[1]
                result = example.expected_output
                
                if result == a + b:
                    spec.patterns.add("addition")
                elif result == a * b:
                    spec.patterns.add("multiplication")
                elif result == a - b:
                    spec.patterns.add("subtraction")
                elif b != 0 and result == a / b:
                    spec.patterns.add("division")
        
        # Infer preconditions
        self._infer_preconditions(spec)
        
        # Infer postconditions
        self._infer_postconditions(spec)
    
    def _infer_preconditions(self, spec: InferredSpecification):
        """Infer preconditions from test examples."""
        # Check for non-negative inputs
        if all(all(isinstance(inp, (int, float)) and inp >= 0 for inp in ex.inputs) 
               for ex in spec.examples if ex.inputs):
            spec.preconditions.add("inputs >= 0")
        
        # Check for non-zero divisor
        if "division" in spec.patterns:
            spec.preconditions.add("divisor != 0")
        
        # Check for sorted input
        if spec.function_name in ["binary_search", "bisect"]:
            spec.preconditions.add("input array is sorted")
    
    def _infer_postconditions(self, spec: InferredSpecification):
        """Infer postconditions from test examples."""
        # Check if output is always positive
        if all(isinstance(ex.expected_output, (int, float)) and ex.expected_output > 0 
               for ex in spec.examples):
            spec.postconditions.add("result > 0")
        
        # Check if output is bounded
        if spec.examples:
            outputs = [ex.expected_output for ex in spec.examples if isinstance(ex.expected_output, (int, float))]
            if outputs:
                min_out, max_out = min(outputs), max(outputs)
                if min_out == max_out:
                    spec.postconditions.add(f"result = {min_out}")
                else:
                    spec.postconditions.add(f"result in [{min_out}, {max_out}]")
    
    def _extract_function_name(self, call_node: ast.Call) -> Optional[str]:
        """Extract function name from a call node."""
        if isinstance(call_node.func, ast.Name):
            return call_node.func.id
        elif isinstance(call_node.func, ast.Attribute):
            return call_node.func.attr
        return None
    
    def _extract_arguments(self, call_node: ast.Call) -> List[Any]:
        """Extract arguments from a call node."""
        args = []
        for arg in call_node.args:
            args.append(self._extract_value(arg))
        return args
    
    def _extract_value(self, node: ast.AST) -> Any:
        """Extract a value from an AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.List):
            return [self._extract_value(elt) for elt in node.elts]
        elif isinstance(node, ast.Name):
            return node.id
        else:
            return str(ast.unparse(node)) if hasattr(ast, 'unparse') else str(node)


class PropertyInferencer:
    """
    Infers properties from code execution and patterns.
    
    This uses dynamic analysis and pattern matching to discover
    properties that might not be explicitly tested.
    """
    
    def __init__(self):
        """Initialize the property inferencer."""
        self.discovered_properties: Dict[str, Set[str]] = defaultdict(set)
    
    def infer_from_execution_trace(self, func_name: str, traces: List[Dict[str, Any]]) -> Set[str]:
        """
        Infer properties from execution traces.
        
        Args:
            func_name: Name of the function
            traces: List of execution traces with inputs/outputs
            
        Returns:
            Set of inferred properties
        """
        properties = set()
        
        # Analyze traces for patterns
        for trace in traces:
            inputs = trace.get('inputs', [])
            output = trace.get('output')
            
            # Check for determinism
            if self._is_deterministic(func_name, traces):
                properties.add("deterministic")
            
            # Check for monotonicity
            if self._is_monotonic(traces):
                properties.add("monotonic")
            
            # Check for idempotence
            if self._is_idempotent(func_name, traces):
                properties.add("idempotent")
            
            # Check for commutativity
            if self._is_commutative(traces):
                properties.add("commutative")
            
            # Check for associativity
            if self._is_associative(traces):
                properties.add("associative")
        
        self.discovered_properties[func_name].update(properties)
        return properties
    
    def _is_deterministic(self, func_name: str, traces: List[Dict[str, Any]]) -> bool:
        """Check if function is deterministic."""
        input_output_map = {}
        
        for trace in traces:
            inputs = tuple(trace.get('inputs', []))
            output = trace.get('output')
            
            if inputs in input_output_map:
                if input_output_map[inputs] != output:
                    return False
            else:
                input_output_map[inputs] = output
        
        return True
    
    def _is_monotonic(self, traces: List[Dict[str, Any]]) -> bool:
        """Check if function is monotonic."""
        # For single-input numeric functions
        numeric_traces = [(t['inputs'][0], t['output']) 
                         for t in traces 
                         if len(t.get('inputs', [])) == 1 
                         and isinstance(t['inputs'][0], (int, float))
                         and isinstance(t['output'], (int, float))]
        
        if len(numeric_traces) < 2:
            return False
        
        # Check if increasing inputs lead to increasing outputs
        sorted_traces = sorted(numeric_traces, key=lambda x: x[0])
        return all(sorted_traces[i][1] <= sorted_traces[i+1][1] 
                  for i in range(len(sorted_traces)-1))
    
    def _is_idempotent(self, func_name: str, traces: List[Dict[str, Any]]) -> bool:
        """Check if function is idempotent (f(f(x)) = f(x))."""
        # This requires special testing - marking as future work
        return False
    
    def _is_commutative(self, traces: List[Dict[str, Any]]) -> bool:
        """Check if function is commutative."""
        # For two-input functions
        for trace in traces:
            inputs = trace.get('inputs', [])
            if len(inputs) == 2:
                # Look for trace with swapped inputs
                swapped = self._find_trace_with_inputs(traces, [inputs[1], inputs[0]])
                if swapped and swapped['output'] != trace['output']:
                    return False
        return True
    
    def _is_associative(self, traces: List[Dict[str, Any]]) -> bool:
        """Check if function is associative."""
        # This requires specific test patterns - marking as future work
        return False
    
    def _find_trace_with_inputs(self, traces: List[Dict[str, Any]], inputs: List[Any]) -> Optional[Dict[str, Any]]:
        """Find a trace with specific inputs."""
        for trace in traces:
            if trace.get('inputs') == inputs:
                return trace
        return None


class SpecificationSynthesizer:
    """
    Synthesizes formal specifications from inferred properties.
    
    This converts the inferred specifications into Coq theorems.
    """
    
    def __init__(self):
        """Initialize the synthesizer."""
        self.test_analyzer = TestAnalyzer()
        self.property_inferencer = PropertyInferencer()
    
    def synthesize_from_tests(self, test_code: str) -> List[str]:
        """
        Synthesize formal specifications from test code.
        
        Args:
            test_code: Python test code
            
        Returns:
            List of Coq specifications
        """
        # Analyze tests
        specs = self.test_analyzer.analyze_test_file(test_code)
        
        coq_specs = []
        for spec in specs:
            coq_spec = self._synthesize_coq_from_spec(spec)
            if coq_spec:
                coq_specs.append(coq_spec)
        
        return coq_specs
    
    def _synthesize_coq_from_spec(self, spec: InferredSpecification) -> Optional[str]:
        """Synthesize Coq specification from inferred spec."""
        if not spec.examples:
            return None
        
        coq_lines = []
        coq_lines.append("Require Import Arith.")
        coq_lines.append("Require Import List.")
        coq_lines.append("")
        
        # Generate function definition based on examples
        func_def = self._generate_function_definition(spec)
        if func_def:
            coq_lines.append(func_def)
            coq_lines.append("")
        
        # Generate theorems for each example
        for i, example in enumerate(spec.examples[:5]):  # Limit to first 5
            theorem = self._generate_example_theorem(spec.function_name, example, i)
            if theorem:
                coq_lines.append(theorem)
                coq_lines.append("")
        
        # Generate property theorems
        for prop in spec.patterns:
            theorem = self._generate_property_theorem(spec.function_name, prop)
            if theorem:
                coq_lines.append(theorem)
                coq_lines.append("")
        
        return '\n'.join(coq_lines)
    
    def _generate_function_definition(self, spec: InferredSpecification) -> Optional[str]:
        """Generate Coq function definition from examples."""
        if "addition" in spec.patterns:
            return f"Definition {spec.function_name} (a b : nat) : nat := a + b."
        elif "multiplication" in spec.patterns:
            return f"Definition {spec.function_name} (a b : nat) : nat := a * b."
        elif "subtraction" in spec.patterns:
            return f"Definition {spec.function_name} (a b : nat) : nat := a - b."
        else:
            return None
    
    def _generate_example_theorem(self, func_name: str, example: TestExample, index: int) -> Optional[str]:
        """Generate theorem for a specific example."""
        if len(example.inputs) == 2 and isinstance(example.expected_output, int):
            a, b = example.inputs
            result = example.expected_output
            
            return f"""Theorem {func_name}_example_{index} : {func_name} {a} {b} = {result}.
Proof.
  unfold {func_name}.
  reflexivity.
Qed."""
        
        return None
    
    def _generate_property_theorem(self, func_name: str, property: str) -> Optional[str]:
        """Generate theorem for a property."""
        if property == "commutative":
            return f"""Theorem {func_name}_commutative : forall a b : nat, {func_name} a b = {func_name} b a.
Proof.
  intros a b.
  unfold {func_name}.
  ring.
Qed."""
        elif property == "associative":
            return f"""Theorem {func_name}_associative : forall a b c : nat, 
  {func_name} ({func_name} a b) c = {func_name} a ({func_name} b c).
Proof.
  intros a b c.
  unfold {func_name}.
  ring.
Qed."""
        
        return None