# Mathematical Certainty in Multi-Agent Belief Conflicts: A Formal Verification Approach to Cognitive Dissonance Resolution

[![Requirements: Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![DSPy](https://img.shields.io/badge/DSPy-Compatible-green.svg)](https://github.com/stanfordnlp/dspy)
[![Coq](https://img.shields.io/badge/Coq-8.18+-orange.svg)](https://coq.inria.fr/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A framework for detecting and resolving cognitive dissonance in multi-agent systems through **formal verification**. This approach combines DSPy-based cognitive dissonance detection with automated theorem proving to provide **mathematically rigorous conflict resolution**.

## 🔬 Research Motivation

Traditional cognitive dissonance detection relies on heuristics and probabilistic approaches. This framework introduces **mathematical certainty** to conflict resolution by:

1. **Translating** natural language claims to formal specifications
2. **Detecting** conflicts between agent beliefs  
3. **Resolving** disputes using automated theorem proving (Coq)
4. **Establishing** ground truth through mathematical proof

### Novel Contribution

This framework uniquely combines:
- **DSPy-based cognitive dissonance detection** with **formal verification methods**
- **Natural language claim translation** to **mathematical specifications**  
- **Real-time theorem proving** for **definitive conflict resolution**

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Coq theorem prover** (`coqc` command available)
- Ollama or compatible API endpoint (for agent reasoning)

### Installation

```bash
# Clone the repository
git clone https://github.com/evalops/cognitive-dissonance-dspy.git
cd cognitive-dissonance-dspy

# Set up virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Install Coq (Ubuntu/Debian)
sudo apt update && sudo apt install -y coq

# Install Coq (macOS)
brew install coq

# Verify installation
coqc --version
```

### Basic Usage

```python
from formal_verification import FormalVerificationConflictDetector, Claim, PropertyType
import time

# Create detector
detector = FormalVerificationConflictDetector()

# Define conflicting claims
claims = [
    Claim("alice", "2 + 2 = 4", PropertyType.CORRECTNESS, 0.95, time.time()),
    Claim("bob", "2 + 2 = 5", PropertyType.CORRECTNESS, 0.80, time.time()),
]

# Analyze with formal verification
results = detector.analyze_claims(claims)

# Alice's claim: ✅ MATHEMATICALLY PROVEN  
# Bob's claim:   ❌ MATHEMATICALLY DISPROVEN
# Resolution:    Alice is correct (formal proof)
```

### Run Examples

```bash
# Mathematical claims demonstration
python examples/mathematical_claims.py

# Advanced theorem proving
python examples/advanced_theorems.py

# Comprehensive framework demo
python examples/comprehensive_demo.py
```

## 📊 Capabilities Demonstrated

Our framework has been validated with **complex mathematical theorem proving**:

### Mathematical Verification Results
- ✅ **15 mathematical claims analyzed** with **80% success rate**
- ✅ **Complex theorems proven**: `factorial 7 = 5040` in 502.9ms
- ✅ **Conflicts resolved**: Mathematical contradictions detected and resolved
- ✅ **Error precision**: "Unable to unify '25' with '24'" (factorial 4 ≠ 25)
- ✅ **Average proof time**: 179.7ms per theorem

### Supported Property Types

- **Mathematical Correctness**: Arithmetic, factorial computations
- **Algorithm Correctness**: Sorting, searching, computation properties  
- **Time Complexity**: Big-O complexity claims
- **Memory Safety**: Buffer overflow, use-after-free properties

## 🛠️ Architecture

### Core Components

```
formal_verification/
├── detector.py          # Main conflict detection logic
├── translator.py        # Natural language → Coq translation  
├── prover.py            # Coq theorem prover interface
├── types.py             # Type definitions
└── __init__.py          # Module exports

cognitive_dissonance/    # DSPy-based agents
├── verifier.py          # Belief extraction and reconciliation
├── config.py            # Configuration management
├── metrics.py           # Evaluation metrics
└── experiment.py        # Co-training framework

examples/
├── mathematical_claims.py       # Basic mathematical verification
├── advanced_theorems.py         # Complex factorial proofs  
├── comprehensive_demo.py        # Complete capabilities showcase
└── algorithm_correctness.py     # Algorithm verification patterns
```

### Translation Patterns

The system recognizes and formalizes common claim patterns:

```python
# Mathematical Claims
"2 + 2 = 4" → "Theorem arithmetic_claim : 2 + 2 = 4. Proof. reflexivity. Qed."
"factorial 5 = 120" → Coq factorial definition with correctness proof

# Algorithm Properties  
"quicksort correctly sorts" → Permutation + LocallySorted specification
"binary search finds element" → Sorted list correctness theorem

# Memory Safety
"function is memory safe" → Buffer overflow prevention specification
```

## 📈 Experimental Results

### Formal Verification Performance
```
🔬 COMPREHENSIVE FORMAL VERIFICATION ANALYSIS
======================================================================
Framework: Formal Verification + Cognitive Dissonance Detection
Theorem Prover: Coq (coqc)
Total Claims Analyzed: 15
Conflicts Detected: 2

Mathematical theorems proven:      12
Mathematical claims disproven:      3
Formal conflicts resolved:          2
Average theorem proof time:        179.7ms
Success rate:                      80.0%
Maximum complexity:                factorial 7 = 5040
```

### Agent Accuracy Rankings
```
🏆 AGENT MATHEMATICAL ACCURACY RANKINGS:
🥇 alice, charlie, diana, eve, grace, henry: 100.0% (Perfect)
🥉 bob, frank, maya:                          0.0% (Failed)
```

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `ollama_chat/llama3.1:8b` | LLM model for agent reasoning |
| `API_BASE` | `http://localhost:11434` | API endpoint |
| `COQC_TIMEOUT` | `10` | Theorem proving timeout (seconds) |

### CLI Usage

```bash
# DSPy-based cognitive dissonance (probabilistic)
python -m cognitive_dissonance.main demo

# Formal verification (mathematical certainty)
python examples/mathematical_claims.py

# Advanced theorem proving
python examples/comprehensive_demo.py
```

## 🔍 Research Validation

### Literature Review (2024-2025)

**Multi-Agent Formal Verification**
- IJCAI 2024: Neural-symbolic multi-agent systems verification
- Recent advances in BDI agent verification using modal logic
- Automated reasoning for belief revision systems

**Cognitive Dissonance & Belief Revision**
- 2024 research on automated cognitive dissonance resolution using LLMs
- Belief revision adaptability in large language models
- Possibilistic logic approaches for uncertain reasoning

**Novel Intersection**: Our framework represents the **first combination** of:
1. Formal verification methods
2. Cognitive dissonance detection  
3. Multi-agent belief conflict resolution

## 🧪 Testing

```bash
# Run all tests (150 tests)
python -m pytest tests/ -v

# Test formal verification specifically  
python -m pytest tests/test_formal_verification.py -v

# Run working examples
python examples/concrete_proofs.py
```

## 📚 Related Work

This research builds on established areas while creating a novel intersection:

- **Multi-Agent Formal Verification**: BDI systems, modal logic frameworks
- **Automated Theorem Proving**: Coq, Isabelle/HOL, proof assistants
- **Belief Revision Theory**: AGM postulates, belief update mechanisms  
- **Cognitive Dissonance Detection**: Consistency checking, conflict resolution
- **DSPy Framework**: Prompt optimization, agent co-training

**Research Contribution**: Mathematical certainty in multi-agent belief conflicts through automated theorem proving.

## 🎯 Applications

### Software Engineering
- **Code Review**: Resolve disputes about algorithm properties with formal proof
- **Verification**: Mathematical correctness proofs for critical systems
- **Education**: Definitive resolution of programming concept disagreements

### AI Safety Research  
- **Truth-Seeking**: Multi-agent systems with mathematical ground truth
- **Belief Revision**: Principled updating based on formal verification
- **Consensus Formation**: Agreement through mathematical certainty

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏢 Organization

This project is maintained by [EvalOps](https://github.com/evalops), focused on advanced LLM evaluation and AI safety tools.

## 🤝 Contributing

We welcome contributions! Areas of interest:

- **New Translation Patterns**: Support for additional property types
- **Performance Optimization**: Faster proof search strategies  
- **Domain Extensions**: Application to new verification domains
- **Integration**: Connect with existing verification tools

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/evalops/cognitive-dissonance-dspy/issues)  
- **Email**: [info@evalops.dev](mailto:info@evalops.dev)

---

> **🔬 Research Framework**: This demonstrates the intersection of formal methods and cognitive dissonance detection. Results provide mathematical certainty in multi-agent belief conflicts, advancing beyond probabilistic approaches through automated theorem proving.