"""Coq theorem prover interface for formal verification."""

import subprocess
import tempfile
import time
import logging
from typing import Optional

from .types import FormalSpec, ProofResult

logger = logging.getLogger(__name__)


class CoqProver:
    """Interface to Coq theorem prover for formal verification."""
    
    def __init__(self, timeout_seconds: int = 30):
        """Initialize Coq prover interface.
        
        Args:
            timeout_seconds: Maximum time to wait for proof completion
        """
        self.timeout_seconds = timeout_seconds
        self.coq_available = self._check_coq_installation()
        
        if not self.coq_available:
            logger.warning("Coq theorem prover not available")
    
    def _check_coq_installation(self) -> bool:
        """Check if Coq is installed and available."""
        try:
            result = subprocess.run(['coqc', '--version'], 
                                  capture_output=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def prove_specification(self, spec: FormalSpec) -> ProofResult:
        """Attempt to prove a formal specification using Coq.
        
        Args:
            spec: The formal specification to prove
            
        Returns:
            ProofResult with success/failure and timing information
        """
        if not self.coq_available:
            return ProofResult(
                spec=spec,
                proven=False,
                proof_time_ms=0,
                error_message="Coq theorem prover not available",
                counter_example=None
            )
        
        start_time = time.time()
        
        # Create temporary file with Coq code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.v', delete=False) as f:
            f.write(spec.coq_code)
            temp_file = f.name
        
        try:
            # Run Coq compiler
            result = subprocess.run([
                'coqc', '-q', temp_file
            ], capture_output=True, timeout=self.timeout_seconds)
            
            proof_time = (time.time() - start_time) * 1000
            
            if result.returncode == 0:
                logger.debug(f"Proof successful for: {spec.spec_text}")
                return ProofResult(
                    spec=spec,
                    proven=True,
                    proof_time_ms=proof_time,
                    error_message=None,
                    counter_example=None
                )
            else:
                error_msg = result.stderr.decode('utf-8') if result.stderr else "Proof failed"
                logger.debug(f"Proof failed for: {spec.spec_text}, error: {error_msg[:100]}")
                
                return ProofResult(
                    spec=spec,
                    proven=False,
                    proof_time_ms=proof_time,
                    error_message=error_msg,
                    counter_example=self._extract_counter_example(error_msg)
                )
        
        except subprocess.TimeoutExpired:
            logger.warning(f"Proof timeout for: {spec.spec_text}")
            return ProofResult(
                spec=spec,
                proven=False,
                proof_time_ms=self.timeout_seconds * 1000,
                error_message="Proof attempt timed out",
                counter_example=None
            )
        
        finally:
            # Clean up temporary file
            try:
                import os
                os.unlink(temp_file)
            except Exception:
                pass
    
    def _extract_counter_example(self, error_msg: str) -> Optional[str]:
        """Extract counter-example from Coq error message if available.
        
        Args:
            error_msg: The error message from Coq
            
        Returns:
            Counter-example string if found, None otherwise
        """
        # Simple heuristic-based counter-example extraction
        if "counter" in error_msg.lower():
            lines = error_msg.split('\n')
            for line in lines:
                if "counter" in line.lower():
                    return line.strip()
        
        return None