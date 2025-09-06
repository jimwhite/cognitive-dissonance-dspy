"""Configuration management for Cognitive Dissonance experiments."""

import os
from dataclasses import dataclass
import logging
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for Cognitive Dissonance experiments."""

    # Model configuration
    model: str = "ollama_chat/llama3.1:8b"
    api_base: str = "http://localhost:11434"
    api_key: str = ""
    temperature: float = 0.5
    max_tokens: int = 512

    # Experiment parameters  
    alpha: float = 0.0  # Truth anchoring weight for belief reconciliation
    rounds: int = 6  # Number of training rounds
    use_cot: bool = False  # Use Chain of Thought
    dissonance_threshold: float = 0.7  # Threshold for detecting cognitive dissonance

    # Optimization settings
    auto_mode: str = "light"
    enable_disk_cache: bool = True
    enable_memory_cache: bool = True
    
    # Checkpoint settings
    checkpoints: Optional[str] = None
    
    # MLFlow settings
    mlflow_tracking_uri: Optional[str] = None  # MLFlow tracking server URI
    mlflow_experiment_name: str = "cognitive-dissonance"  # MLFlow experiment name
    enable_mlflow: bool = False  # Enable MLFlow telemetry

    @classmethod
    def from_env(cls) -> "ExperimentConfig":
        """Create configuration from environment variables."""
        return cls(
            model=os.getenv("MODEL", cls.model),
            api_base=os.getenv("API_BASE", cls.api_base),
            api_key=os.getenv("API_KEY", cls.api_key),
            temperature=float(os.getenv("TEMPERATURE", str(cls.temperature))),
            max_tokens=int(os.getenv("MAX_TOKENS", str(cls.max_tokens))),
            alpha=float(os.getenv("ALPHA", str(cls.alpha))),
            rounds=int(os.getenv("ROUNDS", str(cls.rounds))),
            use_cot=os.getenv("USE_COT", "false").lower() == "true",
            dissonance_threshold=float(os.getenv("DISSONANCE_THRESHOLD", str(cls.dissonance_threshold))),
            auto_mode=os.getenv("AUTO_MODE", cls.auto_mode),
            enable_disk_cache=os.getenv("ENABLE_DISK_CACHE", "false").lower() == "true",
            enable_memory_cache=os.getenv("ENABLE_MEMORY_CACHE", "false").lower() == "true",
            checkpoints=os.getenv("CHECKPOINTS", cls.checkpoints),
            mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", cls.mlflow_tracking_uri),
            mlflow_experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", cls.mlflow_experiment_name),
            enable_mlflow=os.getenv("ENABLE_MLFLOW", "false").lower() == "true",
        )

    def validate(self) -> None:
        """Validate configuration parameters."""
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"Alpha must be between 0.0 and 1.0, got {self.alpha}")

        if not (0.0 <= self.dissonance_threshold <= 1.0):
            raise ValueError(f"Dissonance threshold must be between 0.0 and 1.0, got {self.dissonance_threshold}")

        if self.rounds < 1:
            raise ValueError(f"Rounds must be >= 1, got {self.rounds}")

        if self.temperature < 0:
            raise ValueError(f"Temperature must be >= 0, got {self.temperature}")

        if self.max_tokens < 1:
            raise ValueError(f"Max tokens must be >= 1, got {self.max_tokens}")

        logger.info(f"Configuration validated: {self}")

    def setup_dspy(self) -> None:
        """Configure DSPy with this configuration."""
        import dspy

        lm = dspy.LM(
            self.model,
            api_base=self.api_base,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        dspy.configure_cache(
            enable_disk_cache=self.enable_disk_cache,
            enable_memory_cache=self.enable_memory_cache,
        )

        dspy.configure(lm=lm)
        logger.info(f"DSPy configured with model: {self.model}")

    def setup_mlflow(self) -> None:
        """Configure MLFlow tracking if enabled."""
        if not self.enable_mlflow:
            return
            
        try:
            import mlflow
            import mlflow.dspy
            
            # Set tracking URI if specified
            if self.mlflow_tracking_uri:
                mlflow.set_tracking_uri(self.mlflow_tracking_uri)
                logger.info(f"MLFlow tracking URI set to: {self.mlflow_tracking_uri}")
            
            # Set experiment name
            mlflow.set_experiment(self.mlflow_experiment_name)
            logger.info(f"MLFlow experiment set to: {self.mlflow_experiment_name}")
            
            # Enable DSPy autologging
            mlflow.dspy.autolog()
            logger.info("MLFlow DSPy autologging enabled")
            
        except ImportError as e:
            logger.warning(f"MLFlow not available, skipping telemetry setup: {e}")
        except Exception as e:
            logger.warning(f"Failed to setup MLFlow: {e}")


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,  # Set root level to INFO
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set our application loggers to the requested level
    app_loggers = [
        "cognitive_dissonance",
        "dspy",
    ]

    for logger_name in app_loggers:
        logging.getLogger(logger_name).setLevel(getattr(logging, level.upper()))

    # Reduce noise from external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)