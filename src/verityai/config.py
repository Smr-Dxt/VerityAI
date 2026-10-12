"""
VerityAI - Core Configuration Module
Phase 1: Project Foundation

This module centralizes application configuration using Python's standard
library dataclasses and environment variable loading. It provides safe defaults
tailored for offline-first research workflows.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, Any


@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration dataclass.

    Attributes:
        app_name: Name of the application.
        version: Semantic version of the current release.
        phase: Project lifecycle phase description.
        tagline: Brief mission statement / description.
        environment: Execution environment (development, testing, production).
        host: Host address for the web interface.
        port: Port number for the web interface.
        offline_mode: Boolean indicating if external network requests are disabled.
        log_level: Standard logging level (DEBUG, INFO, WARNING, ERROR).
        project_root: Absolute path to the repository root directory.
        data_dir: Path to the local document storage directory.
        models_dir: Path to the local models directory.
    """

    app_name: str = "VerityAI"
    version: str = "0.1.0"
    phase: str = "Phase 1: Project Foundation"
    tagline: str = "Research with evidence, not guesses."
    environment: str = "development"
    host: str = "localhost"
    port: int = 8501
    offline_mode: bool = True
    log_level: str = "INFO"
    project_root: Path = Path(__file__).resolve().parent.parent.parent
    data_dir: Path = Path(__file__).resolve().parent.parent.parent / "data"
    models_dir: Path = Path(__file__).resolve().parent.parent.parent / "models"

    def to_display_dict(self) -> Dict[str, Any]:
        """Returns a sanitized dictionary of settings suitable for UI display."""
        return {
            "Application": self.app_name,
            "Version": self.version,
            "Phase": self.phase,
            "Environment": self.environment,
            "Offline Mode": "Enabled" if self.offline_mode else "Disabled",
            "Server Address": f"{self.host}:{self.port}",
            "Log Level": self.log_level,
        }


def _parse_bool(value: str, default: bool = True) -> bool:
    """Helper to safely parse boolean values from environment strings."""
    if not value:
        return default
    return value.strip().lower() in ("true", "1", "yes", "t", "y", "on")


def get_config() -> AppConfig:
    """Loads configuration values from environment variables or returns defaults.

    Returns:
        AppConfig: An initialized AppConfig instance.
    """
    project_root = Path(__file__).resolve().parent.parent.parent

    # Read values safely from environment with fallback defaults
    app_name = os.getenv("VERITY_APP_NAME", "VerityAI")
    environment = os.getenv("VERITY_ENV", "development")
    host = os.getenv("VERITY_HOST", "localhost")

    # Safe integer parsing for port
    raw_port = os.getenv("VERITY_PORT", "8501")
    try:
        port = int(raw_port)
    except ValueError:
        port = 8501

    raw_offline = os.getenv("VERITY_OFFLINE_MODE", "true")
    offline_mode = _parse_bool(raw_offline, default=True)

    log_level = os.getenv("VERITY_LOG_LEVEL", "INFO").upper()

    return AppConfig(
        app_name=app_name,
        version="0.1.0",
        phase="Phase 1: Project Foundation",
        tagline="Research with evidence, not guesses.",
        environment=environment,
        host=host,
        port=port,
        offline_mode=offline_mode,
        log_level=log_level,
        project_root=project_root,
        data_dir=project_root / "data",
        models_dir=project_root / "models",
    )
