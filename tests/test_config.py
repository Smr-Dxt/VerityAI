"""
Tests for VerityAI Configuration Module
Phase 1: Project Foundation

These unit tests verify default configuration values, environment variable
handling, and immutability without requiring any external services or models.
"""

import os
import sys
from pathlib import Path
import pytest
from dataclasses import FrozenInstanceError

# Ensure src/ is accessible on sys.path regardless of how pytest is invoked
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from verityai.config import AppConfig, get_config, _parse_bool


def test_default_config():
    """Verify default configuration values."""
    config = get_config()
    assert config.app_name == "VerityAI"
    assert config.version == "0.1.0"
    assert config.phase == "Phase 1: Project Foundation"
    assert config.host == "localhost"
    assert config.port == 8501
    assert config.offline_mode is True
    assert config.log_level == "INFO"
    assert config.project_root.exists()


def test_config_immutability():
    """Verify that AppConfig instances cannot be modified at runtime."""
    config = get_config()
    with pytest.raises(FrozenInstanceError):
        # Trying to change an attribute on a frozen dataclass must raise an error
        config.port = 9000  # type: ignore


def test_config_environment_overrides(monkeypatch):
    """Verify that environment variables safely override defaults."""
    monkeypatch.setenv("VERITY_APP_NAME", "CustomVerity")
    monkeypatch.setenv("VERITY_PORT", "8888")
    monkeypatch.setenv("VERITY_OFFLINE_MODE", "false")
    monkeypatch.setenv("VERITY_ENV", "production")

    config = get_config()
    assert config.app_name == "CustomVerity"
    assert config.port == 8888
    assert config.offline_mode is False
    assert config.environment == "production"


def test_config_invalid_port_fallback(monkeypatch):
    """Verify fallback when an invalid port is provided in the environment."""
    monkeypatch.setenv("VERITY_PORT", "not-a-number")
    config = get_config()
    assert config.port == 8501


def test_parse_bool():
    """Verify boolean parser helper function."""
    assert _parse_bool("true") is True
    assert _parse_bool("True") is True
    assert _parse_bool("1") is True
    assert _parse_bool("yes") is True
    assert _parse_bool("false") is False
    assert _parse_bool("0") is False
    assert _parse_bool("no") is False
    assert _parse_bool("", default=True) is True
    assert _parse_bool("", default=False) is False


def test_display_dict():
    """Verify that display dictionary outputs expected diagnostic keys."""
    config = get_config()
    display = config.to_display_dict()
    assert "Application" in display
    assert "Version" in display
    assert "Phase" in display
    assert "Offline Mode" in display
    assert display["Application"] == "VerityAI"
    assert display["Offline Mode"] == "Enabled"
