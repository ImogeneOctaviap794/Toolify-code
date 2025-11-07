# SPDX-License-Identifier: GPL-3.0-or-later
#
# Toolify: Empower any LLM with function calling capabilities.
# Copyright (C) 2025 FunnyCups (https://github.com/funnycups)

"""
Base capability detector.
All provider-specific detectors should inherit from this class.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CapabilityStatus(Enum):
    """Capability support status"""
    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"
    UNKNOWN = "unknown"
    ERROR = "error"


@dataclass
class CapabilityResult:
    """Result of a capability detection"""
    capability_name: str
    status: CapabilityStatus
    details: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "capability_name": self.capability_name,
            "status": self.status.value,
            "details": self.details,
            "error_message": self.error_message
        }


@dataclass
class DetectionReport:
    """Complete detection report for a provider"""
    provider: str
    model: str
    capabilities: List[CapabilityResult] = field(default_factory=list)
    summary: Optional[Dict[str, int]] = None
    
    def __post_init__(self):
        """Calculate summary after initialization"""
        if not self.summary:
            self.summary = {
                "supported": sum(1 for c in self.capabilities if c.status == CapabilityStatus.SUPPORTED),
                "not_supported": sum(1 for c in self.capabilities if c.status == CapabilityStatus.NOT_SUPPORTED),
                "unknown": sum(1 for c in self.capabilities if c.status == CapabilityStatus.UNKNOWN),
                "error": sum(1 for c in self.capabilities if c.status == CapabilityStatus.ERROR),
                "total": len(self.capabilities)
            }
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "provider": self.provider,
            "model": self.model,
            "capabilities": [c.to_dict() for c in self.capabilities],
            "summary": self.summary
        }


class BaseCapabilityDetector(ABC):
    """
    Base class for capability detection.
    Each detector tests specific AI provider capabilities.
    """
    
    def __init__(self, api_key: str, base_url: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize detector.
        
        Args:
            api_key: API key for the provider
            base_url: Optional custom base URL
            model: Optional specific model to test
        """
        self.api_key = api_key
        self.base_url = base_url or self.get_default_base_url()
        self.model = model or self.get_default_model()
        self.timeout = 30
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the provider name"""
        pass
    
    @abstractmethod
    def get_default_base_url(self) -> str:
        """Return the default base URL for this provider"""
        pass
    
    @abstractmethod
    def get_default_model(self) -> str:
        """Return the default model for testing"""
        pass
    
    @abstractmethod
    async def detect_all_capabilities(self) -> DetectionReport:
        """
        Detect all capabilities for this provider.
        
        Returns:
            DetectionReport with all capability results
        """
        pass
    
    @abstractmethod
    async def test_basic_chat(self) -> CapabilityResult:
        """Test basic chat completion"""
        pass
    
    @abstractmethod
    async def test_streaming(self) -> CapabilityResult:
        """Test streaming responses"""
        pass
    
    @abstractmethod
    async def test_function_calling(self) -> CapabilityResult:
        """Test function calling / tool use"""
        pass
    
    @abstractmethod
    async def test_vision(self) -> CapabilityResult:
        """Test vision / image understanding"""
        pass
    
    def _create_success_result(self, capability_name: str, details: Optional[Dict] = None) -> CapabilityResult:
        """Helper to create success result"""
        return CapabilityResult(
            capability_name=capability_name,
            status=CapabilityStatus.SUPPORTED,
            details=details
        )
    
    def _create_failure_result(self, capability_name: str, error: str) -> CapabilityResult:
        """Helper to create failure result"""
        return CapabilityResult(
            capability_name=capability_name,
            status=CapabilityStatus.NOT_SUPPORTED,
            error_message=error
        )
    
    def _create_error_result(self, capability_name: str, error: Exception) -> CapabilityResult:
        """Helper to create error result"""
        return CapabilityResult(
            capability_name=capability_name,
            status=CapabilityStatus.ERROR,
            error_message=str(error)
        )

