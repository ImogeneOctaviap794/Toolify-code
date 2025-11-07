# SPDX-License-Identifier: GPL-3.0-or-later
#
# Toolify: Empower any LLM with function calling capabilities.
# Copyright (C) 2025 FunnyCups (https://github.com/funnycups)

"""
Factory for creating capability detectors.
"""

from typing import Dict, Optional
import logging
from .base_detector import BaseCapabilityDetector

logger = logging.getLogger(__name__)


class DetectorFactory:
    """Factory for creating capability detectors"""
    
    _detector_classes: Dict[str, type] = {}
    
    @classmethod
    def register_detector(cls, provider: str, detector_class: type):
        """
        Register a detector class for a provider.
        
        Args:
            provider: Provider identifier (openai, anthropic, gemini)
            detector_class: Detector class to register
        """
        cls._detector_classes[provider] = detector_class
        logger.info(f"Registered detector for provider: {provider}")
    
    @classmethod
    def create_detector(
        cls,
        provider: str,
        api_key: str,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ) -> Optional[BaseCapabilityDetector]:
        """
        Create a detector instance.
        
        Args:
            provider: Provider identifier
            api_key: API key
            base_url: Optional custom base URL
            model: Optional specific model
            
        Returns:
            Detector instance or None if not found
        """
        detector_class = cls._detector_classes.get(provider)
        if detector_class:
            return detector_class(api_key, base_url, model)
        else:
            logger.error(f"No detector registered for provider: {provider}")
            return None
    
    @classmethod
    def get_supported_providers(cls) -> list:
        """Get list of supported providers"""
        return list(cls._detector_classes.keys())
    
    @classmethod
    def is_provider_supported(cls, provider: str) -> bool:
        """Check if provider is supported"""
        return provider in cls._detector_classes

