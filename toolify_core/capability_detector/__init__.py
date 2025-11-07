# SPDX-License-Identifier: GPL-3.0-or-later
#
# Toolify: Empower any LLM with function calling capabilities.
# Copyright (C) 2025 FunnyCups (https://github.com/funnycups)

"""
Capability detection for AI providers.
Detects support for various features across OpenAI, Anthropic, and Gemini.
"""

from .base_detector import BaseCapabilityDetector, CapabilityResult
from .detector_factory import DetectorFactory
from .openai_detector import OpenAICapabilityDetector
from .anthropic_detector import AnthropicCapabilityDetector
from .gemini_detector import GeminiCapabilityDetector

__all__ = [
    'BaseCapabilityDetector',
    'CapabilityResult',
    'DetectorFactory',
    'OpenAICapabilityDetector',
    'AnthropicCapabilityDetector',
    'GeminiCapabilityDetector',
]

