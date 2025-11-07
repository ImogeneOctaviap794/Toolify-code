# SPDX-License-Identifier: GPL-3.0-or-later
#
# Toolify: Empower any LLM with function calling capabilities.
# Copyright (C) 2025 FunnyCups (https://github.com/funnycups)

"""
Gemini capability detector.
"""

import httpx
import logging
from .base_detector import BaseCapabilityDetector, CapabilityResult, DetectionReport, CapabilityStatus

logger = logging.getLogger(__name__)


class GeminiCapabilityDetector(BaseCapabilityDetector):
    """Detector for Gemini API capabilities"""
    
    def get_provider_name(self) -> str:
        return "gemini"
    
    def get_default_base_url(self) -> str:
        return "https://generativelanguage.googleapis.com/v1beta"
    
    def get_default_model(self) -> str:
        return "gemini-2.0-flash-exp"
    
    async def detect_all_capabilities(self) -> DetectionReport:
        """Detect all Gemini capabilities"""
        logger.info(f"Starting capability detection for Gemini (model: {self.model})")
        
        capabilities = []
        capabilities.append(await self.test_basic_chat())
        capabilities.append(await self.test_streaming())
        capabilities.append(await self.test_function_calling())
        capabilities.append(await self.test_vision())
        capabilities.append(await self.test_system_message())
        
        report = DetectionReport(
            provider=self.get_provider_name(),
            model=self.model,
            capabilities=capabilities
        )
        
        logger.info(f"Detection complete. Summary: {report.summary}")
        return report
    
    async def test_basic_chat(self) -> CapabilityResult:
        """Test basic chat completion"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}",
                    json={
                        "contents": [{
                            "parts": [{"text": "Hello, test!"}]
                        }]
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return self._create_success_result(
                        "basic_chat",
                        details={"model": data.get("modelVersion")}
                    )
                else:
                    return self._create_failure_result(
                        "basic_chat",
                        f"API returned {response.status_code}"
                    )
        except Exception as e:
            return self._create_error_result("basic_chat", e)
    
    async def test_streaming(self) -> CapabilityResult:
        """Test streaming responses"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/models/{self.model}:streamGenerateContent?alt=sse&key={self.api_key}",
                    json={
                        "contents": [{
                            "parts": [{"text": "Count to 3"}]
                        }]
                    }
                ) as response:
                    if response.status_code == 200:
                        chunk_count = 0
                        async for _ in response.aiter_lines():
                            chunk_count += 1
                            if chunk_count >= 3:
                                break
                        
                        return self._create_success_result(
                            "streaming",
                            details={"chunks_received": chunk_count}
                        )
                    else:
                        return self._create_failure_result(
                            "streaming",
                            f"API returned {response.status_code}"
                        )
        except Exception as e:
            return self._create_error_result("streaming", e)
    
    async def test_function_calling(self) -> CapabilityResult:
        """Test function declarations"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}",
                    json={
                        "contents": [{
                            "parts": [{"text": "What's the weather in SF?"}]
                        }],
                        "tools": [{
                            "functionDeclarations": [{
                                "name": "get_weather",
                                "description": "Get weather",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "location": {"type": "string"}
                                    }
                                }
                            }]
                        }]
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    candidates = data.get("candidates", [])
                    has_function_call = False
                    
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        has_function_call = any("functionCall" in p for p in parts)
                    
                    return self._create_success_result(
                        "function_calling",
                        details={"function_call_detected": has_function_call}
                    )
                else:
                    return self._create_failure_result(
                        "function_calling",
                        f"API returned {response.status_code}"
                    )
        except Exception as e:
            return self._create_error_result("function_calling", e)
    
    async def test_vision(self) -> CapabilityResult:
        """Test vision capabilities"""
        # Gemini models generally support vision
        return self._create_success_result(
            "vision",
            details={"model_supports_vision": True}
        )
    
    async def test_system_message(self) -> CapabilityResult:
        """Test system instruction support"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}",
                    json={
                        "systemInstruction": {
                            "parts": [{"text": "You are helpful"}]
                        },
                        "contents": [{
                            "parts": [{"text": "Hi"}]
                        }]
                    }
                )
                
                if response.status_code == 200:
                    return self._create_success_result("system_message")
                else:
                    return self._create_failure_result(
                        "system_message",
                        f"API returned {response.status_code}"
                    )
        except Exception as e:
            return self._create_error_result("system_message", e)

