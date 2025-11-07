# SPDX-License-Identifier: GPL-3.0-or-later
#
# Toolify: Empower any LLM with function calling capabilities.
# Copyright (C) 2025 FunnyCups (https://github.com/funnycups)

"""
Anthropic capability detector.
"""

import httpx
import logging
from .base_detector import BaseCapabilityDetector, CapabilityResult, DetectionReport, CapabilityStatus

logger = logging.getLogger(__name__)


class AnthropicCapabilityDetector(BaseCapabilityDetector):
    """Detector for Anthropic API capabilities"""
    
    def get_provider_name(self) -> str:
        return "anthropic"
    
    def get_default_base_url(self) -> str:
        return "https://api.anthropic.com"
    
    def get_default_model(self) -> str:
        return "claude-3-5-sonnet-20241022"
    
    async def detect_all_capabilities(self) -> DetectionReport:
        """Detect all Anthropic capabilities"""
        logger.info(f"Starting capability detection for Anthropic (model: {self.model})")
        
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
                    f"{self.base_url}/v1/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01"
                    },
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": "Hello, test!"}],
                        "max_tokens": 50
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return self._create_success_result(
                        "basic_chat",
                        details={"model": data.get("model"), "type": data.get("type")}
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
                    f"{self.base_url}/v1/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01"
                    },
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": "Count to 3"}],
                        "max_tokens": 50,
                        "stream": True
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
        """Test tool use (Anthropic's function calling)"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/v1/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01"
                    },
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": "What's the weather in SF?"}],
                        "max_tokens": 200,
                        "tools": [{
                            "name": "get_weather",
                            "description": "Get weather",
                            "input_schema": {
                                "type": "object",
                                "properties": {
                                    "location": {"type": "string"}
                                }
                            }
                        }]
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    content = data.get("content", [])
                    has_tool_use = any(c.get("type") == "tool_use" for c in content)
                    
                    return self._create_success_result(
                        "function_calling",
                        details={"tool_use_detected": has_tool_use}
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
        # Claude 3+ models support vision
        vision_models = ["claude-3", "claude-3.5"]
        
        if any(vm in self.model for vm in vision_models):
            return self._create_success_result(
                "vision",
                details={"model_supports_vision": True}
            )
        else:
            return CapabilityResult(
                capability_name="vision",
                status=CapabilityStatus.UNKNOWN,
                details={"model_supports_vision": False}
            )
    
    async def test_system_message(self) -> CapabilityResult:
        """Test system message support"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/v1/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01"
                    },
                    json={
                        "model": self.model,
                        "system": "You are helpful",
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 50
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

