# SPDX-License-Identifier: GPL-3.0-or-later
#
# Toolify: Empower any LLM with function calling capabilities.
# Copyright (C) 2025 FunnyCups (https://github.com/funnycups)

"""
OpenAI capability detector.
"""

import httpx
import logging
from typing import Optional
from .base_detector import BaseCapabilityDetector, CapabilityResult, DetectionReport, CapabilityStatus

logger = logging.getLogger(__name__)


class OpenAICapabilityDetector(BaseCapabilityDetector):
    """Detector for OpenAI API capabilities"""
    
    def get_provider_name(self) -> str:
        return "openai"
    
    def get_default_base_url(self) -> str:
        return "https://api.openai.com/v1"
    
    def get_default_model(self) -> str:
        return "gpt-4o-mini"
    
    async def detect_all_capabilities(self) -> DetectionReport:
        """Detect all OpenAI capabilities"""
        logger.info(f"Starting capability detection for OpenAI (model: {self.model})")
        
        capabilities = []
        
        # Test各种能力
        capabilities.append(await self.test_basic_chat())
        capabilities.append(await self.test_streaming())
        capabilities.append(await self.test_function_calling())
        capabilities.append(await self.test_vision())
        capabilities.append(await self.test_system_message())
        capabilities.append(await self.test_json_mode())
        
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
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": "Hello, test!"}],
                        "max_tokens": 10
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return self._create_success_result(
                        "basic_chat",
                        details={"model": data.get("model"), "object": data.get("object")}
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
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": "Count to 3"}],
                        "stream": True,
                        "max_tokens": 20
                    }
                ) as response:
                    if response.status_code == 200:
                        chunk_count = 0
                        async for _ in response.aiter_lines():
                            chunk_count += 1
                            if chunk_count >= 3:  # 确认收到流式数据
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
        """Test function calling"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": "What's the weather in SF?"}],
                        "tools": [{
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "description": "Get weather",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "location": {"type": "string"}
                                    }
                                }
                            }
                        }],
                        "max_tokens": 100
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    message = data.get("choices", [{}])[0].get("message", {})
                    has_tool_calls = "tool_calls" in message
                    
                    return self._create_success_result(
                        "function_calling",
                        details={"tool_calls_used": has_tool_calls}
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
        # 简化的视觉测试 - 只检查模型是否支持
        vision_models = ["gpt-4-vision", "gpt-4o", "gpt-4o-mini"]
        
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
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "You are helpful"},
                            {"role": "user", "content": "Hi"}
                        ],
                        "max_tokens": 10
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
    
    async def test_json_mode(self) -> CapabilityResult:
        """Test JSON mode / structured output"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": "Output JSON"}],
                        "response_format": {"type": "json_object"},
                        "max_tokens": 50
                    }
                )
                
                if response.status_code == 200:
                    return self._create_success_result("json_mode")
                else:
                    # JSON模式可能不被所有模型支持
                    return CapabilityResult(
                        capability_name="json_mode",
                        status=CapabilityStatus.NOT_SUPPORTED,
                        details={"status_code": response.status_code}
                    )
        except Exception as e:
            return self._create_error_result("json_mode", e)

