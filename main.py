# SPDX-License-Identifier: GPL-3.0-or-later
#
# Toolify: Empower any LLM with function calling capabilities.
# Copyright (C) 2025 FunnyCups (https://github.com/funnycups)

"""
Main FastAPI application for Toolify middleware.
Refactored for better modularity and maintainability.
"""

import os
import json
import uuid
import httpx
import traceback
import time
import logging
import yaml
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, Request, Header, HTTPException, Depends, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

# Toolify modules
from toolify_core.models import ChatCompletionRequest, AnthropicMessage, GeminiRequest, Tool, ToolFunction
from toolify_core.token_counter import TokenCounter
from config_loader import config_loader, AppConfig
from admin_auth import (
    LoginRequest, LoginResponse, verify_admin_token,
    verify_password, create_access_token, get_admin_credentials
)
from toolify_core.function_calling import (
    generate_function_prompt,
    generate_random_trigger_signal,
    parse_function_calls_xml
)
from toolify_core.tool_mapping import store_tool_call_mapping

# Format converters (new unified system)
from toolify_core.converters import (
    ConverterFactory,
    OpenAIConverter,
    AnthropicConverter,
    GeminiConverter
)

# Capability detection
from toolify_core.capability_detector import (
    DetectorFactory,
    OpenAICapabilityDetector,
    AnthropicCapabilityDetector,
    GeminiCapabilityDetector
)

# Backward compatibility - keep old adapter functions
from toolify_core.anthropic_adapter import (
    anthropic_to_openai_request,
    openai_to_anthropic_response,
    stream_openai_to_anthropic
)
from toolify_core.message_processor import (
    preprocess_messages,
    validate_message_structure,
    safe_process_tool_choice
)
from toolify_core.upstream_router import find_upstream
from toolify_core.streaming_proxy import stream_proxy_with_fc_transform

logger = logging.getLogger(__name__)


def build_upstream_url(base_url: str, endpoint: str) -> str:
    """
    智能构建上游URL，自动处理 /v1 路径
    
    Args:
        base_url: 上游基础URL
        endpoint: 端点路径（如 /chat/completions, /messages）
    
    Returns:
        完整的URL
    """
    # 移除尾部斜杠
    base_url = base_url.rstrip('/')
    
    # 如果 endpoint 不以 / 开头，添加
    if not endpoint.startswith('/'):
        endpoint = '/' + endpoint
    
    # 智能处理 /v1 前缀
    # 如果 base_url 已经包含 /v1, /v1beta 等，直接拼接
    if any(base_url.endswith(suffix) for suffix in ['/v1', '/v1beta', '/v1alpha']):
        return base_url + endpoint
    
    # 如果 endpoint 需要 /v1 但 base_url 没有，自动添加
    # OpenAI 端点
    if endpoint in ['/chat/completions', '/completions', '/models', '/embeddings']:
        if '/v1' not in base_url:
            return base_url + '/v1' + endpoint
    
    # Anthropic 端点 - 已经包含 /v1/messages
    if endpoint.startswith('/v1/'):
        return base_url + endpoint
    
    # 默认直接拼接
    return base_url + endpoint


# Global variables
app_config: AppConfig = None
MODEL_TO_SERVICE_MAPPING: Dict[str, List[Dict[str, Any]]] = {}
ALIAS_MAPPING: Dict[str, List[str]] = {}
DEFAULT_SERVICE: Dict[str, Any] = {}
ALLOWED_CLIENT_KEYS: List[str] = []
GLOBAL_TRIGGER_SIGNAL: str = ""
token_counter = TokenCounter()


def load_runtime_config(reload: bool = False):
    """Load or reload runtime configuration and derived globals."""
    global app_config, MODEL_TO_SERVICE_MAPPING, ALIAS_MAPPING, DEFAULT_SERVICE
    global ALLOWED_CLIENT_KEYS, GLOBAL_TRIGGER_SIGNAL

    if reload:
        app_config = config_loader.reload_config()
        logger.info("🔄 Reloaded configuration from disk")
    else:
        app_config = config_loader.load_config()
    
    log_level_str = app_config.features.log_level
    if log_level_str == "DISABLED":
        log_level = logging.CRITICAL + 1
    else:
        log_level = getattr(logging, log_level_str, logging.INFO)
    
    # Configure logging (avoid adding duplicate handlers on reload)
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        root_logger.setLevel(log_level)
    
    logger.info(f"✅ Configuration loaded successfully: {config_loader.config_path}")
    logger.info(f"📊 Configured {len(app_config.upstream_services)} upstream services")
    logger.info(f"🔑 Configured {len(app_config.client_authentication.allowed_keys)} client keys")
    
    MODEL_TO_SERVICE_MAPPING, ALIAS_MAPPING = config_loader.get_model_to_service_mapping()
    DEFAULT_SERVICE = config_loader.get_default_service()
    ALLOWED_CLIENT_KEYS = config_loader.get_allowed_client_keys()
    GLOBAL_TRIGGER_SIGNAL = generate_random_trigger_signal()
    
    logger.info(f"🎯 Configured {len(MODEL_TO_SERVICE_MAPPING)} model mappings")
    if ALIAS_MAPPING:
        logger.info(f"🔄 Configured {len(ALIAS_MAPPING)} model aliases: {list(ALIAS_MAPPING.keys())}")
    logger.info(f"🔄 Default service: {DEFAULT_SERVICE['name']}")


# Initialize FastAPI app (don't load config at module level for better IDE support)
app = FastAPI()
http_client = httpx.AsyncClient()

# Register converters
ConverterFactory.register_converter("openai", OpenAIConverter)
ConverterFactory.register_converter("anthropic", AnthropicConverter)
ConverterFactory.register_converter("gemini", GeminiConverter)
logger.info("✅ Registered format converters: OpenAI, Anthropic, Gemini")

# Register capability detectors
DetectorFactory.register_detector("openai", OpenAICapabilityDetector)
DetectorFactory.register_detector("anthropic", AnthropicCapabilityDetector)
DetectorFactory.register_detector("gemini", GeminiCapabilityDetector)
logger.info("✅ Registered capability detectors: OpenAI, Anthropic, Gemini")

# Flag to track if configuration is loaded
_config_loaded = False


def ensure_config_loaded():
    """Ensure configuration is loaded before handling requests."""
    global _config_loaded
    if not _config_loaded:
        try:
            load_runtime_config()
            _config_loaded = True
            logger.info("✅ Configuration loaded successfully on first request")
        except Exception as e:
            logger.error(f"❌ Configuration loading failed: {type(e).__name__}")
            logger.error(f"❌ Error details: {str(e)}")
            logger.error("💡 Please ensure config.yaml file exists and is properly formatted")
            raise HTTPException(
                status_code=500,
                detail=f"Server configuration error: {str(e)}"
            )

# Add CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def debug_middleware(request: Request, call_next):
    """Middleware for debugging - logs response status."""
    response = await call_next(request)
    
    if response.status_code == 422:
        logger.error(f"🔍 ❌ Validation failed for {request.method} {request.url.path}")
        logger.error(f"🔍 Response status code: 422 (Pydantic validation failure)")
        logger.error(f"🔍 Check the detailed error logs above for validation details")
    
    return response


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle Pydantic validation errors with detailed error information."""
    logger.error("=" * 80)
    logger.error("❌ PYDANTIC VALIDATION ERROR DETAILS")
    logger.error("=" * 80)
    logger.error(f"📍 Request URL: {request.url}")
    logger.error(f"📍 Request Method: {request.method}")
    logger.error(f"📍 Model being validated: {exc.title if hasattr(exc, 'title') else 'Unknown'}")
    
    # Log request headers
    logger.error(f"📋 Request Headers:")
    for header_name, header_value in request.headers.items():
        if header_name.lower() in ["authorization", "x-api-key"]:
            masked_value = "***" + header_value[-8:] if len(header_value) > 8 else "***"
            logger.error(f"   {header_name}: {masked_value}")
        else:
            logger.error(f"   {header_name}: {header_value}")
    
    # Try to read and log the raw request body
    try:
        body_bytes = await request.body()
        body_text = body_bytes.decode('utf-8')
        logger.error(f"📦 Raw Request Body (first 2000 chars):")
        logger.error(body_text[:2000])
        if len(body_text) > 2000:
            logger.error(f"   ... (total {len(body_text)} chars)")
        
        # Try to parse as JSON for better readability
        try:
            import json
            body_json = json.loads(body_text)
            logger.error(f"📦 Parsed Request JSON:")
            logger.error(f"   Keys: {list(body_json.keys())}")
            logger.error(f"   Model: {body_json.get('model', 'N/A')}")
            logger.error(f"   Messages count: {len(body_json.get('messages', []))}")
            logger.error(f"   Max tokens: {body_json.get('max_tokens', 'NOT PROVIDED')}")
            logger.error(f"   Stream: {body_json.get('stream', 'N/A')}")
            logger.error(f"   Tools: {len(body_json.get('tools', []))} tools")
        except:
            pass
            
    except Exception as e:
        logger.error(f"⚠️  Could not read request body: {e}")
    
    logger.error(f"🔴 Validation Errors ({len(exc.errors())} error(s)):")
    for i, error in enumerate(exc.errors(), 1):
        logger.error(f"   Error {i}:")
        logger.error(f"      Location: {' -> '.join(str(loc) for loc in error.get('loc', []))}")
        logger.error(f"      Message: {error.get('msg')}")
        logger.error(f"      Type: {error.get('type')}")
        if 'input' in error:
            input_repr = repr(error['input'])
            logger.error(f"      Input: {input_repr[:300]}{'...' if len(input_repr) > 300 else ''}")
    logger.error("=" * 80)
    
    # Build user-friendly error message
    error_messages = []
    for error in exc.errors():
        field = ' -> '.join(str(loc) for loc in error.get('loc', []))
        msg = error.get('msg', 'Validation error')
        error_messages.append(f"{field}: {msg}")
    
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "message": "Request validation failed: " + "; ".join(error_messages[:3]),
                "type": "invalid_request_error",
                "code": "invalid_request",
                "details": exc.errors()
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all uncaught exceptions."""
    logger.error(f"❌ Unhandled exception: {exc}")
    logger.error(f"❌ Request URL: {request.url}")
    logger.error(f"❌ Exception type: {type(exc).__name__}")
    logger.error(f"❌ Error stack: {traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "server_error",
                "code": "internal_error"
            }
        }
    )


async def verify_api_key(authorization: str = Header(...)):
    """Dependency: verify client API key."""
    logger.debug(f"🔐 API Key Verification")
    logger.debug(f"   Authorization header: {authorization[:20]}...{authorization[-8:] if len(authorization) > 20 else authorization}")
    
    if not authorization:
        logger.error("❌ Missing Authorization header")
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    
    # Extract key
    if authorization.startswith("Bearer "):
        client_key = authorization[7:]
    else:
        # Some clients might not include "Bearer " prefix
        client_key = authorization
    
    logger.debug(f"   Extracted key: ***{client_key[-8:] if len(client_key) > 8 else '***'}")
    
    if app_config.features.key_passthrough:
        logger.debug(f"   Mode: Key passthrough (validation skipped)")
        return client_key
    
    logger.debug(f"   Mode: Validating against {len(ALLOWED_CLIENT_KEYS)} allowed keys")
    logger.debug(f"   Allowed keys: {[f'***{k[-8:]}' for k in ALLOWED_CLIENT_KEYS]}")
    
    if client_key not in ALLOWED_CLIENT_KEYS:
        logger.error(f"❌ Unauthorized key: ***{client_key[-8:]}")
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    logger.debug(f"✅ Key validated successfully")
    return client_key


@app.get("/")
def read_root():
    """Root endpoint showing service status."""
    ensure_config_loaded()
    return {
        "status": "OpenAI Function Call Middleware is running",
        "config": {
            "upstream_services_count": len(app_config.upstream_services),
            "client_keys_count": len(app_config.client_authentication.allowed_keys),
            "models_count": len(MODEL_TO_SERVICE_MAPPING),
            "features": {
                "function_calling": app_config.features.enable_function_calling,
                "log_level": app_config.features.log_level,
                "convert_developer_to_system": app_config.features.convert_developer_to_system,
                "random_trigger": True
            }
        }
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    body: ChatCompletionRequest,
    _api_key: str = Depends(verify_api_key)
):
    """Main chat completion endpoint, proxy and inject function calling capabilities."""
    ensure_config_loaded()
    start_time = time.time()
    
    # Count input tokens
    prompt_tokens = token_counter.count_tokens(body.messages, body.model)
    logger.info(f"📊 Request to {body.model} - Input tokens: {prompt_tokens}")
    
    try:
        logger.debug(f"🔧 Received request, model: {body.model}")
        logger.debug(f"🔧 Number of messages: {len(body.messages)}")
        logger.debug(f"🔧 Number of tools: {len(body.tools) if body.tools else 0}")
        logger.debug(f"🔧 Streaming: {body.stream}")
        
        upstreams, actual_model = find_upstream(
            body.model,
            MODEL_TO_SERVICE_MAPPING,
            ALIAS_MAPPING,
            DEFAULT_SERVICE,
            app_config.features.model_passthrough,
            app_config.upstream_services
        )

        logger.debug(f"🔧 Found {len(upstreams)} upstream service(s) for model {body.model}")
        for i, srv in enumerate(upstreams):
            logger.debug(f"🔧 Service {i + 1}: {srv['name']} (priority: {srv.get('priority', 0)})")
        
        logger.debug(f"🔧 Starting message preprocessing, original message count: {len(body.messages)}")
        processed_messages = preprocess_messages(
            body.messages,
            GLOBAL_TRIGGER_SIGNAL,
            app_config.features.convert_developer_to_system
        )
        logger.debug(f"🔧 Preprocessing completed, processed message count: {len(processed_messages)}")
        
        if not validate_message_structure(processed_messages, app_config.features.convert_developer_to_system):
            logger.error(f"❌ Message structure validation failed, but continuing processing")
        
        request_body_dict = body.model_dump(exclude_unset=True)
        request_body_dict["model"] = actual_model
        request_body_dict["messages"] = processed_messages
        is_fc_enabled = app_config.features.enable_function_calling
        has_tools_in_request = bool(body.tools)
        has_function_call = is_fc_enabled and has_tools_in_request
        
        logger.debug(f"🔧 Request body constructed, message count: {len(processed_messages)}")
        
    except Exception as e:
        logger.error(f"❌ Request preprocessing failed: {str(e)}")
        logger.error(f"❌ Error type: {type(e).__name__}")
        if hasattr(app_config, 'debug') and app_config.debug:
            logger.error(f"❌ Error stack: {traceback.format_exc()}")
        
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "message": "Invalid request format",
                    "type": "invalid_request_error",
                    "code": "invalid_request"
                }
            }
        )

    if has_function_call:
        logger.debug(f"🔧 Using global trigger signal for this request: {GLOBAL_TRIGGER_SIGNAL}")
        
        # Check if function calling injection is enabled for this upstream
        upstream_fc_enabled = upstreams[0].get('inject_function_calling')
        if upstream_fc_enabled is None:
            # Inherit from global setting
            upstream_fc_enabled = app_config.features.enable_function_calling
        
        if not upstream_fc_enabled:
            logger.info(f"🔧 Function calling injection disabled for upstream '{upstreams[0]['name']}', passing through tools to native API")
            # Don't inject, let upstream handle tools natively
            has_function_call = False
        else:
            function_prompt, _ = generate_function_prompt(
                body.tools,
                GLOBAL_TRIGGER_SIGNAL,
                app_config.features.prompt_template
            )
            
            tool_choice_prompt = safe_process_tool_choice(body.tool_choice)
            if tool_choice_prompt:
                function_prompt += tool_choice_prompt

            # 打印 prompt 大小信息
            prompt_chars = len(function_prompt)
            estimated_tokens = prompt_chars // 4  # 粗略估算
            logger.info("=" * 80)
            logger.info(f"📏 Function Calling Prompt Size:")
            logger.info(f"   Upstream: {upstreams[0]['name']}")
            logger.info(f"   Tools count: {len(body.tools)}")
            logger.info(f"   Prompt characters: {prompt_chars:,}")
            logger.info(f"   Estimated tokens: ~{estimated_tokens:,}")
            logger.info(f"   Original messages: {len(body.messages)}")
            logger.info("=" * 80)

            system_message = {"role": "system", "content": function_prompt}
            request_body_dict["messages"].insert(0, system_message)
            
            # 计算注入后的总大小
            total_chars = sum(len(str(m.get('content', ''))) for m in request_body_dict["messages"])
            logger.info(f"📏 Total request size after injection: {total_chars:,} characters (~{total_chars//4:,} tokens)")
            logger.info(f"📏 Total messages after injection: {len(request_body_dict['messages'])}")
            
            if "tools" in request_body_dict:
                del request_body_dict["tools"]
            if "tool_choice" in request_body_dict:
                del request_body_dict["tool_choice"]

    elif has_tools_in_request and not is_fc_enabled:
        logger.info(f"🔧 Function calling is disabled by configuration, ignoring 'tools' and 'tool_choice' in request.")
        if "tools" in request_body_dict:
            del request_body_dict["tools"]
        if "tool_choice" in request_body_dict:
            del request_body_dict["tool_choice"]

    # Try each upstream service by priority until one succeeds
    last_error = None

    if not body.stream:
        # Non-streaming: try each upstream with failover
        for upstream_idx, upstream in enumerate(upstreams):
            upstream_url = build_upstream_url(upstream['base_url'], '/chat/completions')

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {_api_key}" if app_config.features.key_passthrough else f"Bearer {upstream['api_key']}",
                "Accept": "application/json"
            }

            logger.info(
                f"📝 Attempting upstream {upstream_idx + 1}/{len(upstreams)}: {upstream['name']} (priority: {upstream.get('priority', 0)})")
            logger.info(
                f"📝 Model: {request_body_dict.get('model', 'unknown')}, Messages: {len(request_body_dict.get('messages', []))}")

            try:
                logger.debug(f"🔧 Sending upstream request to: {upstream_url}")
                logger.debug(f"🔧 has_function_call: {has_function_call}")
                logger.debug(f"🔧 Request body contains tools: {bool(body.tools)}")
                
                upstream_response = await http_client.post(
                    upstream_url, json=request_body_dict, headers=headers, timeout=app_config.server.timeout
                )
                upstream_response.raise_for_status()
                
                # 添加响应内容检查，防止空响应或非JSON响应
                response_text = upstream_response.text
                print(f"\n{'='*80}")
                print(f"🔍 UPSTREAM NON-STREAMING RESPONSE")
                print(f"{'='*80}")
                print(f"Status: {upstream_response.status_code}")
                print(f"Headers: {dict(upstream_response.headers)}")
                print(f"Body length: {len(response_text)} bytes")
                print(f"Body (first 1000 chars):\n{response_text[:1000]}")
                if len(response_text) > 1000:
                    print(f"... (total {len(response_text)} bytes)")
                print(f"{'='*80}\n")
                
                logger.debug(f"🔧 Upstream response status code: {upstream_response.status_code}")
                logger.debug(f"🔧 Upstream response length: {len(response_text)} bytes")

                if not response_text or response_text.strip() == "":
                    logger.error(f"❌ Upstream {upstream['name']} returned empty response body with 200 status")
                    raise ValueError("Empty response from upstream service")

                try:
                    response_json = upstream_response.json()
                except json.JSONDecodeError as json_err:
                    logger.error(f"❌ Failed to parse JSON from {upstream['name']}")
                    logger.error(f"❌ Response content (first 500 chars): {response_text[:500]}")
                    logger.error(f"❌ Content-Type: {upstream_response.headers.get('content-type', 'unknown')}")
                    raise ValueError(f"Invalid JSON response: {json_err}")
                
                # Count output tokens and handle usage
                completion_text = ""
                if response_json.get("choices") and len(response_json["choices"]) > 0:
                    content = response_json["choices"][0].get("message", {}).get("content")
                    if content:
                        completion_text = content
                
                # Calculate our estimated tokens
                estimated_completion_tokens = token_counter.count_text_tokens(completion_text,
                                                                              body.model) if completion_text else 0
                estimated_prompt_tokens = prompt_tokens
                estimated_total_tokens = estimated_prompt_tokens + estimated_completion_tokens
                elapsed_time = time.time() - start_time
                
                # Check if upstream provided usage and respect it
                upstream_usage = response_json.get("usage", {})
                if upstream_usage:
                    # Preserve upstream's usage structure and only replace zero values
                    final_usage = upstream_usage.copy()
                    
                    # Replace zero or missing values with our estimates
                    if not final_usage.get("prompt_tokens") or final_usage.get("prompt_tokens") == 0:
                        final_usage["prompt_tokens"] = estimated_prompt_tokens
                        logger.debug(f"🔧 Replaced zero/missing prompt_tokens with estimate: {estimated_prompt_tokens}")
                    
                    if not final_usage.get("completion_tokens") or final_usage.get("completion_tokens") == 0:
                        final_usage["completion_tokens"] = estimated_completion_tokens
                        logger.debug(
                            f"🔧 Replaced zero/missing completion_tokens with estimate: {estimated_completion_tokens}")
                    
                    if not final_usage.get("total_tokens") or final_usage.get("total_tokens") == 0:
                        final_usage["total_tokens"] = final_usage.get("prompt_tokens",
                                                                      estimated_prompt_tokens) + final_usage.get(
                            "completion_tokens", estimated_completion_tokens)
                        logger.debug(
                            f"🔧 Replaced zero/missing total_tokens with calculated value: {final_usage['total_tokens']}")
                    
                    response_json["usage"] = final_usage
                    logger.debug(f"🔧 Preserved upstream usage with replacements: {final_usage}")
                else:
                    # No upstream usage, provide our estimates
                    response_json["usage"] = {
                        "prompt_tokens": estimated_prompt_tokens,
                        "completion_tokens": estimated_completion_tokens,
                        "total_tokens": estimated_total_tokens
                    }
                    logger.debug(f"🔧 No upstream usage found, using estimates")
                
                # Log token statistics
                actual_usage = response_json["usage"]
                logger.info("=" * 60)
                logger.info(f"📊 Token Usage Statistics - Model: {body.model}")
                logger.info(f"   Input Tokens: {actual_usage.get('prompt_tokens', 0)}")
                logger.info(f"   Output Tokens: {actual_usage.get('completion_tokens', 0)}")
                logger.info(f"   Total Tokens: {actual_usage.get('total_tokens', 0)}")
                logger.info(f"   Duration: {elapsed_time:.2f}s")
                logger.info("=" * 60)
                
                if has_function_call:
                    content = response_json["choices"][0]["message"]["content"]
                    logger.debug(f"🔧 Complete response content: {repr(content)}")
                    
                    parsed_tools = parse_function_calls_xml(content, GLOBAL_TRIGGER_SIGNAL)
                    logger.debug(f"🔧 XML parsing result: {parsed_tools}")
                    
                    if parsed_tools:
                        logger.debug(f"🔧 Successfully parsed {len(parsed_tools)} tool calls")
                        tool_calls = []
                        for tool in parsed_tools:
                            tool_call_id = f"call_{uuid.uuid4().hex}"
                            store_tool_call_mapping(
                                tool_call_id,
                                tool["name"],
                                tool["args"],
                                f"Calling tool {tool['name']}"
                            )
                            tool_calls.append({
                                "id": tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": tool["name"],
                                    "arguments": json.dumps(tool["args"])
                                }
                            })
                        logger.debug(f"🔧 Converted tool_calls: {tool_calls}")
                        
                        response_json["choices"][0]["message"] = {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": tool_calls,
                        }
                        response_json["choices"][0]["finish_reason"] = "tool_calls"
                        logger.debug(f"🔧 Function call conversion completed")
                    else:
                        logger.debug(f"🔧 No tool calls detected, returning original content (including think blocks)")
                else:
                    logger.debug(f"🔧 No function calls detected or conversion conditions not met")
                
                return JSONResponse(content=response_json)

            except httpx.HTTPStatusError as e:
                logger.warning(f"⚠️  Upstream {upstream['name']} failed: status_code={e.response.status_code}")
                logger.debug(f"🔧 Error details: {e.response.text}")

                last_error = e

                # Check if we should retry with next upstream
                # Don't retry for client errors (400, 401, 403) - these won't succeed with different upstream
                if e.response.status_code in [400, 401, 403]:
                    logger.error(f"❌ Client error from {upstream['name']}, not retrying other upstreams")
                    if e.response.status_code == 400:
                        error_response = {
                            "error": {"message": "Invalid request parameters", "type": "invalid_request_error",
                                      "code": "bad_request"}}
                    elif e.response.status_code == 401:
                        error_response = {"error": {"message": "Authentication failed", "type": "authentication_error",
                                                    "code": "unauthorized"}}
                    elif e.response.status_code == 403:
                        error_response = {
                            "error": {"message": "Access forbidden", "type": "permission_error", "code": "forbidden"}}
                    return JSONResponse(content=error_response, status_code=e.response.status_code)

                # For 429 and 5xx errors, try next upstream if available
                if upstream_idx < len(upstreams) - 1:
                    logger.info(f"🔄 Trying next upstream service (failover)...")
                    continue
                else:
                    # All upstreams failed
                    logger.error(f"❌ All {len(upstreams)} upstream services failed")
                    if e.response.status_code == 429:
                        error_response = {
                            "error": {"message": "Rate limit exceeded on all upstreams", "type": "rate_limit_error",
                                      "code": "rate_limit_exceeded"}}
                    elif e.response.status_code >= 500:
                        error_response = {"error": {"message": "All upstream services temporarily unavailable",
                                                    "type": "service_error", "code": "upstream_error"}}
                    else:
                        error_response = {
                            "error": {"message": "Request processing failed on all upstreams", "type": "api_error",
                                      "code": "unknown_error"}}
                    return JSONResponse(content=error_response, status_code=e.response.status_code)
            
            except ValueError as e:
                # 捕获空响应或JSON解析错误
                logger.error(f"❌ Invalid response from {upstream['name']}: {e}")
                last_error = e
                if upstream_idx < len(upstreams) - 1:
                    logger.info(f"🔄 Trying next upstream service...")
                    continue
                else:
                    logger.error(f"❌ All upstreams failed - invalid responses")
                    return JSONResponse(
                        status_code=502,
                        content={"error": {"message": "All upstream services returned invalid responses",
                                           "type": "bad_gateway", "code": "invalid_upstream_response"}}
                    )

            except Exception as e:
                logger.error(f"❌ Unexpected error with {upstream['name']}: {type(e).__name__}: {e}")
                logger.error(f"❌ Error traceback: {traceback.format_exc()}")
                last_error = e
                if upstream_idx < len(upstreams) - 1:
                    logger.info(f"🔄 Trying next upstream service...")
                    continue
                else:
                    logger.error(f"❌ All upstreams failed with errors")
                    return JSONResponse(
                        status_code=500,
                        content={"error": {"message": "All upstream services failed", "type": "service_error",
                                           "code": "all_upstreams_failed"}}
                    )

    else:
        # Streaming: use the highest priority upstream (first in list)
        upstream = upstreams[0]
        upstream_url = build_upstream_url(upstream['base_url'], '/chat/completions')

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {_api_key}" if app_config.features.key_passthrough else f"Bearer {upstream['api_key']}",
            "Accept": "text/event-stream"
        }

        logger.info(f"📝 Streaming to upstream: {upstream['name']} (priority: {upstream.get('priority', 0)})")

        async def stream_with_token_count():
            completion_tokens = 0
            completion_text = ""
            done_received = False
            upstream_usage_chunk = None
            
            async for chunk in stream_proxy_with_fc_transform(
                upstream_url,
                request_body_dict,
                headers,
                body.model,
                has_function_call,
                GLOBAL_TRIGGER_SIGNAL,
                http_client,
                app_config.server.timeout
            ):
                # Check if this is the [DONE] marker
                if chunk.startswith(b"data: "):
                    try:
                        line_data = chunk[6:].decode('utf-8').strip()
                        if line_data == "[DONE]":
                            done_received = True
                            # Don't yield the [DONE] marker yet, we'll send it after usage info
                            break
                        elif line_data:
                            chunk_json = json.loads(line_data)
                            
                            # Check if this chunk contains usage information
                            if "usage" in chunk_json:
                                upstream_usage_chunk = chunk_json
                                logger.debug(f"🔧 Detected upstream usage chunk: {chunk_json['usage']}")
                                # Don't yield upstream usage chunk yet, we'll process it
                                continue
                            
                            # Process regular content chunks
                            if "choices" in chunk_json and len(chunk_json["choices"]) > 0:
                                delta = chunk_json["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    completion_text += content
                    except (json.JSONDecodeError, KeyError, UnicodeDecodeError) as e:
                        logger.debug(f"Failed to parse chunk for token counting: {e}")
                        pass
                
                yield chunk
            
            # Calculate our estimated tokens
            estimated_completion_tokens = token_counter.count_text_tokens(completion_text,
                                                                          body.model) if completion_text else 0
            estimated_prompt_tokens = prompt_tokens
            estimated_total_tokens = estimated_prompt_tokens + estimated_completion_tokens
            elapsed_time = time.time() - start_time
            
            # Determine final usage
            final_usage = None
            if upstream_usage_chunk and "usage" in upstream_usage_chunk:
                # Respect upstream usage, but replace zero values
                upstream_usage = upstream_usage_chunk["usage"]
                final_usage = upstream_usage.copy()
                
                if not final_usage.get("prompt_tokens") or final_usage.get("prompt_tokens") == 0:
                    final_usage["prompt_tokens"] = estimated_prompt_tokens
                    logger.debug(f"🔧 Replaced zero/missing prompt_tokens with estimate: {estimated_prompt_tokens}")
                
                if not final_usage.get("completion_tokens") or final_usage.get("completion_tokens") == 0:
                    final_usage["completion_tokens"] = estimated_completion_tokens
                    logger.debug(
                        f"🔧 Replaced zero/missing completion_tokens with estimate: {estimated_completion_tokens}")
                
                if not final_usage.get("total_tokens") or final_usage.get("total_tokens") == 0:
                    final_usage["total_tokens"] = final_usage.get("prompt_tokens",
                                                                  estimated_prompt_tokens) + final_usage.get(
                        "completion_tokens", estimated_completion_tokens)
                    logger.debug(
                        f"🔧 Replaced zero/missing total_tokens with calculated value: {final_usage['total_tokens']}")
                
                logger.debug(f"🔧 Using upstream usage with replacements: {final_usage}")
            else:
                # No upstream usage, use our estimates
                final_usage = {
                    "prompt_tokens": estimated_prompt_tokens,
                    "completion_tokens": estimated_completion_tokens,
                    "total_tokens": estimated_total_tokens
                }
                logger.debug(f"🔧 No upstream usage found, using estimates")
            
            # Log token statistics
            logger.info("=" * 60)
            logger.info(f"📊 Token Usage Statistics - Model: {body.model}")
            logger.info(f"   Input Tokens: {final_usage['prompt_tokens']}")
            logger.info(f"   Output Tokens: {final_usage['completion_tokens']}")
            logger.info(f"   Total Tokens: {final_usage['total_tokens']}")
            logger.info(f"   Duration: {elapsed_time:.2f}s")
            logger.info("=" * 60)
            
            # Send usage information if requested via stream_options OR if upstream provided usage
            if (body.stream_options and body.stream_options.get("include_usage", False)) or upstream_usage_chunk:
                usage_chunk_to_send = {
                    "id": f"chatcmpl-{uuid.uuid4().hex}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": body.model,
                    "choices": [],
                    "usage": final_usage
                }
                
                # If upstream provided additional fields in the usage chunk, preserve them
                if upstream_usage_chunk:
                    for key in upstream_usage_chunk:
                        if key not in ["usage", "choices"] and key not in usage_chunk_to_send:
                            usage_chunk_to_send[key] = upstream_usage_chunk[key]
                
                yield f"data: {json.dumps(usage_chunk_to_send)}\n\n".encode('utf-8')
                logger.debug(f"🔧 Sent usage chunk in stream: {usage_chunk_to_send['usage']}")
            
            # Send [DONE] marker if it was received
            if done_received:
                yield b"data: [DONE]\n\n"
        
        return StreamingResponse(
            stream_with_token_count(),
            media_type="text/event-stream"
        )


async def verify_anthropic_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
    authorization: Optional[str] = Header(None)
) -> str:
    """Verify Anthropic API key from x-api-key or Authorization header."""
    logger.debug(f"🔐 Anthropic API Key Verification")
    
    # Priority 1: x-api-key header (standard Anthropic)
    if x_api_key:
        logger.debug(f"   Using x-api-key header: ***{x_api_key[-8:] if len(x_api_key) > 8 else '***'}")
        client_key = x_api_key
    # Priority 2: Authorization Bearer (for Claude CLI compatibility)
    elif authorization:
        logger.debug(f"   Using Authorization header: {authorization[:20]}...")
        if authorization.startswith("Bearer "):
            client_key = authorization[7:]
        else:
            client_key = authorization
        logger.debug(f"   Extracted key: ***{client_key[-8:] if len(client_key) > 8 else '***'}")
    else:
        logger.error("❌ Missing authentication header")
        raise HTTPException(
            status_code=401,
            detail="Missing authentication. Please provide 'x-api-key' or 'Authorization: Bearer' header"
        )
    
    # Validate key if not in passthrough mode
    if not app_config.features.key_passthrough:
        logger.debug(f"   Validating against {len(ALLOWED_CLIENT_KEYS)} allowed keys")
        if client_key not in ALLOWED_CLIENT_KEYS:
            logger.error(f"❌ Unauthorized key: ***{client_key[-8:]}")
            raise HTTPException(status_code=401, detail="Unauthorized")
    
    logger.debug(f"✅ Anthropic key validated successfully")
    return client_key


@app.post("/v1/messages")
async def anthropic_messages(
        request: Request,
        body: AnthropicMessage,
        _api_key: str = Depends(verify_anthropic_api_key)
):
    """Anthropic Messages API endpoint - converts to OpenAI format and back."""
    ensure_config_loaded()
    start_time = time.time()
    
    # 详细日志 - 请求信息
    logger.info("=" * 80)
    logger.info("📨 ANTHROPIC API REQUEST")
    logger.info("=" * 80)
    logger.info(f"🔑 Client API Key: ***{_api_key[-8:] if len(_api_key) > 8 else '***'}")
    logger.info(f"📋 Request Headers:")
    for name, value in request.headers.items():
        if name.lower() in ["authorization", "x-api-key"]:
            logger.info(f"   {name}: ***{value[-8:] if len(value) > 8 else '***'}")
        else:
            logger.info(f"   {name}: {value}")
    
    logger.info(f"📊 Request Body:")
    logger.info(f"   Model: {body.model}")
    logger.info(f"   Max tokens: {body.max_tokens}")
    logger.info(f"   Stream: {body.stream}")
    logger.info(f"   Messages: {len(body.messages)} messages")
    logger.info(f"   System: {'Yes' if body.system else 'No'}")
    logger.info(f"   Tools: {len(body.tools) if body.tools else 0} tools")
    logger.info(f"   Temperature: {body.temperature}")
    logger.info(f"   Top P: {body.top_p}")
    logger.info(f"   Top K: {body.top_k if hasattr(body, 'top_k') else 'N/A'}")
    logger.info("=" * 80)
    
    try:
        # Convert Anthropic request to OpenAI format
        anthropic_dict = body.model_dump()
        
        logger.info("🔄 Converting Anthropic request to OpenAI format...")
        logger.debug(f"📦 Anthropic dict keys: {list(anthropic_dict.keys())}")
        logger.debug(f"📦 Anthropic dict (first 500 chars): {str(anthropic_dict)[:500]}")
        
        openai_request = anthropic_to_openai_request(anthropic_dict)
        
        logger.info(f"✅ Converted to OpenAI format")
        logger.info(f"   OpenAI messages: {len(openai_request['messages'])}")
        logger.info(f"   OpenAI model: {openai_request.get('model')}")
        logger.info(f"   OpenAI max_tokens: {openai_request.get('max_tokens')}")
        logger.info(f"   OpenAI stream: {openai_request.get('stream')}")
        logger.debug(f"📦 Full OpenAI request: {openai_request}")
        
        # Check for tool role messages
        tool_role_msgs = [m for m in openai_request['messages'] if isinstance(m, dict) and m.get('role') == 'tool']
        if tool_role_msgs:
            logger.warning(f"⚠️ Found {len(tool_role_msgs)} messages with 'tool' role after conversion from Anthropic")
            for i, msg in enumerate(tool_role_msgs):
                logger.warning(f"   Tool message {i}: tool_call_id={msg.get('tool_call_id', 'N/A')}")
        
        # Preprocess messages to handle tool role (upstream may not support it)
        logger.debug(f"🔧 Starting message preprocessing for Anthropic API")
        openai_request['messages'] = preprocess_messages(
            openai_request['messages'],
            GLOBAL_TRIGGER_SIGNAL,
            app_config.features.convert_developer_to_system
        )
        logger.debug(f"🔧 After preprocessing: {len(openai_request['messages'])} messages")
        
        # Find upstream service first
        upstreams, actual_model = find_upstream(
            body.model,
            MODEL_TO_SERVICE_MAPPING,
            ALIAS_MAPPING,
            DEFAULT_SERVICE,
            app_config.features.model_passthrough,
            app_config.upstream_services
        )
        upstream = upstreams[0]  # Use highest priority upstream
        
        logger.info("🎯 UPSTREAM SELECTION")
        logger.info(f"   Upstream name: {upstream['name']}")
        logger.info(f"   Service type: {upstream.get('service_type', 'N/A')}")
        logger.info(f"   Priority: {upstream.get('priority', 0)}")
        logger.info(f"   Base URL: {upstream['base_url']}")
        logger.info(f"   API Key: ***{upstream['api_key'][-8:] if len(upstream.get('api_key', '')) > 8 else '***'}")
        logger.info(f"   Actual model: {actual_model}")
        logger.info(f"   FC injection: {upstream.get('inject_function_calling', 'inherit')}")
        
        # Apply Toolify's function calling logic if tools are present
        has_tools = "tools" in openai_request and openai_request["tools"]
        
        # Check if function calling is enabled for this upstream
        upstream_fc_enabled = upstream.get('inject_function_calling')
        if upstream_fc_enabled is None:
            # Inherit from global setting
            upstream_fc_enabled = app_config.features.enable_function_calling
        
        has_function_call = upstream_fc_enabled and has_tools
        
        if has_function_call:
            logger.info(f"🔧 Applying Toolify function calling injection for {len(openai_request['tools'])} tools")
            
            # Convert OpenAI tools format to Toolify Tool objects
            from pydantic import ValidationError
            tool_objects = []
            for tool_dict in openai_request["tools"]:
                try:
                    # Create Tool object from the converted format
                    tool_obj = Tool(
                        type="function",
                        function=ToolFunction(
                            name=tool_dict["function"]["name"],
                            description=tool_dict["function"].get("description", ""),
                            parameters=tool_dict["function"].get("parameters", {})
                        )
                    )
                    tool_objects.append(tool_obj)
                except (ValidationError, KeyError) as e:
                    logger.warning(f"⚠️  Failed to parse tool: {e}")
            
            if tool_objects:
                # Generate function calling prompt
                function_prompt, _ = generate_function_prompt(
                    tool_objects,
                    GLOBAL_TRIGGER_SIGNAL,
                    app_config.features.prompt_template
                )
                
                # 打印 Anthropic API 的 prompt 大小信息
                prompt_chars = len(function_prompt)
                estimated_tokens = prompt_chars // 4
                logger.info("=" * 80)
                logger.info(f"📏 Anthropic API - Function Calling Prompt Size:")
                logger.info(f"   Tools count: {len(tool_objects)}")
                logger.info(f"   Prompt characters: {prompt_chars:,}")
                logger.info(f"   Estimated tokens: ~{estimated_tokens:,}")
                logger.info(f"   Original messages: {len(openai_request['messages'])}")
                logger.info("=" * 80)
                
                # Inject into system message
                system_message = {"role": "system", "content": function_prompt}
                openai_request["messages"].insert(0, system_message)
                
                # 计算注入后的总大小
                total_chars = sum(len(str(m.get('content', ''))) for m in openai_request["messages"])
                logger.info(f"📏 Total request size after injection: {total_chars:,} characters (~{total_chars//4:,} tokens)")
                logger.info(f"📏 Total messages after injection: {len(openai_request['messages'])}")
                
                logger.debug(f"🔧 Injected function calling prompt with trigger signal")
                
                # Remove tools parameter (we're using prompt injection instead)
                if "tools" in openai_request:
                    del openai_request["tools"]
        
        elif has_tools and not upstream_fc_enabled:
            logger.info(f"🔧 Function calling injection disabled for upstream '{upstream['name']}', passing through tools to native API")
            # Keep tools in request for native API handling
        
        # Update model to actual upstream model
        openai_request["model"] = actual_model
        
        # 🔧 根据上游服务类型决定请求格式和端点
        upstream_service_type = upstream.get('service_type', 'openai')
        
        logger.info(f"🔄 FORMAT ROUTING")
        logger.info(f"   Client format: Anthropic")
        logger.info(f"   Upstream type: {upstream_service_type}")
        
        # 准备发送到上游的请求和URL
        if upstream_service_type == 'anthropic':
            # Anthropic → Anthropic: 直接使用原始 Anthropic 格式
            logger.info(f"   Strategy: Direct Anthropic passthrough")
            final_request = anthropic_dict  # 使用原始 Anthropic 请求
            upstream_url = f"{upstream['base_url'].rstrip('/')}/v1/messages"
            headers = {
                "Content-Type": "application/json",
                "x-api-key": _api_key if app_config.features.key_passthrough else upstream['api_key'],
                "anthropic-version": "2023-06-01"
            }
        elif upstream_service_type == 'openai':
            # Anthropic → OpenAI: 使用已转换的 OpenAI 格式
            logger.info(f"   Strategy: Convert to OpenAI format")
            final_request = openai_request
            upstream_url = build_upstream_url(upstream['base_url'], '/chat/completions')
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {_api_key}" if app_config.features.key_passthrough else f"Bearer {upstream['api_key']}",
            }
        elif upstream_service_type == 'gemini':
            # Anthropic → Gemini: 需要转换为 Gemini 格式
            logger.info(f"   Strategy: Convert to Gemini format")
            from toolify_core.converters import convert_request
            conversion = convert_request("anthropic", "gemini", anthropic_dict)
            if not conversion.success:
                raise HTTPException(status_code=500, detail=f"Format conversion failed: {conversion.error}")
            final_request = conversion.data
            # Gemini 需要从 final_request 中移除 model 和 stream 字段
            gemini_model = final_request.pop('model', actual_model)
            final_request.pop('stream', None)
            upstream_url = f"{upstream['base_url'].rstrip('/')}/models/{gemini_model}:generateContent?key={upstream['api_key']}"
            headers = {"Content-Type": "application/json"}
        else:
            raise HTTPException(status_code=500, detail=f"Unsupported upstream service type: {upstream_service_type}")
        
        logger.info("🚀 SENDING TO UPSTREAM")
        logger.info(f"   URL: {upstream_url}")
        logger.info(f"   Headers: {list(headers.keys())}")
        logger.info(f"   Request format: {upstream_service_type}")
        logger.info(f"   Request model: {final_request.get('model')}")
        if upstream_service_type == 'anthropic':
            logger.info(f"   Request messages: {len(final_request.get('messages', []))}")
            logger.info(f"   Request max_tokens: {final_request.get('max_tokens')}")
            logger.info(f"   Request stream: {final_request.get('stream')}")
        else:
            logger.info(f"   Request messages: {len(final_request.get('messages', [])) if 'messages' in final_request else len(final_request.get('contents', []))}")
        
        import json as json_mod
        logger.debug(f"📦 Full upstream request: {json_mod.dumps(final_request, indent=2, ensure_ascii=False)[:1000]}")
        
        if body.stream:
            # Streaming response
            logger.info(f"📡 Starting Anthropic streaming response")
            if "Accept" not in headers:
                headers["Accept"] = "text/event-stream"
            if upstream_service_type in ['openai', 'gemini']:
                final_request["stream"] = True
            
            async def anthropic_stream_generator():
                try:
                    logger.debug(f"🔧 Anthropic stream generator started")
                    logger.debug(f"   Upstream type: {upstream_service_type}")
                    logger.debug(f"   Has function call injection: {has_function_call}")
                    
                    # Anthropic → Anthropic: 直接透传
                    if upstream_service_type == 'anthropic':
                        logger.debug(f"🔧 Direct Anthropic streaming (no format conversion)")
                        async with http_client.stream("POST", upstream_url, json=final_request, headers=headers, timeout=app_config.server.timeout) as response:
                            if response.status_code != 200:
                                error_content = await response.aread()
                                logger.error(f"❌ Upstream error: {response.status_code} - {error_content.decode('utf-8', errors='ignore')}")
                                yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': 'api_error', 'message': 'Upstream service error'}})}\n\n"
                                return
                            
                            # 直接透传 Anthropic SSE 流
                            async for chunk in response.aiter_bytes():
                                yield chunk
                    
                    # Anthropic → OpenAI: 需要格式转换
                    elif upstream_service_type == 'openai':
                        # If function calling is enabled, use the special streaming handler
                        if has_function_call:
                            logger.debug(f"🔧 Using function calling streaming handler (OpenAI)")
                            # Stream through Toolify's FC processor, then convert to Anthropic format
                            openai_stream = stream_proxy_with_fc_transform(
                                upstream_url, 
                                final_request, 
                                headers, 
                                final_request["model"], 
                                True,  # has_fc=True
                                GLOBAL_TRIGGER_SIGNAL,
                                http_client,
                                app_config.server.timeout
                            )
                            # Convert to Anthropic format
                            chunk_count = 0
                            async for anthropic_chunk in stream_openai_to_anthropic(openai_stream):
                                chunk_count += 1
                                if chunk_count <= 5 or chunk_count % 50 == 0:
                                    logger.debug(f"🔧 Yielding Anthropic chunk #{chunk_count}")
                                yield anthropic_chunk.encode('utf-8') if isinstance(anthropic_chunk, str) else anthropic_chunk
                            logger.debug(f"🔧 Function calling stream completed, total chunks: {chunk_count}")
                        else:
                            logger.debug(f"🔧 Direct OpenAI streaming (no FC), will convert to Anthropic")
                            # No function calling, direct streaming with format conversion
                            async with http_client.stream("POST", upstream_url, json=final_request, headers=headers, timeout=app_config.server.timeout) as response:
                                logger.debug(f"🔧 Upstream response status: {response.status_code}")
                                logger.debug(f"🔧 Upstream response headers: {dict(response.headers)}")
                                
                                if response.status_code != 200:
                                    error_content = await response.aread()
                                    logger.error(f"❌ Upstream error: {response.status_code} - {error_content.decode('utf-8', errors='ignore')}")
                                    yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': 'api_error', 'message': 'Upstream service error'}})}\n\n"
                                    return
                                
                                chunk_count = 0
                                async for converted_chunk in stream_openai_to_anthropic(response.aiter_bytes()):
                                    chunk_count += 1
                                    if chunk_count <= 5 or chunk_count % 50 == 0:
                                        logger.debug(f"🔧 Yielding direct chunk #{chunk_count}")
                                    yield converted_chunk.encode('utf-8') if isinstance(converted_chunk, str) else converted_chunk
                                logger.debug(f"🔧 Direct stream completed, total chunks: {chunk_count}")
                            
                except httpx.RemoteProtocolError as e:
                    logger.error(f"❌ Remote protocol error: {e}")
                    logger.error(f"❌ This usually means the upstream closed the connection prematurely")
                    logger.error(traceback.format_exc())
                    yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': 'api_error', 'message': 'Connection closed by upstream'}})}\n\n"
                except Exception as e:
                    logger.error(f"❌ Streaming error: {type(e).__name__}: {e}")
                    logger.error(traceback.format_exc())
                    yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': 'api_error', 'message': str(e)}})}\n\n"
            
            return StreamingResponse(
                anthropic_stream_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
        else:
            # Non-streaming response
            logger.info("📡 Sending non-streaming request to upstream")
            if "Accept" not in headers:
                headers["Accept"] = "application/json"
            
            upstream_response = await http_client.post(
                upstream_url,
                json=final_request,
                headers=headers,
                timeout=app_config.server.timeout
            )
            
            logger.info("📥 UPSTREAM RESPONSE")
            logger.info(f"   Status: {upstream_response.status_code}")
            logger.info(f"   Headers: {dict(upstream_response.headers)}")
            logger.info(f"   Body length: {len(upstream_response.text)} bytes")
            logger.debug(f"   Body (first 1000 chars): {upstream_response.text[:1000]}")
            
            upstream_response.raise_for_status()
            
            upstream_resp = upstream_response.json()
            logger.info(f"✅ Received valid JSON response from upstream")
            logger.info(f"   Response keys: {list(upstream_resp.keys())}")
            
            # 根据上游类型处理响应
            if upstream_service_type == 'anthropic':
                # Anthropic → Anthropic: 直接返回
                logger.info(f"   Strategy: Direct Anthropic response")
                anthropic_resp = upstream_resp
            
            elif upstream_service_type == 'openai':
                # OpenAI → Anthropic: 需要转换
                logger.info(f"   Strategy: Convert OpenAI → Anthropic")
                
                # If function calling was enabled, check for tool calls in response
                if has_function_call:
                    choice = upstream_resp.get("choices", [{}])[0]
                    message = choice.get("message", {})
                    content = message.get("content", "")
                    
                    # Check if response contains function call XML
                    if content and GLOBAL_TRIGGER_SIGNAL in content:
                        logger.debug(f"🔧 Detected function call trigger signal in response")
                        parsed_tools = parse_function_calls_xml(content, GLOBAL_TRIGGER_SIGNAL)
                        
                        if parsed_tools:
                            logger.info(f"🔧 Successfully parsed {len(parsed_tools)} function call(s)")
                            
                            # Convert to OpenAI tool_calls format
                            tool_calls = []
                            for tool in parsed_tools:
                                tool_call_id = f"call_{uuid.uuid4().hex}"
                                store_tool_call_mapping(
                                    tool_call_id,
                                    tool["name"],
                                    tool["args"],
                                    f"Calling tool {tool['name']}"
                                )
                                tool_calls.append({
                                    "id": tool_call_id,
                                    "type": "function",
                                    "function": {
                                        "name": tool["name"],
                                        "arguments": json.dumps(tool["args"])
                                    }
                                })
                            
                            # Update OpenAI response with tool_calls
                            upstream_resp["choices"][0]["message"] = {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": tool_calls
                            }
                            upstream_resp["choices"][0]["finish_reason"] = "tool_calls"
                            logger.debug(f"🔧 Converted XML function calls to OpenAI tool_calls format")
                
                # Convert OpenAI response to Anthropic format
                anthropic_resp = openai_to_anthropic_response(upstream_resp)
            
            elif upstream_service_type == 'gemini':
                # Gemini → Anthropic: 需要转换
                logger.info(f"   Strategy: Convert Gemini → Anthropic")
                from toolify_core.converters import convert_response
                conversion = convert_response("gemini", "anthropic", upstream_resp, body.model)
                if conversion.success:
                    anthropic_resp = conversion.data
                else:
                    logger.error(f"❌ Response conversion failed: {conversion.error}")
                    raise HTTPException(status_code=500, detail=f"Response conversion failed: {conversion.error}")
            
            else:
                raise HTTPException(status_code=500, detail=f"Unsupported upstream type: {upstream_service_type}")
            
            logger.info(f"✅ Converted to Anthropic format")
            logger.info(f"   Response keys: {list(anthropic_resp.keys())}")
            logger.info(f"   Response type: {anthropic_resp.get('type')}")
            logger.info(f"   Response role: {anthropic_resp.get('role')}")
            logger.info(f"   Content blocks: {len(anthropic_resp.get('content', []))}")
            logger.info(f"   Stop reason: {anthropic_resp.get('stop_reason')}")
            logger.debug(f"📦 Full Anthropic response: {str(anthropic_resp)[:1000]}")
            
            elapsed_time = time.time() - start_time
            logger.info("=" * 80)
            logger.info(f"📊 ANTHROPIC API RESPONSE SUMMARY")
            logger.info(f"   Model: {body.model}")
            logger.info(f"   Input Tokens: {anthropic_resp['usage']['input_tokens']}")
            logger.info(f"   Output Tokens: {anthropic_resp['usage']['output_tokens']}")
            logger.info(f"   Duration: {elapsed_time:.2f}s")
            logger.info("=" * 80)
            
            return JSONResponse(content=anthropic_resp)
            
    except httpx.HTTPStatusError as e:
        logger.error(f"❌ Upstream HTTP error: {e.response.status_code}")
        error_detail = e.response.text
        return JSONResponse(
            status_code=e.response.status_code,
            content={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": f"Upstream service error: {error_detail}"
                }
            }
        )
    except Exception as e:
        logger.error(f"❌ Error processing Anthropic request: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": str(e)
                }
            }
        )


@app.get("/v1/models")
async def list_models(_api_key: str = Depends(verify_api_key)):
    """List all available models."""
    ensure_config_loaded()
    visible_models = set()
    for model_name in MODEL_TO_SERVICE_MAPPING.keys():
        if ':' in model_name:
            parts = model_name.split(':', 1)
            if len(parts) == 2:
                alias, _ = parts
                visible_models.add(alias)
            else:
                visible_models.add(model_name)
        else:
            visible_models.add(model_name)

    models = []
    for model_id in sorted(visible_models):
        models.append({
            "id": model_id,
            "object": "model",
            "created": 1677610602,
            "owned_by": "openai",
            "permission": [],
            "root": model_id,
            "parent": None
        })
    
    return {
        "object": "list",
        "data": models
    }


# Admin API endpoints
@app.post("/api/admin/login", response_model=LoginResponse)
async def admin_login(login_data: LoginRequest, admin_config=Depends(get_admin_credentials)):
    """Admin login endpoint."""
    if login_data.username != admin_config.username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )

    if not verify_password(login_data.password, admin_config.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )

    access_token = create_access_token(admin_config.username, admin_config.jwt_secret)
    return LoginResponse(access_token=access_token)


@app.get("/api/admin/config")
async def get_config(_username: str = Depends(verify_admin_token)):
    """Get current configuration."""
    try:
        with open(config_loader.config_path, 'r', encoding='utf-8') as f:
            config_content = yaml.safe_load(f)

        # Remove sensitive information from response
        if 'admin_authentication' in config_content:
            config_content['admin_authentication'] = {
                'username': config_content['admin_authentication'].get('username', ''),
                'password': '********',
                'jwt_secret': '********'
            }

        return {"success": True, "config": config_content}
    except Exception as e:
        logger.error(f"Failed to read config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read configuration: {str(e)}")


@app.put("/api/admin/config")
async def update_config(config_data: dict, _username: str = Depends(verify_admin_token)):
    """Update configuration."""
    try:
        # Read current config to preserve admin_authentication
        current_config = None
        try:
            with open(config_loader.config_path, 'r', encoding='utf-8') as f:
                current_config = yaml.safe_load(f)
        except Exception:
            pass

        # If admin_authentication exists in current config and not provided in update, preserve it
        if current_config and 'admin_authentication' in current_config:
            if 'admin_authentication' not in config_data or config_data['admin_authentication'].get(
                    'password') == '********':
                config_data['admin_authentication'] = current_config['admin_authentication']

        # Validate the configuration using Pydantic
        from config_loader import AppConfig
        validated_config = AppConfig(**config_data)

        # Write the validated configuration back to file
        with open(config_loader.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        # Reload runtime configuration so changes take effect immediately
        load_runtime_config(reload=True)
        logger.info(f"Configuration updated and reloaded successfully by admin: {_username}")
        return {
            "success": True,
            "message": "Configuration updated successfully and applied immediately."
        }

    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise HTTPException(status_code=400, detail=f"Configuration validation failed: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to update config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update configuration: {str(e)}")


# Gemini API key extraction
async def verify_gemini_api_key(request: Request) -> str:
    """Verify Gemini API key from query parameter or header"""
    logger.debug(f"🔐 Gemini API Key Verification")
    
    client_key = None
    
    # Priority 1: URL parameter ?key=
    api_key = request.query_params.get("key")
    if api_key:
        logger.debug(f"   Using URL parameter key: ***{api_key[-8:] if len(api_key) > 8 else '***'}")
        client_key = api_key
    
    # Priority 2: x-goog-api-key header
    elif request.headers.get("x-goog-api-key"):
        x_goog_key = request.headers.get("x-goog-api-key")
        logger.debug(f"   Using x-goog-api-key header: ***{x_goog_key[-8:]}")
        client_key = x_goog_key
    
    # Priority 3: Authorization Bearer
    elif request.headers.get("authorization"):
        authorization = request.headers.get("authorization")
        if authorization.startswith("Bearer "):
            client_key = authorization[7:]
            logger.debug(f"   Using Authorization Bearer: ***{client_key[-8:]}")
        else:
            client_key = authorization
    
    # Check if key was found
    if not client_key:
        logger.error("❌ Missing Gemini API key")
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide via 'key' parameter, 'x-goog-api-key', or 'Authorization' header"
        )
    
    # Validate key if not in passthrough mode
    if not app_config.features.key_passthrough:
        logger.debug(f"   Validating against {len(ALLOWED_CLIENT_KEYS)} allowed keys")
        logger.debug(f"   Allowed keys: {[f'***{k[-8:]}' for k in ALLOWED_CLIENT_KEYS]}")
        
        if client_key not in ALLOWED_CLIENT_KEYS:
            logger.error(f"❌ Unauthorized Gemini key: ***{client_key[-8:]}")
            raise HTTPException(status_code=401, detail="Unauthorized API key")
    else:
        logger.debug(f"   Mode: Key passthrough (validation skipped)")
    
    logger.debug(f"✅ Gemini key validated successfully")
    return client_key


# Gemini API unified endpoint (支持多种路径格式和流式/非流式)
@app.post("/v1beta/models/{model_path:path}")
@app.post("/v1/models/{model_path:path}")
@app.post("/v1/v1beta/models/{model_path:path}")  # 兼容性路由
@app.post("/v1/v1beta/{model_path:path}")  # 缺少models/的路径
async def gemini_stream_generate_content(
    model_path: str,
    request: Request,
    _api_key: str = Depends(verify_gemini_api_key)
):
    """Gemini API endpoint (支持 generateContent 和 streamGenerateContent)"""
    ensure_config_loaded()
    
    # 解析模型ID和操作类型
    import urllib.parse
    model_path = urllib.parse.unquote(model_path)
    
    # 提取模型ID和操作（:generateContent 或 :streamGenerateContent）
    if ":streamGenerateContent" in model_path:
        model_id = model_path.replace(":streamGenerateContent", "")
        is_streaming = True
        operation = "streamGenerateContent"
    elif ":generateContent" in model_path:
        model_id = model_path.replace(":generateContent", "")
        is_streaming = False
        operation = "generateContent"
    else:
        # 可能没有操作后缀，根据URL参数判断
        model_id = model_path
        is_streaming = "alt=sse" in str(request.url)
        operation = "streamGenerateContent" if is_streaming else "generateContent"
    
    logger.info("=" * 80)
    logger.info(f"🤖 GEMINI API REQUEST ({'STREAMING' if is_streaming else 'NON-STREAMING'})")
    logger.info("=" * 80)
    logger.info(f"   Original URL: {request.url}")
    logger.info(f"   Model path: {model_path}")
    logger.info(f"   Model ID (cleaned): {model_id}")
    logger.info(f"   Operation: {operation}")
    logger.info(f"   Is streaming: {is_streaming}")
    logger.info(f"   API Key: ***{_api_key[-8:] if len(_api_key) > 8 else '***'}")
    logger.info(f"   Query params: {dict(request.query_params)}")
    
    try:
        request_data = await request.json()
        request_data["model"] = model_id
        request_data["stream"] = is_streaming
        
        logger.info(f"   Contents: {len(request_data.get('contents', []))} items")
        logger.info(f"   System instruction: {'Yes' if request_data.get('systemInstruction') else 'No'}")
        logger.info(f"   Tools: {len(request_data.get('tools', []))} tools")
        logger.info("=" * 80)
        
        # Find upstream for this model
        upstreams, actual_model = find_upstream(
            model_id,
            MODEL_TO_SERVICE_MAPPING,
            ALIAS_MAPPING,
            DEFAULT_SERVICE,
            app_config.features.model_passthrough,
            app_config.upstream_services
        )
        
        upstream = upstreams[0]
        
        logger.info(f"🎯 Selected upstream: {upstream['name']} (type: {upstream.get('service_type')})")
        
        # Handle format conversion if needed
        if upstream.get("service_type") != "gemini":
            # Convert Gemini request to target format
            from toolify_core.converters import convert_request
            conversion = convert_request("gemini", upstream["service_type"], request_data)
            if not conversion.success:
                raise HTTPException(status_code=400, detail=f"Conversion failed: {conversion.error}")
            request_data = conversion.data
        
        # Prepare upstream URL and headers based on service type and streaming mode
        if upstream.get("service_type") == "gemini":
            # Gemini service
            if is_streaming:
                url = f"{upstream['base_url']}/models/{actual_model}:streamGenerateContent?alt=sse&key={upstream['api_key']}"
                headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
            else:
                url = f"{upstream['base_url']}/models/{actual_model}:generateContent?key={upstream['api_key']}"
                headers = {"Content-Type": "application/json"}
        
        elif upstream.get("service_type") == "openai":
            # OpenAI service
            url = f"{upstream['base_url']}/chat/completions"
            headers = {
                "Authorization": f"Bearer {upstream['api_key']}",
                "Content-Type": "application/json"
            }
            if is_streaming:
                headers["Accept"] = "text/event-stream"
                request_data["stream"] = True
            else:
                request_data["stream"] = False
        
        elif upstream.get("service_type") == "anthropic":
            # Anthropic service
            url = f"{upstream['base_url']}/v1/messages"
            headers = {
                "x-api-key": upstream['api_key'],
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            }
            if is_streaming:
                headers["Accept"] = "text/event-stream"
                request_data["stream"] = True
            else:
                request_data["stream"] = False
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported service type: {upstream.get('service_type')}")
        
        logger.info(f"🚀 {'Streaming' if is_streaming else 'Sending'} to: {url}")
        
        # Handle based on streaming flag
        if not is_streaming:
            # Non-streaming request
            logger.info("📤 DETAILED REQUEST TO UPSTREAM")
            logger.info(f"   URL: {url}")
            logger.info(f"   Headers: {dict((k, '***' + v[-8:] if k.lower() == 'authorization' else v) for k, v in headers.items())}")
            logger.info(f"   Body keys: {list(request_data.keys())}")
            logger.info(f"   Body.model: {request_data.get('model')}")
            logger.info(f"   Body.messages: {request_data.get('messages')}")
            logger.info(f"   Body.max_tokens: {request_data.get('max_tokens')}")
            logger.info(f"   Body.stream: {request_data.get('stream')}")
            
            import json as json_module
            logger.debug(f"📦 Complete request body:")
            logger.debug(json_module.dumps(request_data, indent=2, ensure_ascii=False))
            
            async with httpx.AsyncClient(timeout=app_config.server.timeout) as client:
                response = await client.post(url, json=request_data, headers=headers)
                
                logger.info(f"📥 Upstream response: {response.status_code}")
                logger.info(f"   Response headers: {dict(response.headers)}")
                logger.info(f"   Response body length: {len(response.text)} bytes")
                logger.debug(f"   Response body (first 500 chars): {response.text[:500]}")
                
                if response.status_code != 200:
                    error_content = response.text
                    logger.error(f"❌ Upstream error: {response.status_code}")
                    logger.error(f"   Error body: {error_content}")
                    logger.error(f"   Sent URL: {url}")
                    logger.error(f"   Sent headers: {dict((k, '***' if 'key' in k.lower() or 'auth' in k.lower() else v) for k, v in headers.items())}")
                    logger.error(f"   Sent body: {json_module.dumps(request_data, indent=2, ensure_ascii=False)[:1000]}")
                    raise HTTPException(status_code=response.status_code, detail=error_content)
                
                # Parse response JSON with better error handling
                try:
                    response_data = response.json()
                    logger.info(f"✅ Parsed response JSON successfully")
                    logger.info(f"   Response keys: {list(response_data.keys())}")
                except json_module.JSONDecodeError as e:
                    logger.error(f"❌ Failed to parse response JSON: {e}")
                    logger.error(f"   Response text: {response.text[:1000]}")
                    logger.error(f"   Response content-type: {response.headers.get('content-type')}")
                    raise HTTPException(status_code=500, detail=f"Invalid JSON response from upstream: {e}")
                
                # Convert response back if needed
                if upstream.get("service_type") != "gemini":
                    logger.info(f"🔄 Converting response from {upstream.get('service_type')} to Gemini format")
                    from toolify_core.converters import convert_response
                    conversion = convert_response(upstream["service_type"], "gemini", response_data, model_id)
                    if conversion.success:
                        logger.info(f"✅ Conversion successful")
                        response_data = conversion.data
                    else:
                        logger.error(f"❌ Response conversion failed: {conversion.error}")
                        # Return original data as fallback
                
                logger.debug(f"📦 Final response: {json_module.dumps(response_data, indent=2, ensure_ascii=False)[:500]}")
                return JSONResponse(content=response_data)
        
        # Streaming response
        async def gemini_stream_generator():
            async with httpx.AsyncClient(timeout=app_config.server.timeout) as client:
                async with client.stream("POST", url, json=request_data, headers=headers) as response:
                    logger.info(f"📥 Upstream response: {response.status_code}")
                    
                    if response.status_code != 200:
                        error_content = await response.aread()
                        logger.error(f"❌ Upstream error: {response.status_code} - {error_content.decode('utf-8', errors='ignore')}")
                        # Send Gemini error format
                        yield f"data: {json.dumps({'error': {'message': 'Upstream error', 'status': response.status_code}})}\n\n"
                        return
                    
                    # Direct passthrough or convert based on upstream type
                    if upstream.get("service_type") == "gemini":
                        # Direct passthrough
                        async for chunk in response.aiter_bytes():
                            if chunk:
                                yield chunk
                    else:
                        # Need to convert from OpenAI/Anthropic to Gemini format
                        # For now, simple passthrough (TODO: implement conversion)
                        async for chunk in response.aiter_bytes():
                            if chunk:
                                yield chunk
        
        return StreamingResponse(
            gemini_stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Gemini streaming API error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# Capability detection endpoints
@app.post("/api/detect/capabilities")
async def detect_capabilities(
    request: Request,
    _username: str = Depends(verify_admin_token)
):
    """
    Detect AI provider capabilities.
    Request body: {"provider": "openai|anthropic|gemini", "api_key": "...", "base_url": "...", "model": "..."}
    """
    try:
        data = await request.json()
        provider = data.get("provider")
        api_key = data.get("api_key")
        base_url = data.get("base_url")
        model = data.get("model")
        
        if not provider or not api_key:
            raise HTTPException(status_code=400, detail="Missing provider or api_key")
        
        # Create detector
        detector = DetectorFactory.create_detector(provider, api_key, base_url, model)
        if not detector:
            raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")
        
        # Run detection
        logger.info(f"Starting capability detection for {provider}")
        report = await detector.detect_all_capabilities()
        
        return JSONResponse(content=report.to_dict())
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Capability detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/detect/providers")
async def list_supported_providers(_username: str = Depends(verify_admin_token)):
    """List supported providers for capability detection"""
    return {
        "providers": DetectorFactory.get_supported_providers(),
        "formats": ConverterFactory.get_supported_formats()
    }


# Mount static files for admin interface (if exists)
try:
    if os.path.exists("frontend/dist"):
        app.mount("/admin", StaticFiles(directory="frontend/dist", html=True), name="admin")
        logger.info("📁 Admin interface mounted at /admin")
except Exception as e:
    logger.warning(f"⚠️  Failed to mount admin interface: {e}")


@app.on_event("startup")
async def startup_event():
    """Startup event - load configuration"""
    global _config_loaded
    if not _config_loaded:
        try:
            load_runtime_config()
            _config_loaded = True
            logger.info("✅ Configuration loaded on server startup")
        except Exception as e:
            logger.error(f"❌ FATAL: Configuration loading failed")
            logger.error(f"❌ Error: {type(e).__name__}: {str(e)}")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event - cleanup"""
    logger.info("👋 Server shutting down...")
    await http_client.aclose()


def main():
    """
    Main entry point for running the server.
    This function can be called directly from PyCharm or command line.
    """
    import uvicorn
    import sys
    
    # Setup basic logging for startup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    print("=" * 80)
    print("🚀 Toolify - LLM Function Calling Middleware")
    print("=" * 80)
    
    # Try to load config to get server settings
    try:
        load_runtime_config()
        host = app_config.server.host
        port = app_config.server.port
        timeout = app_config.server.timeout
        log_level = app_config.features.log_level.lower() if app_config.features.log_level != "DISABLED" else "critical"
        
        print(f"📋 Configuration loaded from: {config_loader.config_path}")
        print(f"🌐 Server will start on: http://{host}:{port}")
        print(f"⏱️  Request timeout: {timeout} seconds")
        print(f"📊 Upstream services: {len(app_config.upstream_services)}")
        print(f"🔑 Client keys configured: {len(app_config.client_authentication.allowed_keys)}")
        print(f"📝 Log level: {app_config.features.log_level}")
        print("=" * 80)
        print()
        
    except FileNotFoundError:
        print("❌ ERROR: config.yaml not found!")
        print("💡 Please copy config.example.yaml to config.yaml and configure it")
        print("=" * 80)
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: Failed to load configuration: {type(e).__name__}")
        print(f"   Details: {str(e)}")
        print("💡 Please check your config.yaml file for syntax errors")
        print("=" * 80)
        sys.exit(1)
    
    # Start the server
    try:
        logger.info(f"🚀 Starting Uvicorn server on {host}:{port}")
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level=log_level,
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("👋 Server stopped by user")
        print("=" * 80)
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ Server crashed: {type(e).__name__}")
        print(f"   Details: {str(e)}")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()

