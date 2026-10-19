"""Official Vanillux2Agent with request-local Agent Fleet transport headers.

Prompts, bash execution, stopping behavior and observations remain inherited.
Do not patch the shared official source used by Phase1.
"""
import asyncio
import ipaddress
import json
import logging
import os
from urllib.parse import urlsplit

import httpx
import litellm
from openai import OpenAI
from sandbox_model_transport import SandboxModelTransport
from Vanillux2Agent.agent import (
    Vanillux2Agent as OfficialVanillux2Agent,
    ABORT_EXCEPTIONS, MAX_RETRIES, RETRY_BASE_DELAY,
    LLM_TIMEOUT_SECONDS, LLM_OUTER_TIMEOUT_BUFFER_SECONDS, TOOL_SCHEMAS,
)


def sandbox_model_api_base(api_base, headers):
    """Keep the gateway intact: only the sandbox may open the model socket."""
    # Trusted launcher configuration, not the request's api_base or X-Backend.
    # Preserve the historical gateway when no explicit override is configured.
    gateway = os.environ.get("TMAX_MODEL_GATEWAY", "http://10.100.184.46")
    parsed = urlsplit(gateway)
    if (parsed.scheme != "http" or parsed.username is not None
            or parsed.password is not None or parsed.path not in ("", "/")
            or parsed.query or parsed.fragment):
        raise ValueError("TMAX_MODEL_GATEWAY must be a credential-free HTTP origin")
    ipaddress.IPv4Address(parsed.hostname)
    if parsed.port is not None and not 0 < parsed.port < 65536:
        raise ValueError("TMAX_MODEL_GATEWAY requires a valid port")
    expected_base = gateway.rstrip("/") + "/v1"
    if (api_base or "").rstrip("/") != expected_base:
        raise ValueError("TMax sandbox model transport requires the configured gateway")
    backend = headers.get("X-Backend", "")
    host, separator, port = backend.rpartition(":")
    if not separator or not port.isdigit() or not 0 < int(port) < 65536:
        raise ValueError("Sandbox model transport requires a valid X-Backend IP:port")
    ipaddress.IPv4Address(host)
    return api_base.rstrip('/')


class Vanillux2Agent(OfficialVanillux2Agent):
    async def run(self, instruction, environment, context):
        self._model_environment = environment
        try:
            return await super().run(instruction, environment, context)
        finally:
            self._model_environment = None

    async def _query_with_retry(self, model, messages):
        headers = json.loads(os.environ.get("TMAX_MODEL_HEADERS_JSON", "{}"))
        if not headers.get("X-Session-Id"):
            raise ValueError("Vanillux rollout requires an Agent Fleet session header")
        configured_base = (self.api_base or os.environ.get("OPENAI_BASE_URL")
                           or os.environ.get("OPENAI_API_BASE"))
        api_base = sandbox_model_api_base(configured_base, headers)
        environment = getattr(self, '_model_environment', None)
        if environment is None:
            raise RuntimeError('Sandbox model transport requires the active task environment')
        if not getattr(self, "_transport_logged", False):
            logging.getLogger(__name__).info(
                "tmax_model_transport origin=opensandbox api_base=%s session_id=%s",
                api_base, headers["X-Session-Id"],
            )
            self._transport_logged = True
        kwargs = dict(
            model=model, messages=messages, tools=TOOL_SCHEMAS,
            max_tokens=self.max_tokens,
            api_base=api_base,
            timeout=LLM_TIMEOUT_SECONDS, request_timeout=LLM_TIMEOUT_SECONDS,
            num_retries=0, extra_headers=headers,
        )
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None and not (
            model.startswith("anthropic/") and self.temperature is not None
        ):
            kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            kwargs["extra_body"] = {"top_k": self.top_k}
        transport = SandboxModelTransport(environment, asyncio.get_running_loop(),
            api_base=api_base, timeout=LLM_TIMEOUT_SECONDS, session_id=headers['X-Session-Id'])
        # Keep SDK serialization, response parsing, HTTP error mapping and the
        # official retry loop. Only replace the socket-opening transport.
        with OpenAI(api_key=os.environ.get('OPENAI_API_KEY') or 'local-no-auth',
                    base_url=api_base, max_retries=0, timeout=LLM_TIMEOUT_SECONDS,
                    http_client=httpx.Client(transport=transport, trust_env=False)) as client:
            kwargs['client'] = client
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    return await asyncio.wait_for(
                        asyncio.to_thread(litellm.completion, **kwargs),
                        timeout=LLM_TIMEOUT_SECONDS + LLM_OUTER_TIMEOUT_BUFFER_SECONDS,
                    )
                except ABORT_EXCEPTIONS:
                    raise
                except Exception:
                    if attempt == MAX_RETRIES:
                        raise
                    await asyncio.sleep(RETRY_BASE_DELAY * 2 ** (attempt - 1))
