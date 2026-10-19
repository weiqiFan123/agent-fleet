"""Execute the actual remote Python helper with a mocked sandbox HTTP socket."""
import asyncio
import base64
import contextlib
import io
import gzip
import json
import os
from pathlib import Path
import tempfile
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error

import httpx
import litellm
from sandbox_model_transport import SandboxModelTransport, _REMOTE_HTTP
from tmax_vanillux_agent import Vanillux2Agent


class Environment:
    def __init__(self, root, *, status=200, body=None, corrupt=False):
        self.root, self.status, self.corrupt = root, status, corrupt
        self.body = body or {'id': 'test', 'object': 'chat.completion', 'created': 1,
            'model': 'default', 'choices': [{'index': 0, 'finish_reason': 'tool_calls',
            'message': {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 'call_1', 'type': 'function', 'function': {'name': 'bash', 'arguments': '{"command":"true"}'}}]}}],
            'usage': {'prompt_tokens': 12, 'completion_tokens': 5, 'total_tokens': 17}}
        self.calls, self.requests = [], []

    def open(self, request, timeout):
        self.requests.append(request)
        response = io.BytesIO(json.dumps(self.body).encode())
        response.code = self.status
        response.headers = {'content-type': 'application/json'}
        if self.status >= 400:
            raise urllib.error.HTTPError(request.full_url, self.status, 'test', response.headers, response)
        return response

    async def exec(self, command, **kwargs):
        self.calls.append(command)
        self.assert_safe(command)
        source = command.split('\n', 1)[1].rsplit('\nPY', 1)[0]
        source = source.replace('/tmp/harbor-model-rpc-', self.root + '/rpc-')
        output = io.StringIO()
        with patch('urllib.request.build_opener', return_value=self) as opener, contextlib.redirect_stdout(output):
            exec(compile(source, '<sandbox-helper>', 'exec'), {})
            if opener.called:
                # Sandbox requests bypass proxy env, not CPU requests.
                assert opener.call_args.args[0].proxies == {}
        return SimpleNamespace(return_code=0, stdout='truncated' if self.corrupt else output.getvalue(), stderr='')

    def assert_safe(self, command):
        assert len(command.encode()) < 60000, len(command)


class SandboxTransportTests(unittest.TestCase):
    def test_mss_is_set_before_connect_and_large_http_bytes_are_unchanged(self):
        sent = ('request-中文-' * 8000).encode()
        received = b'response-' * 16000
        observations = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                observations.append(self.rfile.read(int(self.headers['Content-Length'])))
                self.send_response(200)
                self.send_header('Content-Length', str(len(received)))
                self.end_headers()
                self.wfile.write(received)
        events = []
        class TrackedSocket(socket.socket):
            def setsockopt(self, level, option, value, *args):
                if level == socket.IPPROTO_TCP and option == socket.TCP_MAXSEG:
                    events.append(('mss', value))
                return super().setsockopt(level, option, value, *args)
            def connect(self, address):
                events.append(('connect', address))
                return super().connect(address)
        with HTTPServer(('127.0.0.1', 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                rpc = dict(url=f'http://127.0.0.1:{server.server_port}/test', method='POST',
                           headers={'Content-Length': str(len(sent))},
                           body=base64.b64encode(sent).decode(), timeout=5, tcp_maxseg=1024)
                encoded = base64.b64encode(gzip.compress(json.dumps(rpc).encode())).decode()
                stdout, stderr = io.StringIO(), io.StringIO()
                with patch('socket.socket', TrackedSocket), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    exec(compile(_REMOTE_HTTP, '<sandbox-helper>', 'exec'), {'encoded': encoded})
                result = json.loads(gzip.decompress(base64.b64decode(stdout.getvalue().strip().split(':', 1)[1])))
                self.assertEqual(observations, [sent])
                self.assertEqual(base64.b64decode(result['body']), received)
                self.assertEqual(events[0], ('mss', 1024))
                self.assertEqual(events[1][0], 'connect')
                timing = json.loads(stderr.getvalue().strip().split(':', 1)[1])
                self.assertEqual(timing['tcp_maxseg_requested'], 1024)
                self.assertGreater(timing['tcp_maxseg_effective'], 0)
                self.assertLessEqual(timing['tcp_maxseg_effective'], 1024)
            finally:
                server.shutdown()
                thread.join(timeout=5)

    def test_error_timing_survives_execd_joined_stderr(self):
        class FailedEnvironment:
            async def exec(self, *args, **kwargs):
                return SimpleNamespace(return_code=1, stdout='', stderr=
                    'SANDBOX_MODEL_TIMING_V1:{"phase":"response_headers_wait","error_type":"TimeoutError"}Traceback: redacted')
        async def run():
            transport = SandboxModelTransport(FailedEnvironment(), asyncio.get_running_loop(),
                api_base='http://127.0.0.1/v1', timeout=5, session_id='test')
            with patch.object(transport, '_timing') as timing:
                with self.assertRaises(RuntimeError):
                    await transport._exec('pass', 5)
                timing.assert_called_once_with({'phase':'response_headers_wait','error_type':'TimeoutError'})
        asyncio.run(run())

    def test_configured_nginx_uses_sandbox_not_cpu_socket(self):
        with tempfile.TemporaryDirectory() as root:
            env = Environment(root)
            agent = Vanillux2Agent(logs_dir=Path(root), model_name='openai/default',
                api_base='http://117.135.59.6/v1', temperature=1., top_p=.95, max_tokens=16384)
            agent._model_environment = env
            headers = {'X-Backend': '10.100.0.147:18083', 'X-Session-Id': 'new-gateway-test',
                       'Proxy-X-Session-Id': 'new-gateway-test'}
            with patch.dict(os.environ, TMAX_MODEL_GATEWAY='http://117.135.59.6',
                            TMAX_MODEL_HEADERS_JSON=json.dumps(headers)), \
                 patch.object(httpx.HTTPTransport, 'handle_request', side_effect=AssertionError('CPU MODEL SOCKET')):
                result = asyncio.run(agent._query_with_retry('openai/default', [{'role': 'user', 'content': 'test'}]))
            self.assertEqual(len(env.requests), 1)
            request = env.requests[0]
            self.assertEqual(request.full_url, 'http://117.135.59.6/v1/chat/completions')
            actual = {k.lower(): v for k, v in request.header_items()}
            for k, v in headers.items():
                self.assertEqual(actual[k.lower()], v)
            body = json.loads(request.data)
            self.assertEqual((body['model'], body['temperature'], body['top_p'], body['max_tokens']),
                             ('default', 1., .95, 16384))
            self.assertEqual(result.choices[0].message.tool_calls[0].function.name, 'bash')

    def test_real_sdk_serialization_and_response_preserved(self):
        with tempfile.TemporaryDirectory() as root:
            env = Environment(root)
            agent = Vanillux2Agent(logs_dir=Path(root), model_name='openai/default',
                api_base='http://10.100.184.46/v1', temperature=1., top_p=.95, max_tokens=16384)
            agent._model_environment = env
            headers = {'X-Backend': '172.18.111.19:18083', 'X-Session-Id': 'session-test',
                       'Proxy-X-Session-Id': 'session-test'}
            with patch.dict(os.environ, TMAX_MODEL_HEADERS_JSON=json.dumps(headers), HTTP_PROXY='http://invalid:1'), \
                 patch.object(httpx.HTTPTransport, 'handle_request', side_effect=AssertionError('CPU MODEL SOCKET')):
                result = asyncio.run(agent._query_with_retry('openai/default', [{'role': 'user', 'content': 'test'}]))
            self.assertEqual(len(env.requests), 1)
            request = env.requests[0]
            self.assertEqual(request.full_url, 'http://10.100.184.46/v1/chat/completions')
            sent_headers = {k.lower(): v for k,v in request.header_items()}
            for k,v in headers.items(): self.assertEqual(sent_headers[k.lower()], v)
            body = json.loads(request.data)
            self.assertEqual((body['model'],body['temperature'],body['top_p'],body['max_tokens']), ('default',1.,.95,16384))
            self.assertEqual(result.usage.total_tokens, 17)
            self.assertEqual(result.choices[0].message.tool_calls[0].function.name, 'bash')

    def test_context_limit_remains_official_abort_exception(self):
        with tempfile.TemporaryDirectory() as root:
            env = Environment(root, status=400, body={'error': {'message': "This model's maximum context length is 65536 tokens. However, you requested 67585 tokens.",
                'type': 'invalid_request_error', 'code': 'context_length_exceeded'}})
            agent = Vanillux2Agent(logs_dir=Path(root), model_name='openai/default', api_base='http://10.100.184.46/v1')
            agent._model_environment = env
            with patch.dict(os.environ, TMAX_MODEL_HEADERS_JSON=json.dumps({'X-Backend':'172.18.111.19:18083','X-Session-Id':'s'})):
                with self.assertRaises(litellm.exceptions.ContextWindowExceededError):
                    asyncio.run(agent._query_with_retry('openai/default', [{'role':'user','content':'test'}]))
            self.assertEqual(len(env.requests), 1)

    def test_generic_bad_request_matches_unmodified_sdk(self):
        # Error-code alone does not trigger LiteLLM's context-window classifier.
        # Compare transports instead of imposing new retry/abort semantics.
        from openai import OpenAI
        body = {'error': {'message': 'maximum context length exceeded',
            'type': 'invalid_request_error', 'code': 'context_length_exceeded'}}
        def query(client):
            return litellm.completion(model='openai/default',
                messages=[{'role': 'user', 'content': 'test'}],
                api_base='http://10.100.184.46/v1', client=client, num_retries=0)
        with tempfile.TemporaryDirectory() as root:
            env = Environment(root, status=400, body=body)
            with OpenAI(api_key='fake', base_url='http://10.100.184.46/v1', max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(
                    lambda request: httpx.Response(400, json=body, request=request)))) as client:
                with self.assertRaises(litellm.exceptions.BadRequestError) as baseline:
                    query(client)
            async def run():
                transport = SandboxModelTransport(env, asyncio.get_running_loop(),
                    api_base='http://10.100.184.46/v1', timeout=10, session_id='s')
                with OpenAI(api_key='fake', base_url='http://10.100.184.46/v1', max_retries=0,
                    http_client=httpx.Client(transport=transport, trust_env=False)) as client:
                    return await asyncio.to_thread(query, client)
            with self.assertRaises(type(baseline.exception)) as actual:
                asyncio.run(run())
            self.assertEqual(actual.exception.status_code, baseline.exception.status_code)
            self.assertEqual(len(env.requests), 1)

    def test_long_incompressible_request_is_chunked_and_lossless(self):
        with tempfile.TemporaryDirectory() as root:
            env = Environment(root)
            text = os.urandom(180000).hex()
            async def run():
                transport = SandboxModelTransport(env, asyncio.get_running_loop(), api_base='http://10.100.184.46/v1', timeout=10, session_id='s')
                with httpx.Client(transport=transport) as client:
                    return await asyncio.to_thread(client.post, 'http://10.100.184.46/v1/chat/completions', json={'messages':[{'role':'user','content':text}]})
            result = asyncio.run(run())
            self.assertEqual(result.status_code, 200)
            self.assertGreater(len(env.calls), 3)
            self.assertEqual(json.loads(env.requests[0].data)['messages'][0]['content'], text)

    def test_corrupt_response_fails_not_empty_success(self):
        with tempfile.TemporaryDirectory() as root:
            env = Environment(root, corrupt=True)
            async def run():
                transport = SandboxModelTransport(env, asyncio.get_running_loop(), api_base='http://10.100.184.46/v1', timeout=10, session_id='s')
                await asyncio.to_thread(transport.handle_request, httpx.Request('GET','http://10.100.184.46/v1/models'))
            with self.assertRaises(httpx.ConnectError): asyncio.run(run())


if __name__ == '__main__':
    unittest.main()
