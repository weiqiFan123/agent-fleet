"""Exercise urllib/http.client itself, mocking only the sandbox socket."""
import asyncio
import base64
import contextlib
import gzip
import io
import json
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from sandbox_model_transport import _REMOTE_HTTP, SandboxModelTransport


class DelayedBody(io.BytesIO):
    def read(self, *args):
        time.sleep(.015)
        return super().read(*args)


class Socket:
    def __init__(self, status=200, empty=False):
        body = b'{"choices":[],"usage":{"total_tokens":3}}'
        self.body = body
        self.sent = []
        self.raw = b'' if empty else (
            f'HTTP/1.1 {status} test\r\nContent-Length: {len(body)}\r\n'
            'Content-Type: application/json\r\nConnection: close\r\n\r\n'.encode() + body)

    def sendall(self, data):
        self.sent.append(bytes(data))

    def setsockopt(self, *args):
        pass

    def getsockopt(self, *args):
        return 1024

    def settimeout(self, *args):
        pass

    def bind(self, *args):
        pass

    def connect(self, *args):
        pass

    def makefile(self, *args, **kwargs):
        time.sleep(.025)
        return DelayedBody(self.raw)

    def close(self):
        pass


class SocketEnvironment:
    def __init__(self, socket=None, failure=None, join_output_lines=False):
        self.socket = socket or Socket()
        self.failure = failure
        self.join_output_lines = join_output_lines

    def connect(self, *args, **kwargs):
        time.sleep(.015)
        if self.failure:
            raise self.failure
        return self.socket

    async def exec(self, command, **kwargs):
        source = command.split('\n', 1)[1].rsplit('\nPY', 1)[0]
        output = io.StringIO()
        errors = io.StringIO()
        code = 0
        # The production MSS path creates the socket before connect(). Keep
        # the fake on both connection paths instead of opening a real socket.
        with patch('socket.create_connection', side_effect=self.connect), \
                patch('socket.socket', return_value=self.socket), \
                patch.object(self.socket, 'connect', side_effect=self.connect), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            try:
                exec(compile(source, '<sandbox-helper>', 'exec'), {})
            except Exception:
                code = 1
        stdout, stderr = output.getvalue(), errors.getvalue()
        if self.join_output_lines:
            stdout = ''.join(stdout.splitlines())
            stderr = ''.join(stderr.splitlines())
        return SimpleNamespace(return_code=code, stdout=stdout, stderr=stderr)


class TimingTests(unittest.TestCase):
    def test_stdout_line_coalescing_preserves_response_and_http_status(self):
        # Execd stdout events may be concatenated without line separators.
        # Telemetry must never become a suffix of the base64 response frame.
        for status in (200, 504):
            with self.subTest(status=status):
                env = SocketEnvironment(Socket(status=status), join_output_lines=True)
                run, records = self.request(env)
                response = asyncio.run(run())
                self.assertEqual(response.status_code, status)
                self.assertEqual(response.content, env.socket.body)

    def request(self, env):
        records = []
        async def run():
            transport = SandboxModelTransport(env, asyncio.get_running_loop(),
                api_base='http://10.100.184.46/v1', timeout=1, session_id='session-test')
            transport._timing = records.append
            with httpx.Client(transport=transport) as client:
                return await asyncio.to_thread(client.post,
                    'http://10.100.184.46/v1/chat/completions',
                    headers={'Authorization': 'secret-for-test', 'X-Backend': '127.0.0.1:18083'},
                    content=b'{"messages":[{"content":"private-prompt"}]}')
        return run, records

    def test_separate_connect_headers_and_body_preserve_response(self):
        env = SocketEnvironment()
        run, records = self.request(env)
        response = asyncio.run(run())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, env.socket.body)
        remote = next(r for r in records if r['scope'] == 'sandbox_http')
        self.assertTrue(remote['connect_ok'])
        self.assertGreaterEqual(remote['connect_s'], .014)
        self.assertGreaterEqual(remote['response_headers_wait_s'], .024)
        self.assertGreaterEqual(remote['response_read_s'], .014)
        self.assertEqual(remote['phase'], 'completed')
        self.assertIsNone(remote['sglang_inference_s'])
        self.assertIsNone(remote['upstream_connect_s'])
        host = records[-1]
        self.assertEqual(host['request_id'], remote['request_id'])
        self.assertGreaterEqual(host['helper_exec_s'], remote['http_total_s'])
        sent = b''.join(env.socket.sent)
        self.assertIn(remote['request_id'].encode(), sent)
        self.assertTrue(sent.endswith(b'{"messages":[{"content":"private-prompt"}]}'))
        self.assertNotIn('private-prompt', json.dumps(records))
        self.assertNotIn('secret-for-test', json.dumps(records))

    def test_504_is_preserved_not_reclassified_as_connect_failure(self):
        run, records = self.request(SocketEnvironment(Socket(status=504)))
        response = asyncio.run(run())
        self.assertEqual(response.status_code, 504)
        remote = next(r for r in records if r['scope'] == 'sandbox_http')
        self.assertTrue(remote['connect_ok'])
        self.assertTrue(remote['response_headers_received'])
        self.assertEqual(remote['status'], 504)

    def test_connect_failure_records_failed_phase_without_false_inference(self):
        run, records = self.request(SocketEnvironment(failure=TimeoutError('private-error')))
        with self.assertRaises(httpx.ConnectError):
            asyncio.run(run())
        remote = next(r for r in records if r['scope'] == 'sandbox_http')
        self.assertFalse(remote['connect_ok'])
        self.assertGreaterEqual(remote['connect_s'], .014)
        self.assertEqual(remote['phase'], 'sandbox_to_gateway_connect')
        self.assertIsNone(remote['response_headers_wait_s'])
        self.assertNotIn('private-error', json.dumps(records))
        self.assertIn('error_type', records[-1])

    def test_peer_disconnect_distinguished_from_connect_failure(self):
        run, records = self.request(SocketEnvironment(Socket(empty=True)))
        with self.assertRaises(httpx.ConnectError):
            asyncio.run(run())
        remote = next(r for r in records if r['scope'] == 'sandbox_http')
        self.assertTrue(remote['connect_ok'])
        self.assertFalse(remote['response_headers_received'])
        self.assertEqual(remote['phase'], 'response_headers_wait')

    def test_logger_failure_is_nonfatal(self):
        with patch('logging.Logger.info', side_effect=RuntimeError('disk full')):
            SandboxModelTransport._timing({'scope': 'test'})


if __name__ == '__main__':
    unittest.main()
