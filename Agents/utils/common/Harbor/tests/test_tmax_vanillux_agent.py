import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock

from tmax_vanillux_agent import Vanillux2Agent, OfficialVanillux2Agent, TOOL_SCHEMAS, sandbox_model_api_base


class VanilluxTransportTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop('TMAX_MODEL_GATEWAY', None)

    def test_sandbox_transport_keeps_gateway_and_backend_header(self):
        self.assertEqual(sandbox_model_api_base('http://10.100.184.46/v1',
                         {'X-Backend': '172.17.231.3:18083'}),
                         'http://10.100.184.46/v1')
        self.assertEqual(sandbox_model_api_base('http://10.100.184.46/v1/',
                         {'X-Backend': '172.17.231.4:18083'}),
                         'http://10.100.184.46/v1')

    def test_direct_host_endpoint_is_rejected(self):
        with self.assertRaises(ValueError):
            sandbox_model_api_base('http://172.17.231.3:18083/v1', {})

    def test_explicit_gateway_override(self):
        with patch.dict(os.environ, TMAX_MODEL_GATEWAY='http://117.135.59.6'):
            self.assertEqual(sandbox_model_api_base('http://117.135.59.6/v1/',
                             {'X-Backend': '10.100.0.147:18083'}),
                             'http://117.135.59.6/v1')
            for base in ('http://10.100.184.46/v1', 'http://10.100.0.147:18083/v1',
                         'http://117.135.59.6/v1?x=1', 'http://117.135.59.6/v1/other'):
                with self.subTest(base=base), self.assertRaises(ValueError):
                    sandbox_model_api_base(base, {'X-Backend': '10.100.0.147:18083'})
            with self.assertRaises(ValueError):
                sandbox_model_api_base('http://117.135.59.6/v1', {})

    def test_gateway_origin_validation(self):
        for gateway in ('', 'https://117.135.59.6', 'http://user:pass@117.135.59.6',
                        'http://117.135.59.6/v1', 'http://117.135.59.6?x=1',
                        'http://117.135.59.6/#x', 'http://example.invalid',
                        'http://117.135.59.6:0', 'http://117.135.59.6:65536'):
            with self.subTest(gateway=gateway), patch.dict(os.environ, TMAX_MODEL_GATEWAY=gateway):
                with self.assertRaises(ValueError):
                    sandbox_model_api_base(gateway.rstrip('/') + '/v1',
                                           {'X-Backend': '10.100.0.147:18083'})

    def test_gateway_requires_valid_backend(self):
        for backend in ('', 'example.invalid:18083', '172.17.231.3:0',
                        '172.17.231.3:65536', '172.17.231.3:18083/path'):
            with self.subTest(backend=backend), self.assertRaises(ValueError):
                sandbox_model_api_base('http://10.100.184.46/v1', {'X-Backend': backend})

    def test_preserves_official_harness(self):
        for method in ('setup', '_execute_bash', '_format_tool_result'):
            self.assertIs(getattr(Vanillux2Agent, method), getattr(OfficialVanillux2Agent, method))

    def test_run_delegates_unchanged_to_official_harness(self):
        with tempfile.TemporaryDirectory() as folder:
            agent = Vanillux2Agent(logs_dir=Path(folder), model_name='openai/default')
            environment, context = object(), object()
            with patch.object(OfficialVanillux2Agent, 'run', new_callable=AsyncMock) as run:
                asyncio.run(agent.run('instruction', environment, context))
                run.assert_awaited_once_with('instruction', environment, context)
            self.assertIsNone(agent._model_environment)

    def test_session_headers_and_sampling_reach_model(self):
        with tempfile.TemporaryDirectory() as folder:
            agent = Vanillux2Agent(logs_dir=Path(folder), model_name='openai/default',
                                  api_base='http://10.100.184.46/v1', temperature=1.,
                                  top_p=.95, max_tokens=16384, max_steps=64)
            headers = {'X-Backend': '172.17.231.3:18083', 'X-Session-Id': 'test-session',
                       'Proxy-X-Session-Id': 'test-session'}
            agent._model_environment = object()
            with patch.dict(os.environ, TMAX_MODEL_HEADERS_JSON=json.dumps(headers)), \
                 patch('tmax_vanillux_agent.litellm.completion', return_value='response') as query:
                result = asyncio.run(agent._query_with_retry('openai/default', [{'role': 'user', 'content': 'test'}]))
            self.assertEqual(result, 'response')
            self.assertEqual(query.call_args.kwargs['extra_headers'], headers)
            self.assertEqual(query.call_args.kwargs['api_base'], 'http://10.100.184.46/v1')
            self.assertEqual(type(query.call_args.kwargs['client']._client._transport).__name__, 'SandboxModelTransport')
            self.assertEqual(query.call_args.kwargs['tools'], TOOL_SCHEMAS)
            self.assertEqual(query.call_args.kwargs['max_tokens'], 16384)
            self.assertEqual(query.call_args.kwargs['temperature'], 1.)

    def test_missing_session_fails_before_model_call(self):
        with tempfile.TemporaryDirectory() as folder:
            agent = Vanillux2Agent(logs_dir=Path(folder), model_name='openai/default')
            with patch.dict(os.environ, TMAX_MODEL_HEADERS_JSON='{}'), \
                 patch('tmax_vanillux_agent.litellm.completion') as query:
                with self.assertRaisesRegex(ValueError, 'session header'):
                    asyncio.run(agent._query_with_retry('openai/default', []))
                query.assert_not_called()


if __name__ == '__main__':
    unittest.main()
