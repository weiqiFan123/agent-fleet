"""HTTPX transport for non-streaming model calls through a Harbor environment.

Only environment.exec runs on the host; the HTTP socket is opened in the
task's sandbox. No daemon, image changes, dependency install, or S3 writes.
"""
import asyncio
import base64
import gzip
import hashlib
import json
import logging
import time
import uuid

import httpx

_CHUNK_SIZE = 48000  # Below Linux's single shell-argument limit, even after wrapping.
_REMOTE_HTTP = r'''
import base64, gzip, json, urllib.request, urllib.error, http.client, time, sys, socket
rpc = json.loads(gzip.decompress(base64.b64decode(encoded)))
timing = dict(schema=1, scope='sandbox_http', request_id=rpc.get('request_id'),
    session_id=rpc.get('session_id'), started_unix_s=time.time(),
    connect_s=None, connect_ok=False, request_send_s=None,
    response_headers_wait_s=None, response_headers_received=False, response_read_s=None,
    upstream_connect_s=None, sglang_queue_s=None, sglang_inference_s=None,
    tcp_maxseg_requested=rpc.get('tcp_maxseg', 0), tcp_maxseg_effective=None)
http_started = time.monotonic()
def model_connection(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None):
    # TMax validates an IPv4 gateway. Clamp this socket before connect(), so
    # large request/response bodies survive the observed gateway path's MSS
    # black hole. This does not split/rewrite HTTP bodies or change timeouts.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
            sock.settimeout(timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG, rpc['tcp_maxseg'])
        if source_address:
            sock.bind(source_address)
        sock.connect(address)
        return sock
    except BaseException:
        sock.close()
        raise
class TimedHTTPConnection(http.client.HTTPConnection):
    # The fixed gateway is HTTP. connect_s includes DNS if a hostname is used;
    # it does NOT measure nginx's connection to its upstream.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if rpc.get('tcp_maxseg', 0):
            self._create_connection = model_connection
    def connect(self):
        timing['phase'] = 'sandbox_to_gateway_connect'
        started = time.monotonic()
        try:
            result = super().connect()
            timing['connect_ok'] = True
            timing['tcp_maxseg_effective'] = self.sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG)
            timing['phase'] = 'request_send'
            return result
        finally:
            timing['connect_s'] = time.monotonic() - started
    def request(self, *args, **kwargs):
        started = time.monotonic()
        try:
            return super().request(*args, **kwargs)
        finally:
            if timing['connect_ok']:
                timing['request_send_s'] = max(0., time.monotonic() - started - timing['connect_s'])
    def getresponse(self):
        timing['phase'] = 'response_headers_wait'
        started = time.monotonic()
        try:
            response = super().getresponse()
            timing['response_headers_received'] = True
            return response
        finally:
            timing['response_headers_wait_s'] = time.monotonic() - started
class TimedHTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req):
        return self.do_open(TimedHTTPConnection, req)
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect(), TimedHTTPHandler())
headers = rpc['headers']
headers['accept-encoding'] = 'identity'
# Correlation only; do not change the request body, model/session headers or retry policy.
if rpc.get('request_id'):
    headers['X-TMax-Request-Id'] = rpc['request_id']
request = urllib.request.Request(rpc['url'], method=rpc['method'], headers=headers,
    data=base64.b64decode(rpc['body']) if rpc['method'] != 'GET' else None)
try:
    try:
        response = opener.open(request, timeout=rpc['timeout'])
    except urllib.error.HTTPError as error:
        response = error
    with response:
        timing['status'] = response.code
        timing['phase'] = 'response_read'
        started = time.monotonic()
        try:
            body = response.read()
        finally:
            timing['response_read_s'] = time.monotonic() - started
        result = dict(status=response.code, headers=dict(response.headers.items()),
                      body=base64.b64encode(body).decode())
    timing['phase'] = 'completed'
    print('SANDBOX_MODEL_HTTP_V1:' + base64.b64encode(gzip.compress(json.dumps(result).encode())).decode())
except Exception as error:
    timing['error_type'] = type(error).__name__
    timing['cause_type'] = type(getattr(error, 'reason', None)).__name__
    raise
finally:
    timing['http_total_s'] = time.monotonic() - http_started
    timing['finished_unix_s'] = time.time()
    # Keep stdout a SINGLE response frame: execd may concatenate stdout events
    # without preserving line boundaries. Telemetry must not corrupt base64.
    # No URLs, headers, prompts, response text or exception messages in telemetry.
    try:
        print('SANDBOX_MODEL_TIMING_V1:' + json.dumps(timing), file=sys.stderr, flush=True)
    except Exception:
        pass  # Observability must not fail an otherwise valid model response.
'''


class SandboxModelTransport(httpx.BaseTransport):
    def __init__(self, environment, loop, *, api_base, timeout, session_id, tcp_maxseg=1024):
        if type(tcp_maxseg) is not int or (tcp_maxseg != 0 and not 536 <= tcp_maxseg <= 32767):
            raise ValueError('tcp_maxseg must be 0 (system default) or 536..32767')
        self.environment = environment
        self.loop = loop
        self.api_base = api_base.rstrip('/')
        self.timeout = timeout
        self.session_id = session_id
        self.tcp_maxseg = tcp_maxseg

    def handle_request(self, request):
        # Called in the SDK's worker thread, never in the environment event loop.
        if str(request.url) not in (self.api_base + '/chat/completions', self.api_base + '/models'):
            raise httpx.UnsupportedProtocol('sandbox model transport rejected unexpected endpoint', request=request)
        body = request.read()
        if body and json.loads(body).get('stream'):
            raise httpx.UnsupportedProtocol('sandbox model transport requires non-streaming requests', request=request)
        request_id = 'tmax-http-' + uuid.uuid4().hex
        trace = dict(schema=1, scope='cpu_transport', request_id=request_id,
                     session_id=self.session_id, started_unix_s=time.time(),
                     request_bytes=len(body), body_sha256=hashlib.sha256(body).hexdigest())
        self._timing(dict(trace, event='started'))
        rpc = dict(request_id=request_id, session_id=self.session_id, url=str(request.url), method=request.method, headers=dict(request.headers),
                   body=base64.b64encode(body).decode(), timeout=self.timeout,
                   tcp_maxseg=self.tcp_maxseg)
        encoded = base64.b64encode(gzip.compress(json.dumps(rpc).encode())).decode()
        started = time.monotonic()
        future = asyncio.run_coroutine_threadsafe(self._execute(encoded, trace), self.loop)
        try:
            result = future.result(timeout=self.timeout + 120)
            trace['status'] = result['status']
        except TimeoutError as error:
            trace['error_type'] = type(error).__name__
            future.cancel()
            raise httpx.ReadTimeout('sandbox model request timed out', request=request) from error
        except Exception as error:
            trace['error_type'] = type(error).__name__
            raise httpx.ConnectError('sandbox model command failed: ' + type(error).__name__, request=request) from error
        finally:
            self._timing(dict(trace, event='finished', finished_unix_s=time.time(),
                              transport_total_s=time.monotonic() - started))
        logging.getLogger(__name__).info(
            'sandbox_model_http origin=opensandbox session_id=%s status=%s elapsed_sec=%.3f request_bytes=%s',
            self.session_id, result['status'], time.monotonic() - started, len(body))
        return httpx.Response(result['status'], headers=result['headers'],
                              content=base64.b64decode(result['body'], validate=True), request=request)

    @staticmethod
    def _timing(record):
        # Observability must not turn a valid response into an infra failure.
        try:
            logging.getLogger(__name__).info('sandbox_model_timing %s', json.dumps(record, sort_keys=True))
        except Exception:
            pass

    async def _exec(self, code, timeout):
        result = await self.environment.exec(command="python3 - <<'PY'\n" + code + '\nPY',
                                             timeout_sec=int(timeout), cwd='/')
        # Parse timing even on helper failure. Keep the existing failure mapping.
        # execd may concatenate stderr events without newlines, including the
        # traceback after our JSON. Decode only the structured timing object.
        stderr = getattr(result, 'stderr', '') or ''
        for part in stderr.split('SANDBOX_MODEL_TIMING_V1:')[1:]:
            try:
                record, _ = json.JSONDecoder().raw_decode(part.lstrip())
                if isinstance(record, dict):
                    self._timing(record)
            except (ValueError, TypeError):
                pass
        if result.return_code != 0:
            # Do not echo request/header contents or shell commands into host exceptions.
            raise RuntimeError(f'sandbox model helper exit={result.return_code}; inspect sandbox exec logs')
        return result.stdout

    async def _execute(self, encoded, trace=None):
        trace = trace if trace is not None else {}
        trace['upload_chunks_completed'] = 0
        upload_started = time.monotonic()
        if len(encoded) <= _CHUNK_SIZE:
            prelude = 'encoded = ' + repr(encoded) + '\n'
        else:
            # Long task histories must not become one oversized shell argument.
            # Keep request-local files for inspection; no sandbox/log cleanup.
            directory = '/tmp/harbor-model-rpc-' + uuid.uuid4().hex
            await self._exec(f'import os; os.mkdir({directory!r}, 0o700)', 30)
            count = 0
            for index in range(0, len(encoded), _CHUNK_SIZE):
                chunk = encoded[index:index + _CHUNK_SIZE]
                path = f'{directory}/{count:05d}'
                await self._exec(f'from pathlib import Path; Path({path!r}).write_text({chunk!r})', 30)
                count += 1
                trace['upload_chunks_completed'] = count
            prelude = ('from pathlib import Path\n'
                       f'encoded = "".join((Path({directory!r}) / f"{{i:05d}}").read_text() for i in range({count}))\n')
        trace['request_staging_s'] = time.monotonic() - upload_started
        trace['staged_request_chars'] = len(encoded)
        helper_started = time.monotonic()
        try:
            output = await self._exec(prelude + _REMOTE_HTTP, self.timeout + 30)
        finally:
            trace['helper_exec_s'] = time.monotonic() - helper_started
        # helper_exec_s contains remote HTTP plus exec submission/polling/fetch.
        # It is not a separate additive phase alongside http_total_s.
        frames = [line.removeprefix('SANDBOX_MODEL_HTTP_V1:') for line in output.splitlines()
                  if line.startswith('SANDBOX_MODEL_HTTP_V1:')]
        if len(frames) != 1:
            raise RuntimeError('sandbox model response missing or truncated; refusing partial trajectory')
        return json.loads(gzip.decompress(base64.b64decode(frames[0], validate=True)))
