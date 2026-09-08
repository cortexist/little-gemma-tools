#!/usr/bin/env python3
"""Text Chat Completions adapter for a dedicated little-gemma UNIX socket."""
import argparse
import codecs
import json
import os
import secrets
import socket
import struct
import threading
import time
import uuid

from flask import Flask, Response, jsonify, request
from werkzeug.exceptions import HTTPException


class APIError(Exception):
    def __init__(self, message, status=400, code='invalid_request_error'):
        self.message, self.status, self.code = message, status, code


def error_body(message, code):
    return {'error': {'message': message, 'type': code, 'param': None, 'code': code}}


def encode_messages(messages, max_bytes):
    if not isinstance(messages, list) or not messages:
        raise APIError('messages must be a nonempty array')
    parts = []
    previous = 'user'  # The socket supplies the initial user turn opener.
    for i, message in enumerate(messages):
        if not isinstance(message, dict) or set(message)-{'role', 'content'}:
            raise APIError('Messages support role and text content only')
        role, content = message.get('role'), message.get('content')
        if role not in ('system', 'developer', 'user', 'assistant'):
            raise APIError('Supported roles: system, developer, user, assistant')
        if isinstance(content, list):
            if not all(isinstance(p, dict) and set(p)=={'type','text'} and p['type']=='text'
                       and isinstance(p['text'], str) for p in content):
                raise APIError('Only text content parts are supported')
            content = ''.join(p['text'] for p in content)
        if not isinstance(content, str):
            raise APIError('Message content must be text')
        if '\0' in content or '<|' in content or '|>' in content:
            raise APIError('Message content contains reserved engine protocol delimiters')
        mapped = {'assistant': 'model', 'developer': 'system'}.get(role, role)
        if i or mapped != previous:
            parts.append('<turn|>\n<|turn>'+mapped+'\n')
        parts.append(content)
    if messages[-1]['role'] != 'user':
        raise APIError('The last message must have role user')
    payload = ''.join(parts).encode('utf-8')
    if len(payload) > max_bytes:
        raise APIError('Message history exceeds adapter byte budget; shorten messages', 400, 'context_length_exceeded')
    return payload


class Reply:
    """Incremental UTF-8/protocol parser; never expose the thought channel."""
    markers = ('<|channel>', '<channel|>', '<turn|>', '<eos>', ' [SERVE_GEN cap]')

    def __init__(self, stops):
        self.decoder = codecs.getincrementaldecoder('utf-8')()
        self.buffer = ''
        self.hidden = False
        self.done = False
        self.reason = 'stop'
        self.stops = stops

    def feed(self, raw):
        self.buffer += self.decoder.decode(raw)
        output = []
        while self.buffer and not self.done:
            markers = self.markers + (() if self.hidden else tuple(self.stops))
            hits = [(self.buffer.find(m), m) for m in markers if m in self.buffer]
            if hits:
                offset, marker = min(hits, key=lambda h: h[0])
                if not self.hidden: output.append(self.buffer[:offset])
                self.buffer = self.buffer[offset+len(marker):]
                if marker == '<|channel>': self.hidden = True
                elif marker == '<channel|>': self.hidden = False
                elif marker == ' [SERVE_GEN cap]': self.reason = 'length'
                else: self.done = True
            else:
                # Keep any suffix that could become a delimiter in the next recv.
                keep = max((n for m in markers for n in range(1, min(len(m), len(self.buffer))+1)
                            if self.buffer.endswith(m[:n])), default=0)
                emit = self.buffer[:-keep] if keep else self.buffer
                self.buffer = self.buffer[-keep:] if keep else ''
                if not self.hidden: output.append(emit)
                break
        return ''.join(output)


def create_app(socket_path, model='little-gemma', api_key=None, timeout=120, max_bytes=3500):
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 128 * 1024
    gate = threading.Lock()

    @app.errorhandler(APIError)
    def api_error(exc):
        return jsonify(error_body(exc.message, exc.code)), exc.status

    @app.errorhandler(HTTPException)
    def http_error(exc):
        return jsonify(error_body(exc.description, 'invalid_request_error')), exc.code

    @app.before_request
    def authorize():
        if api_key and not secrets.compare_digest(request.headers.get('Authorization', ''), 'Bearer '+api_key):
            raise APIError('Invalid API key', 401, 'authentication_error')

    @app.get('/health')
    def health():
        return jsonify(status='ok', socket_exists=os.path.exists(socket_path), busy=gate.locked())

    @app.get('/v1/models')
    def models():
        return jsonify(object='list', data=[dict(id=model, object='model', created=0, owned_by='little-gemma')])

    @app.post('/v1/chat/completions')
    def completions():
        data = request.get_json()
        if not isinstance(data, dict): raise APIError('Expected a JSON object')
        supported = {'model', 'messages', 'stream', 'stop', 'n', 'stream_options'}
        unsupported = [k for k, v in data.items() if k not in supported and v is not None]
        if unsupported: raise APIError('Unsupported parameters: '+', '.join(unsupported))
        if data.get('model') != model: raise APIError('Unknown model; use '+model, 404, 'model_not_found')
        if type(data.get('n', 1)) is not int or data.get('n', 1) != 1: raise APIError('Only n=1 is supported')
        if type(data.get('stream', False)) is not bool: raise APIError('stream must be boolean')
        options = data.get('stream_options')
        if options not in (None, {}, {'include_usage': False}):
            raise APIError('Token usage is unavailable from this socket protocol')
        stops = data.get('stop')
        if stops is None: stops = []
        if isinstance(stops, str): stops = [stops]
        if not isinstance(stops, list) or len(stops)>4 or any(not isinstance(s, str) or not s or len(s)>256 for s in stops):
            raise APIError('stop must contain one to four nonempty strings of at most 256 characters')
        payload = encode_messages(data.get('messages'), max_bytes)
        if not gate.acquire(blocking=False): raise APIError('Engine busy; retry after the active request', 429, 'rate_limit_error')
        conn = socket.socket(socket.AF_UNIX)
        conn.settimeout(timeout)
        closed = False
        def close():
            nonlocal closed
            if not closed:
                closed = True
                conn.close()
                gate.release()
        try:
            conn.connect(socket_path)
            if payload:
                conn.sendall(struct.pack('<BBHHI', 1, ord('T'), 0, 0, len(payload))+payload)
            conn.sendall(b'\n')
        except OSError:
            close()
            raise APIError('Could not connect or write to the engine socket', 503, 'service_unavailable')
        ident, created = 'chatcmpl-'+uuid.uuid4().hex, int(time.time())
        parser = Reply(stops)
        def fragments():
            deadline = time.monotonic()+timeout
            while not parser.done:
                remaining = deadline-time.monotonic()
                if remaining<=0: raise APIError('Engine response timed out', 504, 'timeout_error')
                conn.settimeout(remaining)
                raw = conn.recv(4096)
                if not raw: raise APIError('Engine closed before end of turn (check its context budget/log)', 502, 'upstream_error')
                text = parser.feed(raw)
                if text: yield text
        def chunk(delta, finish=None):
            return dict(id=ident, object='chat.completion.chunk', created=created, model=model,
                        choices=[dict(index=0, delta=delta, finish_reason=finish)])
        def sse(value): return 'data: '+json.dumps(value, ensure_ascii=False)+'\n\n'
        if data.get('stream'):
            def stream():
                try:
                    yield sse(chunk({'role':'assistant', 'content':''}))
                    for text in fragments(): yield sse(chunk({'content':text}))
                    yield sse(chunk({}, parser.reason))
                    yield 'data: [DONE]\n\n'
                except (OSError, UnicodeError, APIError) as exc:
                    message = exc.message if isinstance(exc, APIError) else 'Engine stream failed'
                    yield sse(error_body(message, 'upstream_error'))
                finally: close()
            response = Response(stream(), mimetype='text/event-stream', headers={'Cache-Control':'no-cache', 'X-Accel-Buffering':'no'})
            # Also release if the HTTP server closes before iterating the body.
            response.call_on_close(close)
            return response
        try:
            content = ''.join(fragments())
            return jsonify(id=ident, object='chat.completion', created=created, model=model,
                           choices=[dict(index=0, message=dict(role='assistant', content=content), finish_reason=parser.reason)])
        except socket.timeout:
            raise APIError('Engine response timed out', 504, 'timeout_error')
        except (OSError, UnicodeError):
            raise APIError('Engine response failed', 502, 'upstream_error')
        finally: close()
    return app


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--socket', required=True, help='Dedicated engine UNIX socket')
    p.add_argument('--model', default='little-gemma', help='Model ID exposed to clients')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8082)
    p.add_argument('--timeout', type=float, default=120)
    p.add_argument('--max-input-bytes', type=int, default=3500)
    args = p.parse_args()
    if args.timeout<=0 or not 1<=args.max_input_bytes<=3500: p.error('timeout must be positive; input budget must be 1–3500')
    app = create_app(args.socket, args.model, os.environ.get('LG_API_KEY'), args.timeout, args.max_input_bytes)
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == '__main__': main()
