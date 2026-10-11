#!/usr/bin/env python3
"""Shared monitor/control page for an explicit collection of voice nodes.

The hub coordinates session budgets and optional host recording. Models and
conversation microphones belong to the nodes. Node keys stay on the server.
"""
import argparse
import json
from pathlib import Path
import re
import secrets
import signal
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from flask import Flask, jsonify, render_template, request
from server import HERE, load_visemes, PHONEME_TO_VISEME
from speaker_collections import load_collection
from session_scenario import ClassroomScenario
from recording import Recording, command
from recording_audio import DigitalMix
from recording_input_tracks import InputTracks
from trio_direct import DirectTrioSession as TrioSession
from trio_latency import TrioLatency

ACTIONS = {'status', 'start', 'stop', 'prompt', 'volume', 'microphone', 'listen'}


def create_app(config):
    app = Flask(__name__, template_folder=str(HERE), static_folder=str(HERE))
    app.config['MAX_CONTENT_LENGTH'] = 65536
    collection = load_collection(config)
    scenario = ClassroomScenario()
    nodes = {}
    for entry in config['nodes']:
        node = dict(entry)
        node_id = node['id']
        if not re.fullmatch(r'[a-z][a-z0-9-]{0,31}', node_id) or node_id in nodes:
            raise ValueError('Nodes need distinct, simple IDs')
        url = urllib.parse.urlsplit(node['url'])
        if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password or url.query or url.fragment or url.path not in ('', '/'):
            raise ValueError('Node URL must be an HTTP(S) origin')
        speaker = collection.speaker(node['speaker_id'])
        if not speaker or not node.get('key'):
            raise ValueError('Each node needs a collection speaker and device key')
        if scenario.participants[node['speaker_id']]['node'] != node_id:
            raise ValueError('Dashboard node and classroom speaker assignment disagree')
        node['speaker'] = speaker
        node['default_prompt'] = scenario.prompt(node['speaker_id'])
        nodes[node_id] = node
    if not 1 <= len(nodes) <= 8 or len({n['speaker_id'] for n in nodes.values()}) != len(nodes):
        raise ValueError('Expected 1–8 distinct speakers')

    def upstream(node, action, data=None):
        url = node['url'].rstrip('/')+'/api/local/'+action
        if action == 'status':
            url += '?after='+str(max(0, int((data or {}).get('after', 0))))
        body = None if action == 'status' else json.dumps(data or {}).encode('utf-8')
        req = urllib.request.Request(url, data=body, headers={
            'X-Voice-Key': node['key'], 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=25 if action == 'stop' else 3) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raise ValueError('Device rejected the request (HTTP '+str(exc.code)+')') from None
        except (OSError, ValueError) as exc:
            raise ValueError('Cannot reach '+node['id']+'; check its node service') from exc

    def checked_status(node, data=None):
        result = upstream(node, 'status', data)
        speaker = result.get('speaker') or {}
        if speaker.get('collection_id') != collection.id or speaker.get('speaker_id') != node['speaker_id']:
            raise ValueError('Device identity does not match this panel; controls are unavailable')
        return result

    def recording_models():
        state = session.snapshot()
        if state['active'] and state.get('mode') == 'passive' and state['phase'] == 'waiting':
            return ['passive']*len(nodes)
        return list(state['profiles'].values()) if state['active'] and state['phase'] == 'running' else None

    def recording_microphone():
        source = config['source']
        base = source.rsplit('.', 1)[0]+'.'
        sources = json.loads(command(['pactl', '-f', 'json', 'list', 'sources']))
        matches = [s['name'] for s in sources if s['name'] == source or s['name'].startswith(base)]
        if len(matches) != 1:
            raise ValueError('Recording microphone unavailable or ambiguous')
        return matches[0]

    recording = Recording(config, recording_models, recording_microphone,
                          digital_factory=lambda: DigitalMix(nodes.values()),
                          input_factory=lambda: InputTracks(nodes.values(), session.snapshot()['id']))
    recording.available = recording.available and bool(config.get('runtime_dir') and
                           (config.get('recording_audio_source') or (config.get('source') and config.get('sink'))))
    recording.update(available=recording.available)
    session = TrioSession(nodes, upstream, checked_status, recording, TrioLatency(config.get('runtime_dir')))
    readiness_lock = threading.Lock()
    readiness = dict(at=0, blockers=['Checking nodes…'])

    def session_status():
        state = session.snapshot()
        if not state['active']:
            with readiness_lock:
                if time.monotonic()-readiness['at'] > 3:
                    _, blockers = session.preflight()
                    readiness.update(at=time.monotonic(), blockers=blockers)
        return dict(**state, blockers=[] if state['active'] else readiness['blockers'], recording=recording.snapshot())

    @app.route('/api/conversation', methods=['GET', 'POST', 'DELETE'])
    def conversation_api():
        try:
            if request.method == 'POST':
                if not request.is_json: return jsonify(error='JSON required'), 415
                session.start(request.get_json())
            elif request.method == 'DELETE':
                session.stop()
            return jsonify(session_status())
        except ValueError as exc:
            return jsonify(error=str(exc)), 409

    @app.route('/api/recording', methods=['GET', 'POST', 'DELETE'])
    def recording_api():
        try:
            if request.method == 'POST':
                data = request.get_json(silent=True)
                if not isinstance(data, dict): raise ValueError('JSON object required')
                if not recording.available:
                    raise ValueError('Recording is not configured on the dashboard host')
                return jsonify(recording.arm(data.get('audio_mode', 'room')))
            if request.method == 'DELETE': return jsonify(recording.stop())
            return jsonify(recording.snapshot())
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

    @app.before_request
    def guard():
        if request.path.startswith('/api/'):
            if not secrets.compare_digest(request.headers.get('X-Voice-Key', ''), config['key']):
                return jsonify(error='invalid dashboard key'), 403
            origin = request.headers.get('Origin')
            if origin and origin != request.host_url.rstrip('/'):
                return jsonify(error='cross-origin control is disabled'), 403

    @app.get('/')
    def index():
        public = [dict(id=n['id'], speaker=n['speaker'], default_prompt=n['default_prompt']) for n in nodes.values()]
        return render_template('index.html', key=config['key'], visemes=load_visemes(), phonemes=PHONEME_TO_VISEME,
                               name='Voice trio', peer=False, role='', capture=False, default_prompt='', nodes=public)

    @app.route('/api/<node_id>/<action>', methods=['GET', 'POST'])
    def node_api(node_id, action):
        if node_id not in nodes or action not in ACTIONS:
            return jsonify(error='unknown node or action'), 404
        if (action == 'status') != (request.method == 'GET'):
            return jsonify(error='GET status; POST controls'), 405
        if request.method == 'POST' and not request.is_json:
            return jsonify(error='JSON required'), 415
        data = request.args if request.method == 'GET' else request.get_json()
        if request.method == 'POST' and not isinstance(data, dict):
            return jsonify(error='JSON object required'), 400
        try:
            node = nodes[node_id]
            if action == 'status':
                return jsonify(checked_status(node, data))
            if session.snapshot()['active'] and action in ('start', 'prompt', 'listen', 'stop'):
                if action == 'stop': return jsonify(session.stop())
                return jsonify(error='Use the session controls while a class is running'), 409
            # Verify the identity again before a mutation, including after
            # reconnect/reconfiguration. Never control an accidentally reused port.
            checked_status(node, {'after': 9223372036854775807})
            if action == 'start':
                data = dict(data)
                data.pop('peer_speaker_id', None)
                data.pop('peer_collection_id', None)
                data['listen'] = False  # Model load is not permission to start a room conversation.
            return jsonify(upstream(node, action, data))
        except (ValueError, TypeError) as exc:
            return jsonify(error=str(exc)), 502

    app.session = session
    app.recording = recording
    app.shutdown = session.shutdown
    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8097)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    import logging
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    app = create_app(config)
    def shutdown(*_):
        app.shutdown()
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    try:
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    finally:
        app.shutdown()
