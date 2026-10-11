"""Multi-node routing, identity checks and isolation from classroom control."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'script/deviceui'), str(ROOT/'script')]
import dashboard
import server


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.cfg = dict(key='hub-key', speaker_collection_id='voice-trio', nodes=[
            dict(id=node, speaker_id=speaker, url='http://'+node+':8097', key=node+'-private-key')
            for node, speaker in [('nowhere', 'agent-smith'), ('cortex1', 'chey'), ('cortex', 'catherine-byers')]])
        self.app = dashboard.create_app(self.cfg)
        self.client = self.app.test_client()
        self.headers = {'X-Voice-Key': 'hub-key'}

    def status(self, speaker):
        return dict(speaker=dict(collection_id='voice-trio', speaker_id=speaker), state={'ready':True})

    def response(self, data):
        return io.BytesIO(json.dumps(data).encode())

    def test_page_lists_three_speakers_without_remote_credentials(self):
        page = self.client.get('/').text
        self.assertIn('Three voices. One classroom.', page)
        for node in self.cfg['nodes']:
            self.assertIn(node['id'], page)
            self.assertNotIn(node['key'], page)
            self.assertNotIn(node['url'], page)

    def test_recording_can_begin_during_passive_listening(self):
        self.app.session.update(active=True, mode='passive', phase='loading')
        self.assertIsNone(self.app.recording.models())
        self.app.session.update(phase='waiting')
        self.assertEqual(self.app.recording.models(), ['passive']*3)
        self.app.session.update(active=False)
        self.assertIsNone(self.app.recording.models())

    def test_auth_and_method_checks_apply_to_every_node(self):
        for node in self.cfg['nodes']:
            path = '/api/'+node['id']+'/volume'
            self.assertEqual(self.client.post(path, json={}).status_code, 403)
            self.assertEqual(self.client.get(path, headers=self.headers).status_code, 405)
            self.assertEqual(self.client.post(path, json={}, headers=dict(self.headers, Origin='http://other')).status_code, 403)
        self.assertEqual(self.client.get('/api/unconfigured/status', headers=self.headers).status_code, 404)
        self.assertEqual(self.client.post('/api/nowhere/status', json={}, headers=self.headers).status_code, 405)
        self.assertEqual(self.client.post('/api/nowhere/volume', json=[], headers=self.headers).status_code, 400)

    def test_routes_each_panel_to_its_own_endpoint_and_key(self):
        for node in self.cfg['nodes']:
            with patch.object(dashboard.urllib.request, 'urlopen', return_value=self.response(self.status(node['speaker_id']))) as call:
                result = self.client.get('/api/'+node['id']+'/status?after=42', headers=self.headers)
                self.assertEqual(result.status_code, 200)
                req = call.call_args.args[0]
                self.assertEqual(req.full_url, node['url']+'/api/local/status?after=42')
                self.assertEqual(req.get_header('X-voice-key'), node['key'])

    def test_one_failed_node_does_not_block_other_panels(self):
        def response(req, **kw):
            if 'cortex1:' in req.full_url:
                raise OSError('offline')
            return self.response(self.status('catherine-byers'))
        with patch.object(dashboard.urllib.request, 'urlopen', side_effect=response):
            self.assertEqual(self.client.get('/api/cortex1/status', headers=self.headers).status_code, 502)
            self.assertEqual(self.client.get('/api/cortex/status', headers=self.headers).status_code, 200)

    def test_identity_mismatch_blocks_controls_before_mutation(self):
        with patch.object(dashboard.urllib.request, 'urlopen', return_value=self.response(self.status('chey'))) as call:
            result = self.client.post('/api/cortex/volume', json={'percent':50}, headers=self.headers)
            self.assertEqual(result.status_code, 502)
            self.assertIn('identity', result.json['error'])
            self.assertEqual(call.call_count, 1)
            self.assertEqual(call.call_args.args[0].get_method(), 'GET')

    def test_loading_does_not_inject_roster_or_enable_microphone(self):
        with patch.object(dashboard.urllib.request, 'urlopen', side_effect=[self.response(self.status('agent-smith')), self.response({'ok':True})]) as call:
            result = self.client.post('/api/nowhere/start', headers=self.headers,
                json=dict(profile='e4b', system_prompt='Known acquaintance: Byers.', listen=True,
                          peer_speaker_id='chey', peer_collection_id='voice-trio'))
            self.assertEqual(result.status_code, 200)
            body = json.loads(call.call_args.args[0].data)
            self.assertFalse(body['listen'])
            self.assertNotIn('peer_speaker_id', body)
            self.assertNotIn('Chey', body['system_prompt'])

    def test_duplicate_nodes_and_url_credentials_rejected(self):
        self.cfg['nodes'].append(dict(self.cfg['nodes'][0]))
        with self.assertRaises(ValueError): dashboard.create_app(self.cfg)
        self.cfg['nodes'].pop()
        self.cfg['nodes'][0]['url'] = 'http://user:password@nowhere'
        with self.assertRaises(ValueError): dashboard.create_app(self.cfg)

    def test_classroom_speaker_must_be_on_its_assigned_node(self):
        self.cfg['nodes'][0]['speaker_id'] = 'catherine-byers'
        with self.assertRaisesRegex(ValueError, 'assignment disagree'):
            dashboard.create_app(self.cfg)

    def test_controls_only_node_never_launches_pipeline(self):
        with tempfile.TemporaryDirectory() as root, patch.object(server.Device, 'audio_poll'):
            cfg = dict(name='Byers', runtime_dir=root, controls_only=True,
                       profiles={'e4b':{'model':'/model'}}, voice='/voice', whisper_model='/asr', sink='sink', source='source')
            device = server.Device(cfg)
            with self.assertRaisesRegex(ValueError, 'not configured'):
                device.start(dict(profile='e4b', system_prompt='Byers', listen=True))
            self.assertIsNone(device.thread)
            snapshot = device.snapshot()
            self.assertFalse(snapshot['capabilities']['pipeline'])
            self.assertFalse(snapshot['capabilities']['capture'])
            self.assertTrue(snapshot['instance_id'])

    def test_session_controls_and_limits_are_in_compact_page(self):
        page = self.client.get('/').text
        for text in ('id="trio-topic"', 'id="trio-begin"', 'id="trio-end"', 'id="trio-record"', 'id="trio-total"'):
            self.assertIn(text, page)
        self.assertNotIn('id="stop-all"', page)

    def test_unconfigured_session_rejected_without_stopping_or_loading_any_node(self):
        def respond(req, **kw):
            node = next(n for n in self.cfg['nodes'] if req.full_url.startswith(n['url']))
            return self.response(self.status(node['speaker_id']))
        with patch.object(dashboard.urllib.request, 'urlopen', side_effect=respond) as call:
            result = self.client.post('/api/conversation', headers=self.headers, json=dict(topic='Agents',
                nodes={n['id']:dict(profile='e4b', system_prompt='Own prior knowledge', turn_limit=1)
                       for n in self.cfg['nodes']}))
            self.assertEqual(result.status_code, 409)
            self.assertIn('not configured', result.json['error'])
            self.assertTrue(all(c.args[0].get_method() == 'GET' for c in call.call_args_list))

    def test_session_and_recording_share_auth_and_origin_guard(self):
        for route in ('conversation', 'recording'):
            self.assertEqual(self.client.get('/api/'+route).status_code, 403)
            self.assertEqual(self.client.delete('/api/'+route, headers=dict(self.headers, Origin='http://other')).status_code, 403)

    def test_recording_audio_selection_validated_and_forwarded(self):
        self.app.recording.available = True
        with patch.object(self.app.recording, 'arm', return_value={'phase':'armed', 'audio_mode':'digital'}) as arm:
            result = self.client.post('/api/recording', headers=self.headers, json={'audio_mode':'digital'})
            self.assertEqual(result.status_code, 200)
            arm.assert_called_once_with('digital')
        for data in ([], None, {'audio_mode':'unknown'}):
            result = self.client.post('/api/recording', headers=self.headers, json=data)
            self.assertEqual(result.status_code, 400)
        self.assertIn('id="trio-audio-mode"', self.client.get('/').text)


if __name__ == '__main__':
    unittest.main()
