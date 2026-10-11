"""Shared spoken-turn budget and arbitration of listener-owned floor requests."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import threading
import time
import uuid

from session_scenario import SHORT_TURN_POLICY
from room_policy import choose


class TurnProgress:
    """Watch useful work, including streamed text and scheduled audio drain."""
    def __init__(self, now):
        self.last = now
        self.previous = None

    def check(self, states, now, wall):
        signature = tuple((s['input_count'], s['reply_count'], s['synth_count'],
                           s.get('model_bytes', 0), len(s.get('reply', '')), s.get('input_commits', 0),
                           s.get('room', {}).get('heard_count', 0),
                           s['synth_bytes'], s['played_bytes'],
                           # Linux queues PCM ahead of real time. A finite,
                           # shrinking playback tail is progress after writes end.
                           int(max(0, s.get('audible_until', 0)-wall)))
                          for s in states.values())
        if signature != self.previous:
            self.previous, self.last = signature, now
        if now-self.last > 90:
            raise RuntimeError('No model, speech or turn progress for 90 seconds; check node logs and microphones')


def drained(state, now):
    return (state['reply_count'] >= state['input_count']
            and state['synth_count'] >= state['reply_count']
            and state['played_bytes'] >= state['synth_bytes']
            and now >= state.get('audible_until', 0)
            and state['phase'] not in ('thinking', 'speaking', 'hearing'))


class TrioSession:
    def __init__(self, nodes, call, status, recording, latency=None):
        self.nodes, self.call, self.status, self.recording = nodes, call, status, recording
        self.latency = latency
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.thread = None
        self.state = dict(id=None, active=False, phase='idle', mode='active', topic='', started=None,
                          error='', reason='', turns={})

    def snapshot(self):
        with self.lock:
            result = deepcopy(self.state)
        if self.latency:
            result['latency'] = self.latency.snapshot()
        return result

    def update(self, **values):
        with self.lock:
            self.state.update(values)

    def preflight(self):
        statuses, blockers = {}, []
        def check(node):
            try:
                return self.status(node), None
            except Exception as exc:
                return None, str(exc)
        with ThreadPoolExecutor(max_workers=len(self.nodes)) as pool:
            results = list(pool.map(check, self.nodes.values()))
        for (node_id, node), (result, error) in zip(self.nodes.items(), results):
            name = node['speaker'].get('address_name', node['speaker']['display_name'])
            if error:
                blockers.append(name+': '+error)
                continue
            statuses[node_id] = result
            caps = result.get('capabilities', {})
            if not caps.get('pipeline'):
                blockers.append(name+': voice pipeline not configured')
            elif not caps.get('capture'):
                blockers.append(name+': microphone conversation not enabled')
            elif not caps.get('room_session'):
                blockers.append(name+': configure the room listening and identity pipeline')
        return statuses, blockers

    def validate(self, data):
        if not isinstance(data, dict):
            raise ValueError('JSON object required')
        mode = data.get('mode', 'active')
        if mode not in ('active', 'passive'):
            raise ValueError('Mode must be active or passive')
        topic = data.get('topic')
        if not isinstance(topic, str) or not 1 <= len(topic.strip()) <= 1000 or len(topic.encode()) > 2500:
            raise ValueError('Subject required, up to 1000 characters / 2500 UTF-8 bytes')
        settings = data.get('nodes')
        if not isinstance(settings, dict) or set(settings) != set(self.nodes):
            raise ValueError('Provide settings for every configured speaker')
        limit = data.get('total_turns', 12)
        if type(limit) is not int or not 1 <= limit <= 48:
            raise ValueError('Total turns must be an integer from 1–48')
        clean = {}
        for node_id, entry in settings.items():
            if not isinstance(entry, dict):
                raise ValueError(node_id+': model settings required')
            if not isinstance(entry.get('profile'), str) or not entry['profile']:
                raise ValueError(node_id+': select a model')
            prompt = entry.get('system_prompt')
            if not isinstance(prompt, str) or not 1 <= len(prompt) <= 16000:
                raise ValueError(node_id+': system prompt required, up to 16000 characters')
            if SHORT_TURN_POLICY not in prompt:
                prompt += '\n'+SHORT_TURN_POLICY
            clean[node_id] = dict(profile=entry['profile'], system_prompt=prompt,
                                  turn_limit=0, listen=False)
        return dict(topic=topic.strip(), total_turns=limit, mode=mode, nodes=clean)

    def start(self, data):
        data = self.validate(data)
        with self.lock:
            if self.state['active']:
                raise ValueError('A session is already running or stopping')
            statuses, blockers = self.preflight()
            if blockers:
                raise ValueError('Cannot start: '+'; '.join(blockers))
            for node_id, settings in data['nodes'].items():
                if settings['profile'] not in statuses[node_id]['profiles']:
                    raise ValueError(node_id+': unknown model profile')
            self.cancel.clear()
            self.state = dict(id=str(uuid.uuid4()), active=True, phase='loading', mode=data['mode'], topic=data['topic'],
                              started=None, error='', reason='', total_turns=data['total_turns'], completed=0,
                              turns={key: dict(accepted=0, completed=0) for key in data['nodes']},
                              profiles={key: entry['profile'] for key, entry in data['nodes'].items()})
            if self.latency:
                self.latency.begin(self.state, statuses)
            self.thread = threading.Thread(target=self.run, args=(data,), daemon=True)
            self.thread.start()
        return self.snapshot()

    def stop(self):
        with self.lock:
            self.cancel.set()
            if self.state['active']:
                self.state.update(phase='stopping', reason='manual')
            else:
                # Stop remains useful for models loaded through individual panels.
                self.state.update(active=True, phase='stopping', error='', reason='manual')
                self.thread = threading.Thread(target=self.finish, daemon=True)
                self.thread.start()
        return self.snapshot()

    def states(self):
        results = {}
        for key, node in self.nodes.items():
            before = time.time()
            result = self.status(node)
            after = time.time()
            if self.latency:
                self.latency.ingest(key, result, before, after)
            state = dict(result['state'])
            # Every audible timestamp belongs to that node's clock.
            state['audible_until'] = time.time()+max(0, state.get('audible_until', 0)-result.get('server_time', time.time()))
            results[key] = state
        if self.latency:
            self.latency.checkpoint()
        return results

    def observe(self, states):
        """Only actual spoken replies count; decisions and heard turns do not."""
        with self.lock:
            for key, state in states.items():
                counter = self.state['turns'][key]
                count = state['input_count']
                if count < counter['accepted']:
                    raise RuntimeError(key+': node restarted during the session')
                counter['accepted'] = count
                if drained(state, time.time()): counter['completed'] = count
            accepted = sum(c['accepted'] for c in self.state['turns'].values())
            completed = sum(c['completed'] for c in self.state['turns'].values())
            if accepted > self.state['total_turns']:
                raise RuntimeError('Shared spoken-turn limit exceeded')
            self.state['completed'] = completed
            return completed == self.state['total_turns']

    def grant(self, speaker, epoch, opening=False):
        # Prepare EVERY listener before any audio can begin. Payloads contain
        # control epochs only: no roster, speaker identity, or transcript relay.
        for key, node in self.nodes.items():
            if self.cancel.is_set(): return
            self.call(node, 'room-prepare', dict(session=self.state['id'], epoch=epoch, speaking=key == speaker))
        if not self.cancel.is_set():
            self.call(self.nodes[speaker], 'room-grant', dict(session=self.state['id'], epoch=epoch, opening=opening))
        self.update(floor=speaker, epoch=epoch, phase='running')

    def run(self, data):
        try:
            for key, node in self.nodes.items():
                if self.cancel.is_set(): return
                self.call(node, 'stop')
                if self.cancel.is_set(): return
                settings = dict(data['nodes'][key], room=dict(session=self.state['id'], topic=data['topic']))
                self.call(node, 'start', settings)
            deadline = time.monotonic()+240
            while not self.cancel.wait(.2):
                states = self.states()
                if any(s.get('error') for s in states.values()):
                    raise RuntimeError('; '.join(k+': '+s['error'] for k, s in states.items() if s.get('error')))
                if all(s['ready'] for s in states.values()): break
                if time.monotonic() > deadline: raise RuntimeError('Models did not become ready within four minutes')
            if self.cancel.is_set(): return
            self.recording.models_ready([data['nodes'][key]['profile'] for key in self.nodes])
            rec = self.recording.snapshot()
            if rec['phase'] in ('error', 'armed', 'starting'):
                raise RuntimeError(rec.get('error') or 'Recording did not become ready')
            current = next((key for key, n in self.nodes.items() if n['speaker']['role'] == 'teacher'), next(iter(self.nodes)))
            epoch = 1
            self.grant(current, epoch, opening=True)
            self.update(started=time.time())
            progress = TurnProgress(time.monotonic())
            idle = False
            idle_heard = {}
            while not self.cancel.wait(.1):
                states = self.states()
                for key, state in states.items():
                    if not state['ready'] or state.get('error'):
                        raise RuntimeError(key+': '+(state.get('error') or 'pipeline stopped'))
                complete = self.observe(states)
                if not idle and not drained(states[current], time.time()):
                    for key in self.nodes:
                        if key != current: self.call(self.nodes[key], 'endpoint-hold', {'ms': 1500})
                else:
                    requests = {key: s.get('room', {}).get('decision') for key, s in states.items()}
                    heard_all = (all(requests[key] and requests[key]['epoch'] == epoch for key in self.nodes if key != current)
                                 and all(s.get('room', {}).get('phase') in ('listening', 'waiting') for s in states.values()))
                    if idle:
                        heard_all = all(requests[key] and requests[key]['epoch'] == epoch
                                        and requests[key]['heard'] != idle_heard.get(key)
                                        and states[key].get('room', {}).get('phase') == 'listening'
                                        for key in self.nodes)
                    if heard_all:
                        failures = [key+': '+value['error'] for key, value in requests.items()
                                    if value and value.get('error')]
                        if failures: raise RuntimeError('Listener decision failed: '+'; '.join(failures))
                        if complete:
                            self.update(reason='limit', phase='stopping')
                            return
                        chosen = choose({k:v for k,v in requests.items() if v and v['epoch'] == epoch}, None if idle else current)
                        if chosen:
                            epoch += 1
                            self.grant(chosen, epoch)
                            current, idle = chosen, False
                            progress = TurnProgress(time.monotonic())
                        elif not idle:
                            # Silence is a valid joint decision. Keep every mic
                            # open for a human contribution; never force a turn.
                            for node in self.nodes.values():
                                self.call(node, 'room-idle', dict(session=self.state['id'], epoch=epoch))
                            self.update(phase='waiting', reason='All listeners deferred; waiting for speech')
                            idle = True
                            idle_heard = {k:v['heard'] if v else None for k,v in requests.items()}
                        else:
                            idle_heard = {k:v['heard'] for k,v in requests.items()}
                if not idle: progress.check(states, time.monotonic(), time.time())
        except Exception as exc:
            self.update(error=str(exc))
        finally:
            self.finish()

    def finish(self):
        errors = []
        def stop(node):
            try:
                before = time.time()
                result = self.status(node)  # Never stop an unrelated service on a reused port.
                if self.latency:
                    self.latency.ingest(node['id'], result, before, time.time())
                self.call(node, 'stop')
            except Exception as exc:
                return node['id']+': '+str(exc)
        self.update(phase='stopping')
        # Playback and ASR taps belong to the running audio services. Begin recorder
        # shutdown before closing them so an intentional stop is not a feed
        # failure. Do not delay a manual speech cut while the MP4 finalizes.
        try:
            self.recording.stop(wait=False)
        except Exception as exc:
            errors.append('Recording: '+str(exc))
        with ThreadPoolExecutor(max_workers=len(self.nodes)) as pool:
            errors.extend(error for error in pool.map(stop, self.nodes.values()) if error)
        try:
            self.recording.stop(wait=True)
            if self.recording.snapshot().get('error'):
                errors.append(self.recording.snapshot()['error'])
        except Exception as exc:
            errors.append('Recording: '+str(exc))
        with self.lock:
            error = '; '.join(filter(None, [self.state['error'], *errors]))
            self.state.update(active=False, phase='error' if error else 'stopped', error=error)
            if self.latency:
                self.latency.finish('error' if error else self.state['reason'])

    def shutdown(self):
        self.cancel.set()
        if self.thread and self.thread.is_alive(): self.thread.join(timeout=60)
        self.recording.stop(wait=True)
