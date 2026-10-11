"""Concurrent listener responses with a shared budget, without floor arbitration."""
import time
from trio_session import TrioSession, TurnProgress


class SpokenBudget:
    """Idempotent reservations: concurrent replies cannot overshoot the last slot."""
    def __init__(self, limit):
        self.limit = limit
        self.permits = {}
        self.finished = {}

    def refresh(self, states):
        for node, state in states.items():
            for key, record in state.get('room', {}).get('responses', {}).items():
                token = (node, key)
                if self.permits.get(token) and record['done']:
                    self.finished[token] = bool(record['spoken'])

    @property
    def occupied(self):
        return sum(allowed and self.finished.get(token, True) for token, allowed in self.permits.items())

    @property
    def complete(self):
        return sum(self.finished.values()) == self.limit

    def reserve(self, node, response):
        token = (node, response)
        if token not in self.permits: self.permits[token] = self.occupied < self.limit
        return self.permits[token]


class DirectTrioSession(TrioSession):
    def run(self, data):
        try:
            for key, node in self.nodes.items():
                if self.cancel.is_set(): return
                self.call(node, 'stop')
                if self.cancel.is_set(): return
                self.call(node, 'start', dict(data['nodes'][key],
                    room=dict(session=self.state['id'], topic=data['topic'], mode=data.get('mode', 'active'))))
            deadline = time.monotonic()+240
            while not self.cancel.wait(.2):
                states = self.states()
                if any(s.get('error') for s in states.values()):
                    raise RuntimeError('; '.join(k+': '+s['error'] for k,s in states.items() if s.get('error')))
                if all(s['ready'] for s in states.values()): break
                if time.monotonic() > deadline: raise RuntimeError('Models did not become ready within four minutes')
            if self.cancel.is_set(): return
            if not all(s.get('room', {}).get('protocol') == 'direct-v1' for s in states.values()):
                raise RuntimeError('Every node must support the same direct response protocol')
            passive = data.get('mode', 'active') == 'passive'
            if any(s.get('pipeline_mode', 'active') != ('passive' if passive else 'active') for s in states.values()):
                raise RuntimeError('Every node must support the requested pipeline mode')
            self.recording.models_ready(['passive' if passive else data['nodes'][key]['profile'] for key in self.nodes])
            rec = self.recording.snapshot()
            if rec['phase'] in ('error', 'armed', 'starting'):
                raise RuntimeError(rec.get('error') or 'Recording did not become ready')
            for node in self.nodes.values():
                self.call(node, 'room-listen', dict(session=self.state['id']))
            if passive:
                self.update(started=time.time(), phase='waiting', protocol='direct-v1', mode='passive')
                while not self.cancel.wait(.2):
                    states = self.states()
                    for key, state in states.items():
                        if not state['ready'] or state.get('error'):
                            raise RuntimeError(key+': '+(state.get('error') or 'listening pipeline stopped'))
                    self.update(reason='Passive listening · no LLM responses')
                return
            teacher = next(key for key, node in self.nodes.items() if node['speaker']['role'] == 'teacher')
            self.call(self.nodes[teacher], 'room-open', dict(session=self.state['id']))
            self.update(started=time.time(), phase='running', protocol='direct-v1')
            budget = SpokenBudget(data['total_turns'])
            progress = TurnProgress(time.monotonic())
            while not self.cancel.wait(.05):
                states = self.states()
                if self.cancel.is_set(): return
                for key, state in states.items():
                    if not state['ready'] or state.get('error'):
                        raise RuntimeError(key+': '+(state.get('error') or 'pipeline stopped'))
                budget.refresh(states)
                # Every candidate may speak while slots remain. The timestamps
                # only resolve competition for the *last budget slots*, not the floor.
                candidates = sorted((r['at'], key, rid) for key,s in states.items()
                    for rid,r in s['room']['responses'].items()
                    if r['status'] == 'candidate' and r['permission'] is None)
                for _, key, rid in candidates:
                    if self.cancel.is_set(): return
                    allowed = budget.reserve(key, rid)
                    self.call(self.nodes[key], 'room-permit',
                              dict(session=self.state['id'], response=rid, allowed=allowed))
                with self.lock:
                    for key, s in states.items():
                        self.state['turns'][key] = dict(accepted=s['input_count'], completed=s['synth_count'])
                    self.state['completed'] = sum(s['synth_count'] for s in states.values())
                    if self.state['completed'] > data['total_turns']: raise RuntimeError('Shared turn limit exceeded')
                # Do not hold endpoints when a peer begins replying. Doing so
                # would implicitly make the first reply suppress slower peers'
                # decisions. Each microphone keeps its own acoustic endpoint.
                idle = all(s['room']['phase'] == 'listening' and not s.get('capture_active')
                           and time.time() >= s.get('audible_until', 0) for s in states.values())
                if budget.complete and idle:
                    self.update(reason='limit', phase='stopping')
                    return
                self.update(phase='waiting' if idle else 'running',
                            reason='No speaker has taken the floor; listening for more speech' if idle else '')
                if not idle: progress.check(states, time.monotonic(), time.time())
        except Exception as exc:
            self.update(error=str(exc))
        finally:
            self.finish()
