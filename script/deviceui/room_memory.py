"""Opt-in async client for the isolated research memory worker.

The voice engine never generates maintenance speech. Completed proposals are
used only at the existing renewal boundary; errors/late work preserve baseline
renewal. Authoritative identity is supplied separately by RoomRuntime.
"""
import copy
import hashlib
import json
import queue
import threading
import time
import urllib.request


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


class RoomMemory:
    def __init__(self, settings, session, listener, cancel, emit):
        if settings.get('mode') != 'clm_trial': raise ValueError('Unknown room memory mode')
        self.endpoint = settings['endpoint'].rstrip('/')
        self.key = settings['key']
        self.session, self.listener = session, listener
        self.cancel, self.emit = cancel, emit
        self.events = []
        self.serial = self.covered = 0
        self.memory = ''
        self.pending = False
        self.results = queue.Queue(maxsize=1)
        self.last_submitted = 0
        self.last_applied = 0
        self.disabled = False

    def poll(self):
        try: request, result, error = self.results.get_nowait()
        except queue.Empty: return
        self.pending = False
        if self.cancel.is_set(): return
        if error:
            self.emit('room_memory_failed', error)
            return
        source = hashlib.sha256(encoded(request)).hexdigest()
        if (result.get('source_sha256') != source or result.get('session') != self.session
                or result.get('listener') != self.listener or result.get('covered') != request['covered']):
            self.emit('room_memory_failed', 'Worker result does not match listener/source revision')
            return
        if not result.get('accepted'):
            self.emit('room_memory_rejected', result.get('reason',''), covered=request['covered'])
            return
        text = result.get('memory')
        if (not isinstance(text, str) or not 1 <= len(text) <= 2200 or '\x00' in text
                or '<' in text or request['covered'] <= self.covered):
            self.emit('room_memory_failed', 'Invalid memory payload or stale revision')
            return
        self.memory, self.covered = text, request['covered']
        self.events = [e for e in self.events if e['serial'] > self.covered]
        self.emit('room_memory_ready', memory=text, covered=self.covered,
                  source_sha256=source, tokens=result.get('tokens'), worker_s=result.get('elapsed_s'))

    def observe(self, entry, identity):
        self.poll()
        self.serial += 1
        self.events.append(dict(serial=self.serial, entry=copy.deepcopy(entry)))
        if self.pending or self.disabled or self.cancel.is_set(): return
        # Keep at least four latest speech events verbatim, including the
        # immediate question/answer pair. Summarize only older heard/spoken data.
        older = self.events[:-4]
        if len(older) < 2 or older[-1]['serial'] <= self.last_submitted: return
        if len(self.events) > 128:
            self.disabled = True
            self.emit('room_memory_failed', 'Trial journal limit reached; baseline renewal retained')
            return
        request = dict(session=self.session, listener=self.listener, covered=older[-1]['serial'],
                       prior_memory=self.memory, events=older, identity=copy.deepcopy(identity))
        if len(encoded(request)) > 24000:
            self.emit('room_memory_failed', 'Maintenance source exceeds trial request bound')
            return
        self.last_submitted = request['covered']
        self.pending = True
        self.emit('room_memory_submitted', covered=request['covered'], events=len(older))
        def work():
            result, error = None, None
            try:
                req = urllib.request.Request(self.endpoint+'/v1/compact', data=encoded(request),
                    headers={'X-Memory-Key':self.key, 'Content-Type':'application/json'})
                with urllib.request.urlopen(req, timeout=35) as response:
                    raw = response.read(16385)
                if len(raw)>16384: raise ValueError('Oversized memory worker response')
                result = json.loads(raw)
                if not isinstance(result, dict): raise ValueError('Invalid memory worker response')
            except Exception as exc:
                error = type(exc).__name__+': '+str(exc)
            if not self.cancel.is_set(): self.results.put_nowait((request,result,error))
        threading.Thread(target=work, name='room-memory-trial', daemon=True).start()

    def retained(self):
        self.poll()
        tail = [copy.deepcopy(e['entry']) for e in self.events]
        if self.disabled or not self.memory or not 1 <= len(tail) <= 6:
            self.emit('room_memory_fallback', 'No sufficiently current accepted edit',
                      pending=self.pending, tail_events=len(tail))
            return None
        self.last_applied = self.covered
        return dict(edited_memory=self.memory, conversation=tail, memory_covered_event=self.covered)
