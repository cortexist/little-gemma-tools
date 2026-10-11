"""Session-owned voice timing; never requires a browser or changes the audio path."""
from copy import deepcopy
import json
import math
from pathlib import Path
import statistics
import threading
import time


def stats(values):
    return dict(n=len(values), last=values[-1], median=statistics.median(values),
                maximum=max(values)) if values else None


class TrioLatency:
    def __init__(self, runtime_dir=None):
        self.lock = threading.RLock()
        self.root = Path(runtime_dir).expanduser()/'latency' if runtime_dir else None
        self.active = False
        self.saved = {}
        self.last_save = 0
        if self.root:
            try:
                self.saved = json.loads((self.root/'latest.json').read_text())
                if self.saved.get('active'):
                    self.saved.setdefault('warnings', []).append('Dashboard restarted; measurement is incomplete')
                self.saved['active'] = False
            except FileNotFoundError:
                pass
            except (OSError, ValueError):
                self.saved = dict(warnings=['Could not read saved latency measurements'])

    def begin(self, session, statuses, source='live'):
        with self.lock:
            self.meta = dict(version=1, id=session['id'], started=time.time(), source=source,
                             profiles=deepcopy(session['profiles']), reason='')
            self.nodes = {key: dict(cursor=value.get('cursor', 0), instance=value.get('instance_id'),
                                   current=None, pending_asr=None, clock=None, turns=[], reset=False,
                                   host=deepcopy(value.get('host', {})),
                                   model={k: value.get('profiles', {}).get(session['profiles'][key], {}).get(k)
                                          for k in ('label', 'model', 'head')})
                          for key, value in statuses.items()}
            self.warnings = []
            self.gaps_valid = True
            self.active = True
            self.last_save = 0

    def warn(self, message):
        if message not in self.warnings:
            self.warnings.append(message)

    def ingest(self, key, result, before, after):
        with self.lock:
            if not self.active:
                return
            node = self.nodes[key]
            if result.get('instance_id') != node['instance']:
                self.warn(key+': node restarted; timing coverage is incomplete')
                self.gaps_valid = False
                node.update(cursor=0, instance=result.get('instance_id'), current=None, pending_asr=None)
            server = result.get('server_time')
            if server is not None and after >= before:
                sample = dict(rtt=after-before, offset=server-(before+after)/2, at=after)
                old = node['clock']
                if old and abs(sample['offset']-old['offset']) > max(.25, sample['rtt']+old['rtt']):
                    self.warn(key+': clock changed; inter-speaker gaps unavailable')
                    self.gaps_valid = False
                if old is None or sample['rtt'] < old['rtt']:
                    node['clock'] = sample
            for event in sorted(result.get('events', []), key=lambda e: e['id']):
                if event['id'] <= node['cursor']:
                    continue
                if event['id'] != node['cursor']+1:
                    self.warn(key+': missing events; timing coverage is incomplete')
                    self.gaps_valid = False
                    node.update(current=None, pending_asr=None)
                node['cursor'] = event['id']
                try:
                    self.event(node, event)
                except (ValueError, TypeError, KeyError, OverflowError):
                    self.warn(key+': malformed timing event')
                    self.gaps_valid = False
                    node.update(current=None, pending_asr=None)

    def event(self, node, event):
        kind, at = event['kind'], float(event['at'])
        if not math.isfinite(at):
            raise ValueError('Invalid event time')
        if kind == 'reset':
            if node['reset']:
                self.warn('Pipeline reloaded during measurement; timing coverage is incomplete')
                self.gaps_valid = False
            node.update(current=None, pending_asr=None, reset=True)
        elif kind == 'asr':
            elapsed = float(event['elapsed'])
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError('Invalid ASR duration')
            node['pending_asr'] = dict(at=at, seconds=elapsed)
        elif kind == 'input_done':
            turn = dict(turn=len(node['turns'])+1, input_at=at, complete=False)
            if event.get('heard_at') is not None:
                heard, decision = float(event['heard_at']), float(event['decision_at'])
                if not all(math.isfinite(v) for v in (heard, decision)) or not heard <= decision <= at:
                    raise ValueError('Invalid room decision timing')
                turn.update(input_at=heard, grant_at=at, decision_s=decision-heard, floor_wait_s=at-decision)
                if event.get('direct_response'):
                    turn.update(direct_response=True, response_start_s=decision-heard,
                                budget_wait_s=at-decision)
                    turn.pop('decision_s', None)
                    turn.pop('floor_wait_s', None)
                for field in ('identity_wait_s', 'name_s', 'floor_decision_s'):
                    if field in event:
                        value = float(event[field])
                        if not math.isfinite(value) or value < 0:
                            raise ValueError('Invalid room stage duration')
                        turn[field] = value
            asr = node.pop('pending_asr', None)
            # Only the immediately preceding transcription can belong to this input.
            if asr and 0 <= at-asr['at'] < 1:
                turn['asr_s'] = asr['seconds']
            # Incremental ASR can finish during the endpoint hold, well before
            # admission. The adapter explicitly associates its final reused pass.
            if event.get('asr_elapsed') is not None:
                elapsed = float(event['asr_elapsed'])
                if not math.isfinite(elapsed) or elapsed < 0:
                    raise ValueError('Invalid associated ASR duration')
                turn['asr_s'] = elapsed
                turn['asr_mode'] = event.get('asr_mode')
                turn['asr_passes'] = event.get('asr_passes')
            node['turns'].append(turn)
            node['current'] = turn
        elif node['current'] is not None:
            turn = node['current']
            if kind in ('model_first_token', 'clause', 'speech_first_pcm', 'play'):
                turn.setdefault(kind+'_at', at)
            if kind == 'play':
                start, offset, count, rate = map(float, event['text'].split('\t'))
                if not all(math.isfinite(v) for v in (start, offset, count, rate)) or rate <= 0 or count < 0:
                    raise ValueError('Invalid PCM duration')
                turn['pcm_end_at'] = max(turn.get('pcm_end_at', 0), start+count/(2*rate))
            elif kind == 'synth_done':
                turn['complete'] = True
                if event.get('interrupted'):
                    turn['interrupted'] = True
                    turn['pcm_end_at'] = min(turn.get('pcm_end_at', at), at)

    def snapshot(self):
        with self.lock:
            if not self.active:
                return deepcopy(self.saved)
            turns = []
            for key, node in self.nodes.items():
                for entry in node['turns']:
                    turn = dict(entry, node=key)
                    pcm = turn.get('speech_first_pcm_at', turn.get('play_at'))
                    turn['pcm_kind'] = 'ready' if 'speech_first_pcm_at' in turn else 'pipe-write'
                    if pcm is not None and pcm >= turn['input_at']:
                        turn['pcm_s'] = pcm-turn['input_at']
                    for kind, field in [('model_first_token', 'model_s'), ('clause', 'clause_s')]:
                        if kind+'_at' in turn:
                            turn[field] = turn[kind+'_at']-turn['input_at']
                    if pcm is not None and 'clause_at' in turn:
                        turn['tts_s'] = pcm-turn['clause_at']
                    if node['clock']:
                        turn['input_host_at'] = turn['input_at']-node['clock']['offset']
                        if pcm is not None:
                            turn['pcm_host_at'] = pcm-node['clock']['offset']
                        if 'pcm_end_at' in turn:
                            turn['pcm_end_host_at'] = turn['pcm_end_at']-node['clock']['offset']
                    turns.append(turn)
            if self.gaps_valid and all(node['clock'] for node in self.nodes.values()):
                turns.sort(key=lambda t: t['input_host_at'])
                for previous, turn in zip(turns, turns[1:]):
                    if (previous['node'] != turn['node'] and previous['complete']
                            and 'pcm_end_host_at' in previous and 'pcm_host_at' in turn):
                        turn['previous_node'] = previous['node']
                        turn['gap_s'] = turn['pcm_host_at']-previous['pcm_end_host_at']
                        turn['gap_clock_uncertainty_s'] = sum(self.nodes[k]['clock']['rtt']/2
                                                             for k in (previous['node'], turn['node']))
            summaries = {}
            for key, node in self.nodes.items():
                group = [t for t in turns if t['node'] == key]
                summaries[key] = dict(host=node['host'], model=node['model'], clock=node['clock'],
                                      observed_turns=len(group),
                                      pcm_kinds=sorted({t['pcm_kind'] for t in group if 'pcm_s' in t}),
                                      **{field: stats([t[field] for t in group if field in t]) for field in
                                         ('pcm_s', 'gap_s', 'asr_s', 'model_s', 'clause_s', 'tts_s', 'decision_s', 'floor_wait_s',
                                          'identity_wait_s', 'name_s', 'floor_decision_s',
                                          'response_start_s', 'budget_wait_s')})
            return dict(self.meta, active=not self.meta.get('finished', False), nodes=summaries, turns=turns, warnings=list(self.warnings),
                        storage='saved' if self.root else 'memory-only')

    def checkpoint(self, force=False):
        with self.lock:
            if not self.active or not self.root or (not force and time.monotonic()-self.last_save < 1):
                return
            self.last_save = time.monotonic()
            data = self.snapshot()
            try:
                self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
                body = json.dumps(data, indent=2, allow_nan=False)+'\n'
                for filename in (self.meta['id']+'.json', 'latest.json'):
                    target = self.root/filename
                    pending = target.with_suffix('.tmp')
                    pending.write_text(body)
                    pending.replace(target)
            except (OSError, ValueError) as exc:
                self.warn('Could not save latency measurements: '+str(exc))

    def finish(self, reason):
        with self.lock:
            if not self.active:
                return
            self.meta.update(reason=reason, ended=time.time())
            # The persisted result must describe a completed capture, not a restart.
            self.meta['finished'] = True
            self.checkpoint(force=True)
            self.saved = self.snapshot()
            self.saved['active'] = False
            self.active = False
