import copy
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'script/deviceui'),
              str(Path(__file__).resolve().parents[1]/'script')]
from room_memory import RoomMemory, encoded
from room_runtime import RoomRuntime, SocketClient
from session_speakers import SpeakerSession


class MemoryTests(unittest.TestCase):
    def test_shortened_boundary_requires_explicit_trial_and_prefills_at_four(self):
        device=SimpleNamespace(cfg=dict(room_memory=dict(mode='off',renew_after=4)),
            state=dict(synth_count=1),lock=threading.RLock(),cancel=threading.Event(),procs={},
            emit=lambda *a,**k:None,speaker=dict(speaker_id='chey'))
        with self.assertRaises(ValueError):RoomRuntime(device,dict(session='s',topic='Agents'))
        device.cfg['room_memory'].update(mode='clm_trial',endpoint='http://unused',key='test')
        room=RoomRuntime(device,dict(session='s',topic='Agents'))
        room.identities=SpeakerSession();room.client=SocketClient.__new__(SocketClient)
        closed=[];room.client.close=lambda:closed.append(True)
        stream=io.BytesIO();room.connect=lambda:setattr(room,'client',SimpleNamespace(stdin=stream))
        room.context_turns=3;room.renew_context();self.assertFalse(closed)
        room.context_turns=4;room.renew_context();self.assertTrue(closed)
        self.assertTrue(stream.getvalue().startswith(b'\x01T'))

    def client(self):
        events=[]
        m=RoomMemory(dict(mode='clm_trial',endpoint='http://unused',key='test'),
            'session','chey',threading.Event(),lambda kind,text='',**data:events.append((kind,data)))
        return m,events

    def result(self,m,covered=2):
        request=dict(session=m.session,listener=m.listener,covered=covered,
                     prior_memory='',events=[],identity={})
        result=dict(session=m.session,listener=m.listener,covered=covered,accepted=True,
                    memory='[[CTX_TURN 1 role=user]]\nThe guest explained harnesses.',
                    source_sha256=hashlib.sha256(encoded(request)).hexdigest())
        return request,result

    def test_listener_or_revision_mismatch_cannot_apply(self):
        for changed in ('listener','covered','source_sha256'):
            m,events=self.client();request,result=self.result(m)
            result[changed]='wrong'
            m.results.put((request,result,None));m.poll()
            self.assertEqual(m.memory,'');self.assertEqual(m.covered,0)
            self.assertEqual(events[-1][0],'room_memory_failed')

    def test_pending_rejected_and_late_results_keep_baseline(self):
        m,events=self.client();m.pending=True
        self.assertIsNone(m.retained())
        request,result=self.result(m);result.update(accepted=False,reason='budget')
        m.results.put((request,result,None));self.assertIsNone(m.retained())
        request,result=self.result(m)
        m.events=[dict(serial=n,entry=dict(heard=str(n))) for n in range(1,10)]
        m.results.put((request,result,None))
        # Seven new speech events must not disappear behind an old summary.
        self.assertIsNone(m.retained());self.assertEqual(len(m.events),7)

    def test_newer_heard_input_preserved_and_cancelled_job_ignored(self):
        m,events=self.client();request,result=self.result(m)
        m.events=[dict(serial=n,entry=dict(heard='speech '+str(n))) for n in range(1,7)]
        m.results.put((request,result,None));retained=m.retained()
        self.assertEqual([e['heard'] for e in retained['conversation']],['speech '+str(n) for n in range(3,7)])
        m.cancel.set();request,result=self.result(m,4)
        m.results.put((request,result,None));m.poll();self.assertEqual(m.covered,2)

    def test_renewal_preserves_authoritative_state_and_sends_only_prefill(self):
        events=[]
        device=SimpleNamespace(cfg={},state=dict(synth_count=3),lock=threading.RLock(),
            cancel=threading.Event(),procs={},emit=lambda kind,text='',**data:events.append((kind,data)))
        room=RoomRuntime(device,dict(session='session',topic='Agents'))
        room.identities=SpeakerSession([dict(name='Mrs. Byers',embedding=[1,0])])
        room.last_spoken_text='My question to the guest.'
        room.conversation=[dict(heard='A recent answer.')]
        room.context_turns=8
        original=copy.deepcopy(room.identities.snapshot())
        room.memory_trial=SimpleNamespace(retained=lambda:dict(edited_memory='Old lesson details.',
            conversation=[dict(heard='A recent answer.')],memory_covered_event=4))
        room.client=SocketClient.__new__(SocketClient);room.client.close=lambda:None
        output=io.BytesIO();room.connect=lambda:setattr(room,'client',SimpleNamespace(stdin=output))
        room.silent=lambda *a: self.fail('Must not generate maintenance on speech engine')
        room.renew_context()
        raw=output.getvalue();prefix=b''
        while raw:
            self.assertEqual(raw[:2],b'\x01T');size=struct.unpack('<I',raw[6:10])[0]
            prefix+=raw[10:10+size];raw=raw[10+size:]
        memory=json.loads(prefix.decode().split('ROOM_CONTEXT ',1)[1].split(' END_ROOM_CONTEXT',1)[0])
        self.assertEqual(memory['identity']['tracks']['voice-1']['name'],'Mrs. Byers')
        self.assertEqual(memory['own_last_spoken_text'],'My question to the guest.')
        self.assertEqual(memory['spoken_turns'],3)
        self.assertEqual(original,room.identities.snapshot())
        self.assertIn('Old lesson details.',prefix.decode())
        self.assertNotIn(b'\n',prefix)
        self.assertIn('room_memory_applied',[e[0] for e in events])


if __name__=='__main__':unittest.main()
