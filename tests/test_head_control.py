"""Control must permit content as well as veto it, without a second request."""
import io
import queue
import struct
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock
from test_room_direct import runtime
import test_turn_head as head_fixtures
from test_turn_head import frame
from turn_head import ShadowHead, HiddenReplyDecoder


class ControlTests(unittest.TestCase):
    def test_forced_tags_are_not_reported_as_independent_agreement(self):
        r=runtime();r.head_control=True
        r.compare_shadow(dict(id='1',heard='h1',head=dict(action='speak'),text_action='speak',control_action='speak'))
        event=next(c for c in r.device.emit.call_args_list if c.args[0]=='turn_head_comparison')
        self.assertIsNone(event.kwargs['agrees'])
        self.assertFalse(event.kwargs['independent_text_decision'])

    def test_same_generation_speak_defer_wait_identity_budget_and_mismatch(self):
        cases = [('speak', b'[speak]Hello.<turn|>', True, False),
                 ('defer', b'[defer]<turn|>', True, False),
                 ('wait', b'[defer]<turn|>', True, False),
                 ('defer', b'[defer]<speaker_name>Agent Smith</speaker_name><turn|>', True, True),
                 ('speak', b'[speak]Hello.<turn|>', False, False),
                 ('defer', b'[speak]Forbidden.<turn|>', True, False)]
        for action, wire, permit, name in cases:
            with self.subTest(action=action, wire=wire, permit=permit):
                r = runtime(); r.head_control = r.text_decision_tags = True
                r.cfg = dict(room_identity={}, name_pronunciations={})
                r.speaker = dict(display_name='Chey', role='student')
                r.shadow_connection = 1
                r.state.update(input_count=0,reply_count=0,synth_bytes=0,played_bytes=0,
                               audible_until=0,history=[],capture_active=False)
                r.replies = queue.Queue(); r.frames = queue.Queue(); r.speech_rate = 16000
                r.synth_sent = r.synth_completed = 0
                writes = []; clauses = []
                def write(data):
                    writes.append(data)
                    if data.startswith(b'\x03'): return
                    result = dict(action=action, answer_channel_matches=True,
                                  ready_monotonic=time.monotonic(), snapshot=dict(turn=1))
                    r.apply_head_decision(result,dict(session='session',response='1',connection=1))
                    r.replies.put(wire)
                r.client = Mock(stdin=Mock(write=write))
                def synthesize(data):
                    clauses.append(data)
                    r.frames.put((b'P', bytes(640)))
                    r.frames.put((b'D', str(r.synth_sent).encode()))
                r.synth = Mock(stdin=Mock(write=synthesize))
                r.open_output = Mock(return_value=Mock(latency=0))
                original = r.publish
                def publish(**extra):
                    if r.current and r.current['status']=='candidate': r.current['permission']=permit
                    original(**extra)
                r.publish = publish
                obs = r.identities.observe('h1',[1,0],'I have invited Agent Smith.',clean=True,duration=3) if name else None
                r.respond('Input','Input',dict(admitted_at=0),obs,0)
                self.assertEqual(r.request_count,1)
                self.assertEqual(len(writes),2)
                self.assertEqual(writes[0],b'Input\n')
                self.assertEqual(writes[1],struct.pack('<BIBB',3,1,{'speak':1,'defer':2,'wait':3}[action],int(name)))
                self.assertEqual(clauses,[b'Hello.\n'] if action=='speak' and permit else [])
                self.assertEqual(r.responses['1']['spoken'], action=='speak' and permit)
                if action!='speak' or not permit: r.open_output.assert_not_called()
                if action=='wait':self.assertEqual(r.responses['1']['status'],'waiting')
                if name:self.assertIn('Agent Smith',r.identities.pending_names)
                if wire.startswith(b'[speak]Forbidden'):self.assertEqual(r.responses['1']['status'],'invalid')
                else:self.assertEqual(r.responses['1']['text_decision']['source'],'head_forced')

    def test_stale_duplicate_and_wrong_boundary_cannot_write_control(self):
        for bad in ('session','connection','current','duplicate','boundary'):
            r=runtime();r.head_control=True;r.shadow_connection=2;r.generation_open=True
            record=dict(id='1',heard='h1',needs_name=False)
            r.responses['1']=record;r.current=record
            context=dict(session='session',response='1',connection=2)
            result=dict(action='speak',answer_channel_matches=True,ready_monotonic=time.monotonic(),snapshot=dict(turn=1))
            if bad=='session':context['session']='old'
            if bad=='connection':context['connection']=1
            if bad=='current':r.current=None
            if bad=='duplicate':record['control_action']='defer'
            if bad=='boundary':result['answer_channel_matches']=False
            with self.assertRaises(ValueError):r.apply_head_decision(result,context)
            self.assertEqual(r.client.stdin.getvalue(),b'')

    def test_observer_error_closes_control_but_keeps_shadow_noncontrolling(self):
        for controlled in (True,False):
            r=runtime();r.head_control=controlled
            r.shadow_error('fixture failure',{})
            self.assertEqual(r.cancel.is_set(),controlled)
            self.assertEqual(r.yielded.is_set(),controlled)

    def test_control_callback_precedes_diagnostic_disk_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture=head_fixtures.ReadoutTests();fixture.package(root,'linear')
            selected=[];completed=threading.Event();errors=[]
            def decision(result,ctx):
                selected.append((result['mode'],result['controlling'],list((root/'out').iterdir())))
            worker=ShadowHead(root,fixture.model,root/'out',lambda *a:completed.set(),
                              lambda *a:(errors.append(a),completed.set()),decision=decision)
            try:
                HiddenReplyDecoder(lambda s:worker.submit(s,dict(connection=1))).feed(frame())
                self.assertTrue(completed.wait(2));self.assertEqual(errors,[])
                self.assertEqual(selected,[('control',True,[])])
            finally:worker.close()


if __name__=='__main__':unittest.main()
