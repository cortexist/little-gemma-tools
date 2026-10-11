"""Passive capture must retain acoustic observations without any LLM work."""
import io
from pathlib import Path
import queue
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'script/deviceui'),str(Path(__file__).resolve().parents[1]/'script')]
from room_direct import DirectRoomRuntime
from session_speakers import SpeakerSession
from trio_direct import DirectTrioSession


def runtime():
    device=Mock();device.cfg={};device.lock=threading.RLock();device.cancel=threading.Event()
    device.state=dict(listening=True,input_commits=0,synth_count=0,reply_count=0);device.procs={}
    r=DirectRoomRuntime(device,dict(session='session',topic='Audio',mode='passive'))
    r.client=Mock(stdin=io.BytesIO())
    r.identities=SpeakerSession([dict(name='Byers',embedding=[1,0])],canonicalize=False)
    return r


class PassiveTests(unittest.TestCase):
    def test_partial_commits_and_release_never_prefill_or_renew(self):
        r=runtime();r.commit_input('Hello ');r.commit_input('world')
        r.context_turns=100;r.reset_context=Mock()
        r.release_engine()
        self.assertEqual(r.state['input'],'Hello world')
        self.assertEqual(r.client.stdin.getvalue(),b'')
        self.assertEqual(r.capture_sent,'')
        r.reset_context.assert_not_called()

    def test_heard_voice_is_observed_without_generation_or_name_inference(self):
        r=runtime();r.sound_input=True;r.cfg={'room_identity':{}}
        r.respond=Mock();r.reset_context=Mock();r.send_prefix=Mock()
        live=Mock();live.finish.return_value=dict(tracks=['voice-1'],embedding=[1,0],status='matched')
        r.decision(('observe','Hello','Hello',{'sound_event':{'candidates':[]}},bytes(32000),0,live),None)
        self.assertEqual(r.heard,1)
        self.assertEqual(r.identities.turns['h1']['track_id'],'voice-1')
        self.assertEqual(r.state['reply_count'],0)
        self.assertEqual(r.request_count,0)
        r.respond.assert_not_called();r.send_prefix.assert_not_called();r.reset_context.assert_not_called()
        self.assertTrue(any(c.args[0]=='room_passive_heard' for c in r.device.emit.call_args_list))

    def test_wordless_sound_does_not_create_voice_or_response(self):
        r=runtime();r.sound_input=True;r.respond=Mock()
        live=Mock();live.finish.return_value=dict(tracks=[],embedding=None,status='unattributed')
        r.decision(('observe','','',{'sound_event':{'candidates':[{'label':'Door'}]}},bytes(32000),0,live),None)
        self.assertEqual(r.heard,0);self.assertEqual(len(r.identities.tracks),1)
        r.respond.assert_not_called()
        self.assertTrue(any(c.args[0]=='room_sound' for c in r.device.emit.call_args_list))

    def test_own_voice_never_resets_a_passive_context(self):
        from room_direct import SelfAwareVoices
        r=runtime();r.reset_context=Mock();r.respond=Mock()
        live=Mock();live.finish.return_value=dict(tracks=[SelfAwareVoices.SELF])
        r.decision(('observe','Hello world','world',{},bytes(32000),0,live),None)
        r.reset_context.assert_not_called();r.respond.assert_not_called()

    def test_passive_cannot_open_or_respond_even_if_requested_directly(self):
        r=runtime()
        with self.assertRaisesRegex(ValueError,'Passive'):r.control('room-open',{'session':'session'})
        with self.assertRaisesRegex(ValueError,'Passive'):r.respond('Hello','Hello',{},None,0)
        self.assertTrue(r.prompts.empty())

    def test_passive_skips_memory_editor(self):
        device=Mock();device.cfg={'room_memory':{'mode':'clm_trial'}}
        device.lock=threading.RLock();device.cancel=threading.Event();device.procs={};device.state={}
        with patch('room_memory.RoomMemory') as memory:
            r=DirectRoomRuntime(device,dict(session='session',topic='Audio',mode='passive'))
        self.assertIsNone(r.memory_trial);memory.assert_not_called()

    def test_passive_startup_spawns_only_capture_whisper_and_identity(self):
        r=runtime();r.cfg={'room_identity':{'model':'fake','sha256':'fake','priors':{}},
                          'far_field':'audio','source':'mic','sink':'speaker',
                          'whisper_command':['whisper'],'whisper_health':'http://health'}
        with tempfile.TemporaryDirectory() as directory:
            r.device.root=Path(directory);r.device.speaker={'speaker_id':'teacher'}
            r.prepare_identity=Mock();r.refresh_asr_names=Mock();r.capture=Mock();r.linux_capture=Mock()
            r.prepare_engine_observer=Mock();r.connect=Mock();r.synthesis_command=Mock()
            spawned=[];r.spawn=lambda name,*args,**kwargs:spawned.append(name)
            r.terminate_owned=Mock()
            r.device.emit.side_effect=lambda kind,*args,**kwargs:r.cancel.set() if kind=='ready' else None
            worker=Mock();worker.error=None
            with patch('room_runtime.AcousticEncoder'),patch('room_runtime.ClassroomScenario') as scenario,\
                 patch('room_runtime.ParallelIdentity',return_value=worker),\
                 patch('room_runtime.urllib.request.urlopen') as health,patch.object(Path,'is_socket',return_value=True):
                scenario.return_value.priors.return_value=[];health.return_value.__enter__.return_value.status=200
                r.work('unused','unused')
            self.assertEqual(spawned,['audio','whisper'])
            self.assertFalse(r.state['error'],r.state['error'])
            self.assertEqual(r.state['pipeline_mode'],'passive')
            r.prepare_engine_observer.assert_not_called();r.connect.assert_not_called();r.synthesis_command.assert_not_called()
            self.assertEqual(r.state['played_bytes'],0)

    def test_passive_session_listens_without_opening_or_spoken_budget(self):
        nodes={k:{'id':k,'speaker':{'role':role}} for k,role in [('teacher','teacher'),('student','student'),('guest','guest')]}
        call=Mock();rec=Mock();rec.snapshot.return_value={'phase':'idle'}
        session=DirectTrioSession(nodes,call,Mock(),rec);session.state['id']='session';session.finish=Mock()
        count=0
        def states():
            nonlocal count
            count+=1
            if count>1:session.cancel.set()
            return {k:dict(ready=True,error='',pipeline_mode='passive',room={'protocol':'direct-v1'}) for k in nodes}
        session.states=states
        data={'mode':'passive','topic':'Audio','nodes':{k:{'profile':'test'} for k in nodes}}
        session.run(data)
        self.assertFalse(session.state['error']);self.assertEqual(session.state['mode'],'passive')
        actions=[c.args[1] for c in call.call_args_list]
        self.assertEqual(actions.count('room-listen'),3)
        self.assertNotIn('room-open',actions);self.assertNotIn('room-permit',actions)
        self.assertTrue(all(c.args[2]['room']['mode']=='passive' for c in call.call_args_list if c.args[1]=='start'))
        session.finish.assert_called_once()

    def test_invalid_mode_rejected_before_starting_nodes(self):
        session=DirectTrioSession({},Mock(),Mock(),Mock())
        with self.assertRaisesRegex(ValueError,'Mode'):session.start({'mode':'other'})
        session.call.assert_not_called()


if __name__=='__main__':unittest.main()
