from pathlib import Path
import sys
import unittest
sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'script/deviceui'),str(Path(__file__).resolve().parents[1]/'script')]
from session_speakers import SpeakerSession
from room_policy import WAIT_ASSURANCE, accept_response_names, needs_response_name, response_prompt

class IdentityTests(unittest.TestCase):
    def test_title_correction_binds_pending_guest_to_the_same_voice(self):
        s=SpeakerSession([dict(name='Mrs. Byers',embedding=[1,0],trusted_introducer=True)])
        s.observe('t1',[1,0],'I have invited Agent Smith to join us.',clean=True,duration=3)
        self.assertEqual(s.apply(dict(session=s.session_id,kind='introduction',name='Agent Smith',
            track=None,turn='t1',target_turn=None))['status'],'pending_voice')
        guest=s.observe('t2',[0,1],'Hello everyone, I am Major Smith.',clean=True,duration=3)
        with self.assertRaisesRegex(ValueError,'clarify before binding'):
            s.apply(dict(session=s.session_id,kind='self',name='Major Smith',
                track=guest['track_id'],turn='t2',target_turn='t2'))
        correction=s.observe('t3',[0,1],'Agent, Agent Smith.',clean=True,duration=2)
        own=dict(display_name='Chey',role='student')
        self.assertTrue(needs_response_name(s,correction,own))
        prompt=response_prompt(s,correction,own,1,
            dict(own_last_spoken_epoch=0,current_epoch=3,own_last_spoken_text=''),
            decision_tags=True,wait_assurance=True)
        self.assertIn('NAME_UPDATE: correction',prompt)
        self.assertIn('not an address to anyone in the room',prompt)
        self.assertNotIn('explicitly addresses Agent Smith',prompt)
        learning=accept_response_names(s,correction,['Agent Smith'],own)
        self.assertEqual(learning['rejected'],[])
        self.assertEqual(learning['accepted'][0]['kind'],'correction')
        self.assertEqual(s.tracks[guest['track_id']]['name'],'Agent Smith')
        self.assertFalse(s.pending_names)

    def test_title_correction_rebinds_a_misheard_title_but_not_name(self):
        s=SpeakerSession()
        first=s.observe('t1',[1,0],'My name is Major Smith.',clean=True,duration=2)
        s.apply(dict(session=s.session_id,kind='self',name='Major Smith',
                     track=first['track_id'],turn='t1',target_turn='t1'))
        second=s.observe('t2',[1,0],'Major, Agent Smith.',clean=True,duration=2)
        learning=accept_response_names(s,second,['Agent Smith'])
        self.assertEqual(learning['rejected'],[])
        self.assertEqual(s.tracks[first['track_id']]['name'],'Agent Smith')
        wrong=s.observe('t3',[1,0],'Agent, Agent Jones.',clean=True,duration=2)
        self.assertFalse(needs_response_name(s,wrong))
        with self.assertRaisesRegex(ValueError,'No supported self-name correction'):
            s.apply(dict(session=s.session_id,kind='correction',name='Agent Jones',
                         track=first['track_id'],turn='t3',target_turn='t3'))

    def test_first_time_vocative_cannot_claim_another_voice(self):
        s=SpeakerSession()
        unknown=s.observe('t1',[1,0],'Agent, Agent Smith.',clean=True,duration=2)
        self.assertIsNone(s.self_correction_name(unknown))
        self.assertFalse(needs_response_name(s,unknown))

    def test_correction_cannot_rename_a_different_acoustic_voice(self):
        s=SpeakerSession()
        smith=s.observe('t1',[1,0],'My name is Major Smith.',clean=True,duration=2)
        s.apply(dict(session=s.session_id,kind='self',name='Major Smith',
                     track=smith['track_id'],turn='t1',target_turn='t1'))
        other=s.observe('t2',[0,1],'Agent, Agent Smith.',clean=True,duration=2)
        self.assertIsNone(s.self_correction_name(other))
        with self.assertRaises(ValueError):
            s.apply(dict(session=s.session_id,kind='correction',name='Agent Smith',
                         track=smith['track_id'],turn='t2',target_turn='t2'))
        self.assertEqual(s.tracks[smith['track_id']]['name'],'Major Smith')

    def test_pending_name_alone_cannot_make_a_vocative_a_self_correction(self):
        s=SpeakerSession([dict(name='Mrs. Byers',embedding=[1,0],trusted_introducer=True)])
        s.observe('t1',[1,0],'I have invited Agent Smith.',clean=True,duration=2)
        s.apply(dict(session=s.session_id,kind='introduction',name='Agent Smith',
                     track=None,turn='t1',target_turn=None))
        guest=s.observe('t2',[0,1],'Agent, Agent Smith.',clean=True,duration=2)
        self.assertIsNone(s.self_correction_name(guest))
        self.assertFalse(needs_response_name(s,guest))

    def test_static_prompts_separate_title_name_and_allow_brief_correction(self):
        from session_scenario import ClassroomScenario
        scenario=ClassroomScenario()
        smith=scenario.prompt('agent-smith')
        byers=scenario.prompt('catherine-byers')
        chey=scenario.prompt('chey')
        self.assertIn('your title is Agent and your name is Smith',smith)
        self.assertIn('"Agent, Agent Smith."',smith)
        self.assertIn('your title is Mrs. and your name is Byers',byers)
        self.assertIn('Chey has no title',chey)
        for prompt in (smith,byers,chey):
            self.assertIn('correction is allowed after an introduction',prompt)
            self.assertIn('not an address to another person',prompt)
            self.assertIn('allow more variation in the title than in the name',prompt)

    def test_student_who_spoke_without_introducing_is_not_told_she_already_did(self):
        s=SpeakerSession([dict(name='Agent Smith',embedding=[1,0])])
        observation=s.observe('t1',[1,0],
            'An agent uses a model to plan and act.',clean=True,duration=2)
        own=dict(display_name='Chey',role='student')
        prompt=response_prompt(s,observation,own,1,
            dict(own_last_spoken_epoch=1,current_epoch=2,
                 own_last_spoken_text='Hello, Agent Smith. What is an agent?'),
            decision_tags=True,wait_assurance=True)
        self.assertIn('last_own_speech',prompt)
        self.assertIn('If you have not yet told this new guest your name',prompt)
        self.assertNotIn('You have already introduced yourself',prompt)

    def test_wait_assurance_keeps_address_without_forcing_a_reply(self):
        s=SpeakerSession([dict(name='Mrs. Byers',embedding=[1,0],trusted_introducer=True)])
        observation=s.observe('t1',[1,0],
            'Chey, hold that answer until I finish the question.',clean=True,duration=3)
        own=dict(display_name='Chey',role='student')
        history=dict(own_last_spoken_epoch=0,current_epoch=1,own_last_spoken_text='')
        legacy=response_prompt(s,observation,own,0,history,decision_tags=True)
        revised=response_prompt(s,observation,own,0,history,decision_tags=True,
                                wait_assurance=True)
        self.assertIn('Start with [speak] and answer the question addressed to YOU now.',legacy)
        self.assertIn('The current utterance explicitly addresses YOU (Chey).',revised)
        self.assertNotIn('Give your answer.',revised)
        self.assertNotIn('[defer] would be incorrect here.',revised)
        self.assertTrue(revised.endswith(WAIT_ASSURANCE))

    def test_confirmed_introduction_only_removes_catchup_sentence(self):
        s=SpeakerSession([dict(name='Agent Smith',embedding=[1,0])])
        own=dict(display_name='Chey',role='student')
        history=dict(own_last_spoken_epoch=1,current_epoch=2,
                     own_last_spoken_text='Could you give an example?')
        for index,text in enumerate(('A thermostat is an agent.', 'What is your name again?',
                                     'Kai, a thermostat is an agent.')):
            observation=s.observe('t'+str(index),[1,0],text,clean=True,duration=2)
            baseline=response_prompt(s,observation,own,3,history,decision_tags=True)
            candidate=response_prompt(s,observation,own,3,
                dict(history,introduction_confirmed=True),decision_tags=True)
            reminder=(' If you have not yet told this new guest your name, give it briefly '
                      'before your next lesson question.')
            self.assertIn(reminder,baseline)
            self.assertEqual(candidate,baseline.replace(reminder,''))
            self.assertIn('You may briefly correct a mistaken title or name',candidate)

    def test_whisper_hint_gains_guest_name_after_trusted_introduction(self):
        import threading
        from types import SimpleNamespace
        from room_runtime import RoomRuntime
        s=SpeakerSession([dict(name='Mrs. Byers',embedding=[1,0],trusted_introducer=True)])
        device=SimpleNamespace(cfg={},state={},lock=threading.RLock(),cancel=threading.Event(),
            procs={},speaker=dict(display_name='Chey',role='student'),emit=lambda *a,**k:None)
        room=RoomRuntime(device,dict(session='room',topic='Agents'))
        room.identities=s
        room.refresh_asr_names()
        self.assertIn('Chey',room.asr_name_hint)
        self.assertIn('Mrs. Byers',room.asr_name_hint)
        self.assertNotIn('Agent Smith',room.asr_name_hint)
        s.observe('t1',[1,0],"I'm pleased to have Agent Smith joining us as a guest.",clean=True,duration=3)
        s.apply(dict(session=s.session_id,kind='introduction',name='Agent Smith',turn='t1',
                     track=None,target_turn=None))
        room.refresh_asr_names()
        self.assertIn('Agent Smith',room.asr_name_hint)
        self.assertNotIn('Magen Smith',room.asr_name_hint)

    def test_live_guest_welcome_is_trusted_introduction_despite_later_question(self):
        s=SpeakerSession([dict(name='Mrs. Byers',embedding=[1,0],trusted_introducer=True)])
        text=("Hello everyone. Today we're discussing LLMs, agents, and harnesses. "
              "I'm pleased to have Agent Smith joining us as a guest. "
              "Agent Smith, would you like to introduce yourself?")
        s.observe('t1',[1,0],text,clean=True,duration=8)
        proposal=dict(session=s.session_id,kind='introduction',name='Agent Smith',
                      turn='t1',track=None,target_turn=None)
        self.assertEqual(s.apply(proposal)['status'],'pending_voice')
        s.observe('t2',[0,1],'Hello everyone, I am Magen Smith.',clean=True,duration=3)
        mistaken=dict(session=s.session_id,kind='self',name='Magen Smith',
                      turn='t2',track='voice-2',target_turn='t2')
        with self.assertRaisesRegex(ValueError,'clarify before binding'):
            s.apply(mistaken)
        self.assertIsNone(s.tracks['voice-2']['name'])
        self.assertIn('Agent Smith',s.pending_names)
        prompt=response_prompt(s,s.turns['t2'],dict(display_name='Chey',role='student'),0,
            dict(own_last_spoken_epoch=0,current_epoch=2,own_last_spoken_text=''),decision_tags=True)
        self.assertIn('Do not say that uncertain guest name',prompt)
        self.assertIn('ask the guest one short question',prompt)

    def test_teacher_introduction_can_be_followed_by_a_separate_question(self):
        s=SpeakerSession([dict(name='Mrs. Byers',embedding=[1,0],trusted_introducer=True)])
        def proposal(turn):return dict(session=s.session_id,kind='introduction',name='Agent Smith',turn=turn,track=None,target_turn=None)
        s.observe('t1',[1,0],'I have invited Agent Smith. Agent Smith, could you introduce yourself?',clean=True,duration=4)
        self.assertEqual(s.apply(proposal('t1'))['status'],'pending_voice')
        s.observe('t2',[1,0],'Have I invited Agent Smith?',clean=True,duration=2)
        with self.assertRaises(ValueError):s.apply(proposal('t2'))

    def test_known_voice_skips_asr_name_conflicts_but_teacher_introductions_still_learn(self):
        from room_policy import needs_name
        s=SpeakerSession([dict(name='Chey',embedding=[1,0]),
            dict(name='Mrs. Byers',embedding=[0,1],trusted_introducer=True)])
        chey=s.observe('t1',[1,0],'My name is Tri.',clean=True,duration=2)
        self.assertFalse(needs_name(s,chey))
        teacher=s.observe('t2',[0,1],'I have invited Agent Smith.',clean=True,duration=2)
        self.assertTrue(needs_name(s,teacher))

    def test_self_introduction_then_question_and_requested_spelling(self):
        s=SpeakerSession(canonicalize=False)
        first=s.observe('t1',[1,0], 'My name is Shai. What is an agent?', clean=True,duration=3)
        def proposal(kind,name,turn,track):
            return dict(session=s.session_id,kind=kind,name=name,turn=turn,track=track,target_turn=turn)
        learned=s.apply(proposal('self','Shai','t1',first['track_id']))
        self.assertEqual(learned['name'],'Shai')
        self.assertTrue(s.ask_spelling(first['track_id'], 'Could you spell your name for me?'))
        s.observe('t2',[1,0],'C, H, E, Y.',clean=True,duration=2)
        result=s.apply(proposal('spelling','Chey','t2',first['track_id']))
        self.assertEqual(result['name'],'Chey')
        self.assertEqual(s.tracks[first['track_id']]['name'],'Chey')
        s.reset(); self.assertFalse(s.tracks); self.assertIsNone(s.spelling_request)

    def test_unsolicited_or_different_voice_spelling_cannot_rename(self):
        for requested in (False,True):
            s=SpeakerSession(canonicalize=False)
            first=s.observe('t1',[1,0],'My name is Shai.',clean=True,duration=2)
            if requested: s.ask_spelling(first['track_id'],'Please spell your name.')
            other=s.observe('t2',[0,1],'C, H, E, Y.',clean=True,duration=2)
            with self.assertRaises(ValueError):
                s.apply(dict(session=s.session_id,kind='spelling',name='Chey',turn='t2',target_turn='t2',track=other['track_id']))

class RoomPipelineTests(unittest.TestCase):
    def test_context_renewal_preserves_local_identity_and_spoken_history(self):
        import threading,io,struct
        from types import SimpleNamespace
        from room_runtime import RoomRuntime, SocketClient
        device=SimpleNamespace(cfg={},state=dict(synth_count=3),lock=threading.RLock(),
            cancel=threading.Event(),procs={},emit=lambda *a,**k:None)
        room=RoomRuntime(device,dict(session='room',topic='Agents'))
        room.identities=SpeakerSession([dict(name='Mrs. Byers',embedding=[1,0])])
        original=room.identities.snapshot()
        room.context_turns=8
        room.conversation=[dict(epoch=2,voice='voice-1',heard='Chey, what would you try?')]
        room.last_spoken_text='I would use a calculator.'
        room.client=SocketClient.__new__(SocketClient)
        closed=[]
        room.client.close=lambda:closed.append(True)
        stream=io.BytesIO()
        room.connect=lambda:setattr(room,'client',SimpleNamespace(stdin=stream))
        room.silent=lambda *args:self.fail('Context restoration must prefill without generation')
        room.input_prefix='An open input '
        with self.assertRaisesRegex(RuntimeError,'open input'):room.renew_context()
        self.assertFalse(closed)
        room.input_prefix=''
        room.renew_context()
        self.assertTrue(closed)
        data=stream.getvalue();prefix=b''
        while data:
            self.assertEqual(data[:2],b'\x01T')
            size=struct.unpack('<I',data[6:10])[0]
            prefix+=data[10:10+size];data=data[10+size:]
        self.assertIn(b'Chey, what would you try?',prefix)
        self.assertIn(b'I would use a calculator.',prefix)
        self.assertNotIn(b'\n',prefix)
        self.assertEqual(original,room.identities.snapshot())
        self.assertFalse(room.turn_active)
        self.assertEqual(room.state['synth_count'],3)

    def test_clear_introduction_gets_name_only_retry_without_speech_or_turn_count(self):
        import io,json,queue,threading
        from types import SimpleNamespace
        from room_runtime import RoomRuntime
        events=[]
        device=SimpleNamespace(cfg={},state=dict(input_count=0,synth_count=0,model_bytes=0),
            lock=threading.RLock(),cancel=threading.Event(),procs={},
            speaker=dict(display_name='Chey',speaker_id='chey',role='student'),
            emit=lambda kind,text='',**extra:events.append(dict(kind=kind,text=text,**extra)))
        room=RoomRuntime(device,dict(session='room',topic='Agents'))
        room.epoch=1
        room.encoder=SimpleNamespace(pcm=lambda pcm:[0,1])
        room.identities=SpeakerSession([dict(name='Mrs. Byers',embedding=[1,0],trusted_introducer=True)],canonicalize=False)
        room.client=SimpleNamespace(stdin=io.BytesIO())
        replies=queue.Queue()
        proposal=dict(session=room.identities.session_id,kind='self',name='Agent Smith',track='voice-2',turn='e1-h1',target_turn='e1-h1')
        replies.put(json.dumps(proposal).encode()+b'<turn|>')
        replies.put(b'{"action":"speak","reason":"invited"}<turn|>')
        live=SimpleNamespace(finish=lambda cancel:dict(embedding=[0,1],tracks=['voice-2']))
        room.decision(('observe','Hello everyone, I am Agent Smith.',' Agent Smith.',None,b'\x01\x00'*32000,1,live),replies)
        self.assertEqual(room.identities.tracks['voice-2']['name'],'Agent Smith')
        self.assertEqual(device.state['input_count'],0)
        self.assertEqual(device.state['synth_count'],0)
        self.assertEqual(device.state['room']['heard_count'],1)
        self.assertTrue(room.client.stdin.getvalue().startswith(b' Agent Smith.'))
        self.assertTrue(any(e['kind']=='room_identity' and e['learning']['accepted'] for e in events))

    def test_unattributed_speech_never_reaches_turn_decision(self):
        import threading
        from types import SimpleNamespace
        from room_runtime import RoomRuntime
        device=SimpleNamespace(cfg={},state=dict(synth_count=0),lock=threading.RLock(),
            cancel=threading.Event(),procs={},emit=lambda *a,**k:None)
        room=RoomRuntime(device,dict(session='room',topic='Agents'))
        room.identities=SpeakerSession()
        room.silent=lambda *args:self.fail('Unattributed input reached the LLM decision')
        live=SimpleNamespace(finish=lambda cancel:dict(embedding=None,tracks=[]))
        with self.assertRaisesRegex(RuntimeError,'turn decision withheld'):
            room.decision(('observe','Who should answer?','Who should answer?',{},b'\0\0'*16000,1,live),None)


class LiveIdentityTests(unittest.TestCase):
    def make(self, vectors):
        import threading
        from types import SimpleNamespace
        from room_identity import LiveUtterance
        vectors=iter(vectors)
        def submit(ticket,job):
            if job is None: ticket.done.set()
            else: ticket.result(next(vectors),job,.01)
        events=[]
        session=SpeakerSession([dict(name='Mrs. Byers',embedding=[1,0])])
        live=LiveUtterance(SimpleNamespace(submit=submit,error=None),session,threading.RLock(),events.append)
        return live,session,events

    def feed(self, live, seconds, rms=1000):
        for _ in range(round(seconds/.02)): live.feed(b'\x01\0'*320,rms)

    def test_original_cadence_two_confirmations_and_no_prior_reinforcement(self):
        live,session,events=self.make([[1,0]]*6)
        self.feed(live,.5)
        self.assertFalse(any(e['status']=='identified' for e in events))
        self.feed(live,.26)
        self.assertEqual(events[-1]['status'],'identified')
        self.assertEqual(events[-1]['name'],'Mrs. Byers')
        self.assertEqual(session.tracks['voice-1']['observations'],0)
        self.assertFalse(session.turns)
        self.feed(live,.36,0)
        self.assertEqual(events[-1]['status'],'quiet')
        self.feed(live,.5)
        self.assertNotEqual(events[-1]['status'],'identified')

    def test_short_unknown_prefix_does_not_create_a_fragmented_voice_track(self):
        live,session,events=self.make([[0,1]]*2)
        self.feed(live,.76)
        self.assertIsNone(events[-1]['track_id'])
        self.assertIsNone(events[-1]['name'])
        self.assertEqual(events[-1]['status'],'unknown')
        self.assertEqual(len(session.tracks),1)
        self.assertFalse(session.turns)

    def test_two_sequential_voices_within_one_transcript_withhold_attribution(self):
        import threading
        live,session,events=self.make([[1,0]]*3+[[0,1]]*2+[[.5,.5],[0,1]])
        self.feed(live,.76); self.feed(live,.36,0); self.feed(live,.76)
        live.seal()
        result=live.finish(threading.Event())
        self.assertEqual(result['status'],'unattributed')
        self.assertIsNone(result['embedding'])

    def test_delayed_or_cancelled_match_never_restores_current_identity(self):
        live,session,events=self.make([[1,0]]*3)
        self.feed(live,.76); self.feed(live,.36,0)
        count=len(events)
        import time
        live.result([1,0],dict(segment=1,end_s=.7,duration=.7,submitted=time.monotonic()),.01)
        self.assertEqual(len(events),count)
        live.cancelled=True
        live.result([0,1],dict(segment=2,end_s=1,duration=1,submitted=time.monotonic()),.01)
        self.assertEqual(len(session.tracks),1)

    def test_quiet_checkpoint_overlaps_buffering_without_enrolling_silence_again(self):
        import threading
        from types import SimpleNamespace
        from room_identity import LiveUtterance
        jobs=[]
        def submit(ticket,job):
            if job is None: ticket.done.set()
            else: jobs.append(job)
        live=LiveUtterance(SimpleNamespace(submit=submit,error=None),SpeakerSession(),
                           threading.RLock(),lambda e:None)
        self.feed(live,1.2)
        self.feed(live,.36,0)
        enroll=[j for j in jobs if j['kind']=='enrollment']
        self.assertEqual(len(enroll),1)
        self.assertAlmostEqual(enroll[0]['duration'],1.56)
        self.assertFalse(live.closed)
        self.assertFalse(live.done.is_set())
        # Complete the pending CPU work during the ASR endpoint buffer.
        for job in jobs: live.result([1,0],job,.05)
        self.feed(live,2,0)  # cross the periodic 3 s boundary with silence only
        count=len(jobs)
        live.seal()
        self.assertEqual(len(jobs),count)
        result=live.finish(threading.Event())
        self.assertEqual(result['status'],'new_voice')

    def test_speech_resuming_after_checkpoint_requires_new_enrollment(self):
        import threading
        live,session,events=self.make([[1,0]]*20)
        self.feed(live,1);self.feed(live,.36,0)
        old=live.enrollment_size
        self.feed(live,.8)
        self.assertGreater(live.last_voice,live.enrollment_voice)
        live.seal()  # forced endpoint cannot reuse the pre-resumption checkpoint
        self.assertGreater(live.enrollment_size,old)
        self.assertEqual(live.enrollment_voice,live.last_voice)
        self.assertEqual(live.finish(threading.Event())['status'],'single_voice')

    def test_brief_speech_enrolls_when_minimum_audio_arrives_during_quiet(self):
        live,session,events=self.make([[1,0]])
        self.feed(live,.24);self.feed(live,.36,0)
        self.assertIsNone(live.enrollment)
        self.feed(live,.16,0)
        self.assertIsNotNone(live.enrollment)
        self.assertAlmostEqual(live.enrollment_size/32000,.76)
        self.assertFalse(live.closed)

    def segment_resolution(self, segment_vector, second_known=False):
        import threading
        from types import SimpleNamespace
        from room_identity import LiveUtterance
        jobs=[]
        def submit(ticket,job):
            if job is None:ticket.done.set();return
            jobs.append(job['kind'])
            vector=segment_vector if job['kind']=='segment' else (
                [-1,0] if job['kind']=='live' and job['segment']==2 else [1,0])
            ticket.result(vector,job,.01)
        known=[dict(name='Mrs. Byers',embedding=[1,0])]
        if second_known: known.append(dict(name='Agent Smith',embedding=[0,1]))
        live=LiveUtterance(SimpleNamespace(submit=submit,error=None),
            SpeakerSession(known),threading.RLock(),lambda e:None)
        self.feed(live,1.2);self.feed(live,.36,0)
        self.feed(live,1.2)
        self.assertEqual(live.unresolved,{2})
        self.feed(live,.36,0)
        self.assertIn('segment',jobs)  # scheduled before final ASR admission
        before=len(jobs);live.seal()
        self.assertEqual(len(jobs),before)
        return live.finish(threading.Event())

    def test_weak_prefix_after_pause_resolves_from_its_own_longer_audio(self):
        result=self.segment_resolution([1,0])
        self.assertEqual(result['status'],'single_voice')
        self.assertEqual(result['tracks'],['voice-1'])
        self.assertFalse(result['unresolved_segments'])

    def test_mixed_utterance_cannot_use_first_voice_to_erase_unknown_second_segment(self):
        result=self.segment_resolution([0,1])
        # Whole-utterance embedding matches the first voice, but the resumed
        # segment does not: never attribute both speakers to that first voice.
        self.assertEqual(result['enrollment_match']['status'],'matched')
        self.assertEqual(result['unresolved_segments'],[2])
        self.assertIsNone(result['embedding'])

    def test_longer_segment_conflicting_with_known_voice_still_blocks(self):
        result=self.segment_resolution([0,1],second_known=True)
        self.assertEqual(result['status'],'multiple_voices')
        self.assertEqual(result['confirmed_tracks'],['voice-1','voice-2'])
        self.assertIsNone(result['embedding'])

    def test_uncertain_longer_segment_still_blocks(self):
        result=self.segment_resolution([.35,(1-.35**2)**.5])
        self.assertEqual(result['status'],'unattributed')
        self.assertEqual(result['unresolved_segments'],[2])
        self.assertIsNone(result['embedding'])

if __name__=='__main__': unittest.main()
