"""Played self-introductions may expire only the catch-up reminder."""
import io
import queue
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from test_room_direct import runtime
from room_runtime import SocketClient


def student():
    r=runtime()
    r.speaker=dict(speaker_id='chey',display_name='Chey',role='student')
    r.introduction_reminder='spoken_evidence'
    return r


class IntroductionTests(unittest.TestCase):
    def test_literal_own_name_only_and_completion_required(self):
        observation=dict(track_id='voice-2',turn_id='h2')
        record=dict(id='2',spoken=True,status='spoken')
        for text,expected in [
            ('My name is Chey. What is an agent?',True),
            ("Hello, I'm Chey. What is an agent?",True),
            ('I am Chey.',True),('Call me Chey.',True),
            ('My name is Kai.',False),('My name is Cheyenne.',False),
            ('He said my name is Chey.',False),('"My name is Chey" is an example.',False),
            ('My name is Chey?',False),('My name is Chey if we are pretending.',False),
            ('Could you give an example?',False)]:
            with self.subTest(text=text):
                r=student();r.remember_introduction(text,observation,record)
                self.assertEqual(r.introduction_confirmed(observation),expected)
        for changes in [dict(spoken=False),dict(status='interrupted'),dict(status='invalid'),
                        dict(status='generating'),dict(status='budget-denied')]:
            r=student();r.remember_introduction('My name is Chey.',observation,dict(record,**changes))
            self.assertFalse(r.introduction_confirmed(observation))
        r=student();r.cancel.set();r.remember_introduction('My name is Chey.',observation,record)
        self.assertFalse(r.introduction_confirmed(observation))

    def test_receipt_survives_renewal_but_is_scoped_to_voice_and_session(self):
        r=student();observation=dict(track_id='voice-2',turn_id='h2')
        r.remember_introduction('My name is Chey.',observation,dict(id='2',spoken=True,status='spoken'))
        r.client=SocketClient.__new__(SocketClient);r.client.close=lambda:None
        r.connect=lambda:setattr(r,'client',SimpleNamespace(stdin=io.BytesIO()))
        r.context_turns=8;r.reset_context()
        self.assertTrue(r.introduction_confirmed(observation))
        self.assertFalse(r.introduction_confirmed(dict(track_id='voice-3')))
        self.assertFalse(student().introduction_confirmed(observation))
        r.introduction_reminder='legacy'
        self.assertFalse(r.introduction_confirmed(observation))

    def test_real_response_path_records_only_completed_permitted_playback(self):
        for permission,interrupted,wire,expected in [
            (True,False,b'[speak]My name is Chey. What is an agent?<turn|>',True),
            (False,False,b'[speak]My name is Chey. What is an agent?<turn|>',False),
            (True,True,b'[speak]My name is Chey. What is an agent?<turn|>',False),
            (True,False,b'[defer]<turn|>',False)]:
            with self.subTest(permission=permission,interrupted=interrupted,wire=wire):
                r=student();r.text_decision_tags=True
                r.cfg=dict(room_identity={},name_pronunciations={})
                r.state.update(input_count=0,reply_count=0,synth_bytes=0,played_bytes=0,
                               audible_until=0,history=[],capture_active=False)
                r.frames=queue.Queue();r.replies=queue.Queue();r.speech_rate=16000
                r.synth_sent=r.synth_completed=0
                observation=r.identities.observe('h1',[1,0],'A thermostat is an agent.',clean=True,duration=2)
                def synthesize(data):
                    r.current['permission']=permission
                    r.frames.put((b'P',bytes(640)))
                    r.frames.put((b'D',str(r.synth_sent).encode()))
                r.synth=Mock(stdin=Mock(write=synthesize))
                stream=Mock(latency=0)
                if interrupted:stream.write.side_effect=lambda data:r.yielded.set()
                r.open_output=Mock(return_value=stream)
                r.replies.put(wire)
                r.respond('Heard','Heard',dict(admitted_at=0),observation,0)
                self.assertEqual(r.introduction_confirmed(observation),expected)
                if expected:
                    stream.stop.assert_called_once()
                    self.assertEqual(r.introduced_to[observation['track_id']]['text'],'My name is Chey')


if __name__=='__main__':unittest.main()
