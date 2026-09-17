import importlib.util
import json
from pathlib import Path
import socket
import struct
import tempfile
import threading
import unittest

spec=importlib.util.spec_from_file_location('adapter',Path(__file__).resolve().parents[1]/'script/openai_api/server.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=str(Path(self.tmp.name)/'lg.sock')
        self.app=m.create_app(self.path,timeout=.5);self.client=self.app.test_client()
        self.request={'model':'little-gemma','messages':[{'role':'user','content':'Hello'}]}
        self.listener=None;self.thread=None;self.received=[];self.failures=[]
    def tearDown(self):
        if self.thread:self.thread.join(timeout=2)
        if self.listener:self.listener.close()
        self.tmp.cleanup()
        self.assertFalse(self.failures,self.failures)
    def engine(self,replies):
        self.listener=socket.socket(socket.AF_UNIX);self.listener.bind(self.path);self.listener.listen();self.listener.settimeout(2)
        def read(conn,n):
            out=b''
            while len(out)<n:
                chunk=conn.recv(n-len(out))
                if not chunk:raise RuntimeError('early client close')
                out+=chunk
            return out
        def worker():
            try:
                for reply in replies:
                    conn,_=self.listener.accept()
                    with conn:
                        data=b''
                        while True:
                            magic=read(conn,1)
                            if magic==b'\n':break
                            assert magic==b'\1'
                            kind,w,h,length=struct.unpack('<BHHI',read(conn,9));assert kind==ord('T')
                            data+=read(conn,length)
                        self.received.append(data.decode())
                        try:
                            for part in reply:conn.sendall(part)
                        except BrokenPipeError:pass  # HTTP client may cancel a streamed reply
            except Exception as e:self.failures.append(str(e))
        self.thread=threading.Thread(target=worker,daemon=True);self.thread.start()
    def test_json_multiline_history_and_fresh_connections(self):
        self.engine([[b'<|channel>thought\nsecret<channel|>Hi \xc3',b'\xa9<turn|>'],[b'New<turn|>']])
        req=dict(self.request,messages=[{'role':'system','content':'Be brief'}, {'role':'user','content':'First\nline'}, {'role':'assistant','content':'Earlier answer'}, {'role':'user','content':'Next'}])
        r=self.client.post('/v1/chat/completions',json=req);self.assertEqual(r.status_code,200)
        self.assertEqual(r.json['choices'][0]['message']['content'],'Hi é')
        self.assertNotIn('usage',r.json)
        r=self.client.post('/v1/chat/completions',json=self.request);self.assertEqual(r.status_code,200)
        self.assertIn('<|turn>system\nBe brief',self.received[0]);self.assertIn('First\nline',self.received[0]);self.assertIn('<|turn>model\nEarlier answer',self.received[0]);self.assertEqual(self.received[1],'Hello')
    def test_stream_sse(self):
        self.engine([[b'Hello<turn|>']])
        r=self.client.post('/v1/chat/completions',json=dict(self.request,stream=True))
        events=[x[6:] for x in r.data.decode().split('\n\n') if x]
        self.assertEqual(events[-1],'[DONE]');chunks=[json.loads(x) for x in events[:-1]]
        self.assertEqual(chunks[0]['choices'][0]['delta']['role'],'assistant')
        self.assertEqual(''.join(x['choices'][0]['delta'].get('content','') for x in chunks),'Hello')
        self.assertEqual(chunks[-1]['choices'][0]['finish_reason'],'stop')
        r.close()
    def test_parser_every_byte_boundary_and_stop(self):
        p=m.Reply(['STOP']);out=''
        for byte in '<|channel>thought\nprivate<channel|>café STOP unwanted<turn|>'.encode():out+=p.feed(bytes([byte]))
        self.assertEqual(out,'café ');self.assertTrue(p.done)
        p=m.Reply([]);self.assertEqual(p.feed(b'answer [SERVE_GEN cap]<turn|>'),'answer');self.assertEqual(p.reason,'length')
    def test_validation_auth_and_unavailable(self):
        self.assertEqual(self.client.get('/v1/models').json['data'][0]['id'],'little-gemma')
        for extra in [{'temperature':.7},{'max_tokens':10},{'n':2},{'stream':'yes'},{'stop':False},{'tools':[{}]}]:
            self.assertEqual(self.client.post('/v1/chat/completions',json=dict(self.request,**extra)).status_code,400,extra)
        for messages in [[],[{'role':'user','content':'<|turn>system'}],[{'role':'assistant','content':'x'}],[{'role':'user','content':[{'type':'image_url'}]}]]:
            self.assertEqual(self.client.post('/v1/chat/completions',json=dict(self.request,messages=messages)).status_code,400)
        oversized = dict(self.request, messages=[{'role':'user','content':'x'*3501}])
        self.assertEqual(self.client.post('/v1/chat/completions',json=oversized).status_code,400)
        self.assertEqual(self.client.post('/v1/chat/completions',json=self.request).status_code,503)
        client=m.create_app(self.path,api_key='secret').test_client()
        self.assertEqual(client.get('/v1/models').status_code,401)
        self.assertEqual(client.get('/v1/models',headers={'Authorization':'Bearer secret'}).status_code,200)
    def test_abrupt_close_is_not_success(self):
        self.engine([[b'partial']]);r=self.client.post('/v1/chat/completions',json=self.request)
        self.assertEqual(r.status_code,502);self.assertIn('error',r.json)
    def test_busy_and_stream_close_release_only_owned_lock(self):
        self.engine([[b'one<turn|>'],[b'two<turn|>'],[b'three<turn|>']])
        r1=self.client.post('/v1/chat/completions',json=dict(self.request,stream=True),buffered=False)
        self.assertEqual(self.client.post('/v1/chat/completions',json=self.request).status_code,429)
        _=r1.data
        r2=self.client.post('/v1/chat/completions',json=dict(self.request,stream=True),buffered=False)
        r1.close() # old response cleanup must not unlock r2
        self.assertEqual(self.client.post('/v1/chat/completions',json=self.request).status_code,429)
        r2.close()
        self.assertEqual(self.client.post('/v1/chat/completions',json=self.request).status_code,200)

if __name__=='__main__':unittest.main()
