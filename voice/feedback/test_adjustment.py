import json,tempfile,unittest,urllib.request,urllib.error
from types import SimpleNamespace
from controller import Controller,LIMITS
from openai_adjustment import Adjuster
from adjustment_http import start_server
class AdjustmentTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.c=Controller(self.tmp.name)
  self.run=self.c.draw({'content':'smiley','request_id':'draw1'})
  self.c.state['runs'][self.run['run_id']].update(status='completed',physical=True)
  self.c.feedback({'run_id':self.run['run_id'],'accepted':False,'feedback':'lift higher'})
  self.data={'action':'propose','message':'Raise clearance 5 mm','changes':dict.fromkeys(LIMITS)};self.data['changes']['lift_delta_mm']=5;self.calls=0
  def create(**kw):
   self.calls+=1;self.assertTrue(kw['text']['format']['strict']);return SimpleNamespace(status='completed',output_text=json.dumps(self.data))
  self.a=Adjuster(self.c,SimpleNamespace(responses=SimpleNamespace(create=create)),model='test-model')
  self.p={'run_id':self.run['run_id'],'feedback':'lift higher','request_id':'adjust1'}
 def tearDown(self):self.tmp.cleanup()
 def test_prepare_and_retry(self):
  result=self.a.request(self.p);self.assertEqual(result['status'],'prepared_not_executed');self.assertEqual(self.a.request(self.p),result);self.assertEqual(self.calls,1);self.assertIsNone(self.c.proc)
 def test_bad_change(self):
  self.data['changes']['lift_delta_mm']=100
  with self.assertRaises(ValueError):self.a.request(self.p)
 def test_clarification(self):
  self.data['action']='clarify';self.assertEqual(self.a.request(self.p)['status'],'needs_clarification')
 def test_http_auth(self):
  server=start_server(self.a,'x'*32,port=0);url='http://127.0.0.1:%s/request-adjustment'%server.server_port
  try:
   req=urllib.request.Request(url,data=json.dumps(self.p).encode(),method='POST')
   with self.assertRaises(urllib.error.HTTPError) as err:urllib.request.urlopen(req)
   self.assertEqual(err.exception.code,401);self.assertEqual(self.calls,0)
   req.add_header('Authorization','Bearer '+'x'*32)
   with urllib.request.urlopen(req) as response:self.assertEqual(json.load(response)['status'],'prepared_not_executed')
  finally:server.shutdown();server.server_close()
if __name__=='__main__':unittest.main()
