import tempfile,unittest
from controller import Controller
class Workflow(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.c=Controller(self.tmp.name)
 def tearDown(self):self.tmp.cleanup()
 def test_dry_run_cannot_earn_reward(self):
  r=self.c.draw({'content':'smiley face','request_id':'a'})
  self.assertEqual(r['status'],'dry_run')
  self.assertEqual(self.c.draw({'content':'smiley face','request_id':'a'})['run_id'],r['run_id'])
  with self.assertRaises(ValueError):self.c.feedback({'run_id':r['run_id'],'accepted':True})
 def completed(self):
  r=self.c.draw({'content':'smiley','request_id':'b'});self.c.state['runs'][r['run_id']].update(status='completed',physical=True);return r
 def test_rating_persists_and_is_once(self):
  r=self.completed();self.c.feedback({'run_id':r['run_id'],'accepted':True})
  with self.assertRaises(ValueError):self.c.feedback({'run_id':r['run_id'],'accepted':True})
  other=Controller(self.tmp.name);self.assertEqual(other.state['preferred']['smiley'],r['revision_id'])
 def test_bounds_and_generated_candidate(self):
  r=self.completed();self.c.feedback({'run_id':r['run_id'],'accepted':False,'feedback':'lift higher'})
  with self.assertRaises(ValueError):self.c.propose({'run_id':r['run_id'],'changes':{'lift_delta_mm':200}})
  p=self.c.propose({'run_id':r['run_id'],'changes':{'lift_delta_mm':5}})
  self.assertLess(p['report']['max_ik_residual_mm'],.1)
  self.assertNotIn('smiley',self.c.state['preferred'])
 def test_wrong_routine(self):
  with self.assertRaises(ValueError):self.c.draw({'content':'rm -rf /','request_id':'c'})
if __name__=='__main__':unittest.main()
