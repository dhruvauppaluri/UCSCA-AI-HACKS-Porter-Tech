"""Authenticated adjustment-only endpoint attached to the active voice controller."""
import hmac,json,threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer

def start_server(adjuster,token,host='127.0.0.1',port=8780):
 if not token or len(token)<24:raise ValueError('ADJUSTMENT_API_TOKEN must contain at least 24 characters')
 class Handler(BaseHTTPRequestHandler):
  def reply(self,status,value):
   body=json.dumps(value).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
  def do_POST(self):
   if self.path!='/request-adjustment':return self.reply(404,{'error':'Not found'})
   if not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+token):return self.reply(401,{'error':'Unauthorized'})
   try:
    n=int(self.headers.get('Content-Length','0'))
    if not 0<n<=8192:return self.reply(413,{'error':'Body must be 1..8192 bytes'})
    self.connection.settimeout(10)
    p=json.loads(self.rfile.read(n))
    if not isinstance(p,dict):raise ValueError('JSON object required')
    self.reply(200,adjuster.request(p))
   except (ValueError,KeyError,TypeError):self.reply(400,{'error':'Invalid request, run state, or rejected model correction. Check run status and supported parameters.'})
   except Exception:self.reply(502,{'error':'Adjustment service failed; no drawing was started.'})
  def log_message(self,*args):pass
 server=ThreadingHTTPServer((host,port),Handler);threading.Thread(target=server.serve_forever,daemon=True).start();return server
