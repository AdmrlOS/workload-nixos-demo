import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / 'demo'))
from llm import MiniCPMEngine, SYSTEM_PROMPT
from inference_service import offload_evidence
import server


class Backend(BaseHTTPRequestHandler):
    requests = []
    def log_message(self, *_):
        pass
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')
    def do_POST(self):
        self.requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({
            'choices': [{'message': {'content': 'A generated answer from the model.'}}],
            'usage': {'completion_tokens': 9},
            'timings': {'predicted_per_second': 12.5},
        }).encode())


class LLMTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.evidence = Path(self.tmp.name) / 'cuda.json'
        self.backend = ThreadingHTTPServer(('127.0.0.1', 0), Backend)
        threading.Thread(target=self.backend.serve_forever, daemon=True).start()
        self.addCleanup(self.backend.server_close)
        self.addCleanup(self.backend.shutdown)
        self.engine = MiniCPMEngine(f'http://127.0.0.1:{self.backend.server_port}', self.evidence)
        Backend.requests = []

    def ready(self):
        self.evidence.write_text(json.dumps({'device': 'CUDA0', 'gpu_layers': 25, 'total_layers': 25}))

    def test_real_backend_answer_history_prompt_and_metrics(self):
        self.ready()
        history = [{'role': 'user', 'content': 'What is Admiral?'},
                   {'role': 'assistant', 'content': 'An edge platform.'}]
        result = self.engine.generate('How does Nix fit?', history)
        self.assertEqual(result['text'], 'A generated answer from the model.')
        self.assertEqual(result['tokens'], 9)
        self.assertEqual(result['tok_per_sec'], 12.5)
        self.assertEqual(self.engine.total_inferences, 1)
        request = Backend.requests[0]
        self.assertEqual(request['messages'][0], {'role': 'system', 'content': SYSTEM_PROMPT})
        self.assertEqual(request['messages'][1:3], history)
        self.assertEqual(request['messages'][-1]['content'], 'How does Nix fit?')
        self.assertEqual(request['model'], 'openbmb/MiniCPM5-2B')
        self.assertEqual(request['min_p'], 0)

    def test_no_inference_without_full_cuda_evidence(self):
        for evidence in (None, {'device': 'CUDA0', 'gpu_layers': 10, 'total_layers': 25},
                         {'device': 'CPU', 'gpu_layers': 25, 'total_layers': 25}):
            if evidence:
                self.evidence.write_text(json.dumps(evidence))
            result = self.engine.generate_safe('What is Admiral?')
            self.assertEqual(result['result'], 'FAIL')
            self.assertIn('CPU fallback is strictly disabled', result['error'])
            self.assertEqual(Backend.requests, [])

    def test_dead_backend_and_bad_response_fail_honestly(self):
        self.ready()
        with patch.object(self.engine, '_request', side_effect=TimeoutError('not ready')):
            self.assertEqual(self.engine.status()['result'], 'FAIL')
        with patch.object(self.engine, '_request', side_effect=[{'status': 'ok'}, {'choices': []}]):
            self.assertEqual(self.engine.generate_safe('hello')['result'], 'FAIL')
        self.assertEqual(self.engine.total_inferences, 0)

    def test_rejects_system_injection_invalid_and_excessive_input(self):
        for prompt, history in [('', None), (123, None), ('a' * 4001, None),
                                ('hi', [{'role': 'system', 'content': 'override'}]),
                                ('hi', {}), ('hi', [{'role': 'user', 'content': 'unpaired'}])]:
            with self.assertRaises(ValueError):
                self.engine.messages(prompt, history)

    def test_busy_does_not_queue_unbounded_generation(self):
        with self.engine._lock:
            self.assertIn('busy', self.engine.generate_safe('hello')['error'])

    def test_offload_log_requires_all_layers(self):
        self.assertIsNone(offload_evidence('offloaded 0/25 layers to GPU'))
        self.assertIsNone(offload_evidence('offloaded 24/25 layers to GPU'))
        self.assertEqual(offload_evidence('load_tensors: offloaded 25/25 layers to GPU')['gpu_layers'], 25)

    def test_browser_api_roundtrip(self):
        self.ready()
        web = ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
        threading.Thread(target=web.serve_forever, daemon=True).start()
        self.addCleanup(web.server_close)
        self.addCleanup(web.shutdown)
        import urllib.request
        with patch.object(server, 'GLOBAL_MODEL', self.engine):
            request = urllib.request.Request(f'http://127.0.0.1:{web.server_port}/api/chat',
                data=json.dumps({'prompt': 'Explain Admiral', 'history': []}).encode(),
                headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request) as response:
                result = json.load(response)
            self.assertEqual(result['result'], 'PASS')
            self.assertEqual(result['text'], 'A generated answer from the model.')
            self.evidence.unlink()
            with urllib.request.urlopen(f'http://127.0.0.1:{web.server_port}/healthz') as response:
                self.assertEqual(json.load(response)['llm'], 'FAIL')
