import importlib.util
import math
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("demo_server", Path(__file__).parents[1] / "demo/server.py")
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


class DemoTests(unittest.TestCase):
    def test_patrol_stays_inside_arena_and_lidar_hits_walls(self):
        for t in range(200):
            scene = server.scene(t)
            self.assertTrue(scene["simulated"])
            self.assertTrue(0 < scene["x"] < 12 and 0 < scene["y"] < 8)
            self.assertTrue(math.isfinite(scene["heading"]))
            self.assertEqual(len(scene["points"]), 144)
            for x, y in scene["points"]:
                self.assertTrue(0 <= x <= 12 and 0 <= y <= 8)
                self.assertTrue(x in (0, 12) or y in (0, 8))

    def test_gpu_is_pending_until_real_probe_runs(self):
        status = server.status()
        self.assertEqual(status["gpu"]["result"], "PENDING")
        self.assertTrue(status["simulation"])
        self.assertIn("llm", status)
        self.assertIn("Admiral Edge Transformer", status["llm"]["model"])

    def test_admiral_logo_assets(self):
        index_html = (Path(__file__).parents[1] / "demo/index.html").read_text()
        self.assertIn("https://admrl.co/_app/immutable/assets/admiral-dark.CawzA3qg.svg", index_html)
        self.assertIn("/admiral-logo.svg", index_html)
        logo_svg = (Path(__file__).parents[1] / "demo/admiral-logo.svg").read_text()
        self.assertTrue(logo_svg.startswith("<svg") and logo_svg.strip().endswith("</svg>"))

    def test_chat_interface_present_in_website(self):
        index_html = (Path(__file__).parents[1] / "demo/index.html").read_text()
        self.assertIn("admiral-chat", index_html)
        self.assertIn("chat-window", index_html)
        self.assertIn("chat-input", index_html)
        self.assertIn("prompt-chip", index_html)
        self.assertIn("Run edge LLMs", index_html)

    def test_on_device_llm_inference(self):
        result = server.GLOBAL_MODEL.generate("What is Admiral OS?")
        self.assertIn("Admiral", result["text"])
        self.assertGreater(result["tokens"], 0)
        self.assertGreater(result["tok_per_sec"], 0)
        self.assertIn("backend", result)

        benchmark = server.GLOBAL_MODEL.generate("Run CUDA benchmark")
        self.assertIn("Benchmark", benchmark["text"])
        self.assertGreater(benchmark["tokens"], 30)


if __name__ == "__main__":
    unittest.main()
