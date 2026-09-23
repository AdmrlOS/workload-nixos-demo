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
        self.assertEqual(status["llm"]["model"], "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertFalse(status["llm"]["cpu_fallback"])
        self.assertTrue(status["llm"]["system_prompt_configured"])

    def test_admiral_logo_assets(self):
        index_html = (Path(__file__).parents[1] / "demo/index.html").read_text()
        self.assertIn("https://admrl.co/_app/immutable/assets/admiral-light.DiddamFK.svg", index_html)
        self.assertIn("/admiral-logo.svg", index_html)
        logo_svg = (Path(__file__).parents[1] / "demo/admiral-logo.svg").read_text()
        self.assertTrue(logo_svg.startswith("<svg") and logo_svg.strip().endswith("</svg>"))

    def test_chat_interface_present_in_website(self):
        index_html = (Path(__file__).parents[1] / "demo/index.html").read_text()
        self.assertIn("admiral-chat", index_html)
        self.assertIn("Qwen2.5-0.5B-Instruct", index_html)
        self.assertIn("chat-window", index_html)
        self.assertIn("chat-input", index_html)
        self.assertIn("prompt-chip", index_html)
        self.assertIn("Run edge LLMs", index_html)

    def test_qwen_system_prompt_and_chatml(self):
        prompt = server.GLOBAL_MODEL.format_chatml("What is Admiral?")
        self.assertIn("<|im_start|>system", prompt)
        self.assertIn("Admiral (admrl.co)", prompt)
        self.assertIn("Qwen2.5", prompt)
        self.assertIn("<|im_start|>user\nWhat is Admiral?\n<|im_end|>", prompt)
        self.assertIn("<|im_start|>assistant", prompt)

    def test_strictly_no_cpu_fallback(self):
        # On hosts without NVIDIA driver, engine must explicitly fail, never fall back to CPU
        if not server.GLOBAL_MODEL.cuda.available:
            with self.assertRaises(RuntimeError) as ctx:
                server.GLOBAL_MODEL.generate("What is Admiral?")
            self.assertIn("CPU fallback is strictly disabled", str(ctx.exception))

            safe_res = server.GLOBAL_MODEL.generate_safe("What is Admiral?")
            self.assertEqual(safe_res["result"], "FAIL")
            self.assertFalse(safe_res["cuda_active"])
            self.assertIn("CPU fallback is strictly disabled", safe_res["error"])


if __name__ == "__main__":
    unittest.main()
