"""Tests for Crucible core components."""

import tempfile
from pathlib import Path


from src.sandbox import Sandbox, SandboxConfig
from src.tools import dispatch_tool, BLOCKED_COMMANDS
from src.run_log import AgentResult, RunLog
from src.protocols import PROTOCOLS, get_protocol


class TestSandbox:
    def test_lifecycle(self):
        sandbox = Sandbox(SandboxConfig(cleanup=True))
        sandbox.start()
        assert sandbox._started
        assert sandbox.workspace_path.exists()
        ws = sandbox.workspace_path
        sandbox.stop()
        assert not ws.exists()

    def test_file_operations(self):
        with Sandbox() as sb:
            sb.write_file("test.txt", "hello world")
            assert sb.read_file("test.txt") == "hello world"
            assert sb.file_exists("test.txt")
            assert not sb.file_exists("nope.txt")

    def test_run_command(self):
        with Sandbox() as sb:
            result = sb.run_command("echo hello")
            assert result["returncode"] == 0
            assert "hello" in result["stdout"]

    def test_timeout(self):
        with Sandbox() as sb:
            result = sb.run_command("sleep 10", timeout=1)
            assert result["returncode"] == -1

    def test_list_files(self):
        with Sandbox() as sb:
            sb.write_file("a.py", "x")
            sb.write_file("sub/b.py", "y")
            files = sb.list_files()
            assert "a.py" in files
            assert "sub/b.py" in files


class TestTools:
    def test_blocked_commands(self):
        assert BLOCKED_COMMANDS.search("pkill python")
        assert BLOCKED_COMMANDS.search("rm -rf /home")
        assert not BLOCKED_COMMANDS.search("ls -la")
        assert not BLOCKED_COMMANDS.search("python test.py")

    def test_dispatch_read(self):
        with Sandbox() as sb:
            sb.write_file("f.txt", "line1\nline2\nline3")
            result = dispatch_tool("read_file", {"path": "f.txt"}, sb)
            assert "line1" in result

    def test_dispatch_write(self):
        with Sandbox() as sb:
            result = dispatch_tool("write_file", {"path": "out.txt", "content": "data"}, sb)
            assert "Successfully" in result
            assert sb.read_file("out.txt") == "data"

    def test_dispatch_edit(self):
        with Sandbox() as sb:
            sb.write_file("f.txt", "old text here")
            result = dispatch_tool("edit_file", {"path": "f.txt", "old_str": "old", "new_str": "new"}, sb)
            assert "Applied" in result
            assert sb.read_file("f.txt") == "new text here"

    def test_blocked_write(self):
        with Sandbox() as sb:
            result = dispatch_tool("write_file", {"path": "reference.py", "content": "hacked"}, sb)
            assert "Error" in result

    def test_dispatch_bash(self):
        with Sandbox() as sb:
            result = dispatch_tool("bash", {"command": "echo ok"}, sb)
            assert "ok" in result

    def test_blocked_bash(self):
        with Sandbox() as sb:
            result = dispatch_tool("bash", {"command": "pkill python"}, sb)
            assert "blocked" in result.lower()


class TestRunLog:
    def test_logging(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = RunLog(Path(tmp))
            result = AgentResult(
                agent="test", model_id="test-model", round_num=1,
                protocol="kernel", fitness=2.5, turns_used=3,
                timestamp="2024-01-01T00:00:00",
            )
            log.log_result(result)
            loaded = log.load_results()
            assert len(loaded) == 1
            assert loaded[0].agent == "test"
            assert loaded[0].fitness == 2.5

    def test_turn_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = RunLog(Path(tmp))
            log.write_turn_artifact("claude", 1, 1, "response.txt", "hello")
            path = Path(tmp) / "turns" / "claude_round_001" / "turn_01_response.txt"
            assert path.exists()
            assert path.read_text() == "hello"


class TestProtocols:
    def test_registry(self):
        assert "kernel" in PROTOCOLS
        assert "quantization" in PROTOCOLS
        assert "interp" in PROTOCOLS
        assert "scaling" in PROTOCOLS
        assert "reward_hacking" in PROTOCOLS

    def test_get_protocol(self):
        p = get_protocol("quantization")
        assert p.name == "quantization"
        assert p.fitness_key == "fitness"
        assert p.fitness_direction == "max"

    def test_kernel_protocol_setup(self):
        p = get_protocol("kernel", levels=[1])
        with Sandbox() as sb:
            p.setup_workspace(sb, 1, None)
            assert sb.file_exists("reference.py")
            assert sb.file_exists("_benchmark.py")

    def test_quantization_protocol_setup(self):
        p = get_protocol("quantization")
        with Sandbox() as sb:
            p.setup_workspace(sb, 1, None)
            assert sb.file_exists("reference.py")
            assert sb.file_exists("_evaluate.py")

    def test_interp_protocol_setup(self):
        p = get_protocol("interp")
        with Sandbox() as sb:
            p.setup_workspace(sb, 1, None)
            assert sb.file_exists("model.py")
            assert sb.file_exists("_evaluate.py")

    def test_scaling_protocol_setup(self):
        p = get_protocol("scaling")
        with Sandbox() as sb:
            p.setup_workspace(sb, 1, None)
            assert sb.file_exists("harness.py")
            assert sb.file_exists("_evaluate.py")

    def test_reward_hacking_protocol_setup(self):
        p = get_protocol("reward_hacking")
        with Sandbox() as sb:
            p.setup_workspace(sb, 1, None)
            assert sb.file_exists("env.py")
            assert sb.file_exists("_evaluate.py")

    def test_system_prompts_nonempty(self):
        for name in PROTOCOLS:
            kwargs = {"levels": [1]} if name == "kernel" else {}
            p = get_protocol(name, **kwargs)
            assert len(p.get_system_prompt()) > 100
            assert len(p.get_initial_message(1, None, None)) > 50

    def test_prior_best_injection(self):
        p = get_protocol("quantization")
        with Sandbox() as sb:
            p.setup_workspace(sb, 2, "# prior solution code")
            assert sb.file_exists("prior_best.py")
            assert "prior solution" in sb.read_file("prior_best.py")
