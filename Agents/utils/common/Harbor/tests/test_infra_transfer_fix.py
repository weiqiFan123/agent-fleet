import asyncio
import base64
import hashlib
import io
import json
import logging
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from test_yicloud_opensandbox import yicloud_opensandbox as mod


def archive_payload(content=b"1", name="reward.txt"):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as tar:
        info = tarfile.TarInfo(name)
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    data = stream.getvalue()
    return hashlib.sha256(data).hexdigest() + ":" + base64.b64encode(data).decode()


def result(stdout="", rc=0):
    return SimpleNamespace(stdout=stdout, stderr="", return_code=rc)


def environment():
    env = object.__new__(mod.YiCloudOpenSandboxEnvironment)
    env.logger = logging.getLogger("transfer-test")
    env._cleanup_remote_exec_paths_sync = Mock()
    return env


class DirectoryTransferTests(unittest.TestCase):
    def run_transfer(self, responses):
        env = environment()
        env.exec = AsyncMock(side_effect=responses)
        with tempfile.TemporaryDirectory() as tmp, patch.object(mod.asyncio, "sleep", new_callable=AsyncMock):
            target = Path(tmp) / "output"
            asyncio.run(env.download_dir("/logs/verifier", target))
            self.assertEqual((target / "reward.txt").read_text(), "1")
        env._cleanup_remote_exec_paths_sync.assert_called_once()
        return env

    def test_success_one_request(self):
        env = self.run_transfer([result(archive_payload())])
        self.assertEqual(env.exec.await_count, 1)
        command = env.exec.call_args.args[0]
        self.assertIn(" && ", command)
        self.assertNotIn(" | ", command)

    def test_empty_response_refetches_same_snapshot_without_tar(self):
        env = self.run_transfer([result(""), result(archive_payload())])
        first, second = [c.args[0] for c in env.exec.call_args_list]
        self.assertTrue(first.endswith(second))
        self.assertNotIn("tar czf", second)
        self.assertEqual(env.exec.await_count, 2)

    def test_checksum_mismatch_is_retried(self):
        bad = "0" * 64 + archive_payload()[64:]
        self.assertEqual(self.run_transfer([result(bad), result(archive_payload())]).exec.await_count, 2)

    def test_truncated_base64_is_retried(self):
        self.run_transfer([result(archive_payload()[:-7]), result(archive_payload())])

    def test_nonzero_refetch_then_success(self):
        self.run_transfer([result(""), result(rc=1), result(archive_payload())])

    def test_persistent_empty_fails_without_touching_target(self):
        env = environment()
        env.exec = AsyncMock(return_value=result(""))
        with tempfile.TemporaryDirectory() as tmp, patch.object(mod.asyncio, "sleep", new_callable=AsyncMock):
            target = Path(tmp) / "output"
            with self.assertRaisesRegex(RuntimeError, "after 3 reads"):
                asyncio.run(env.download_dir("/logs/verifier", target))
            self.assertFalse(target.exists())
        self.assertEqual(env.exec.await_count, 3)
        env._cleanup_remote_exec_paths_sync.assert_called_once()

    def test_snapshot_failure_is_not_reexecuted(self):
        env = environment()
        env.exec = AsyncMock(return_value=result(rc=2))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "snapshot creation"):
                asyncio.run(env.download_dir("/missing", Path(tmp) / "output"))
        self.assertEqual(env.exec.await_count, 1)
        env._cleanup_remote_exec_paths_sync.assert_called_once()

    def test_old_pipeline_masks_missing_source_with_empty_gzip(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = subprocess.run(
                ["sh", "-c", 'tar czf - -C "$1" . | base64 -w0', "sh", str(Path(tmp) / "missing")],
                capture_output=True, text=True, timeout=10,
            )
        self.assertEqual(r.returncode, 0)
        with self.assertRaisesRegex(tarfile.ReadError, "empty file"):
            tarfile.open(fileobj=io.BytesIO(base64.b64decode(r.stdout)), mode="r:gz")

    def test_checked_snapshot_propagates_missing_source(self):
        env = environment()
        with tempfile.TemporaryDirectory() as tmp:
            async def local_exec(command, **_kwargs):
                command = command.replace("/tmp/harbor-download-test.tar.gz", str(Path(tmp) / "snapshot.tar.gz"))
                r = subprocess.run(["sh", "-c", command], capture_output=True, text=True, timeout=10)
                return result(r.stdout, r.returncode)
            env.exec = AsyncMock(side_effect=local_exec)
            with patch.object(mod.uuid, "uuid4", return_value=SimpleNamespace(hex="test")):
                with self.assertRaisesRegex(RuntimeError, "snapshot creation"):
                    asyncio.run(env.download_dir(str(Path(tmp) / "missing"), Path(tmp) / "output"))
            self.assertFalse((Path(tmp) / "output").exists())
            self.assertEqual(env.exec.await_count, 1)

    def test_shell_snapshot_checks_tar_and_preserves_source(self):
        env = environment()
        async def local_exec(command, **_kwargs):
            r = subprocess.run(["sh", "-c", command], capture_output=True, text=True, timeout=10)
            return result(r.stdout, r.returncode)
        env.exec = AsyncMock(side_effect=local_exec)
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source with 'quote"
            source.mkdir()
            (source / "reward.txt").write_text("1")
            # Keep all test snapshots confined to the temporary directory.
            class Identifier:
                hex = "test"
            original_exec = env.exec
            async def confined(command, **kwargs):
                command = command.replace("/tmp/harbor-download-test.tar.gz", str(Path(tmp) / "snapshot.tar.gz"))
                return await original_exec(command, **kwargs)
            env.exec = confined
            with patch.object(mod.uuid, "uuid4", return_value=Identifier()):
                asyncio.run(env.download_dir(str(source), Path(tmp) / "output"))
            self.assertEqual((source / "reward.txt").read_text(), "1")
            self.assertEqual((Path(tmp) / "output/reward.txt").read_text(), "1")

    def test_cancel_propagates_and_cleanup_is_attempted(self):
        env = environment()
        env.exec = AsyncMock(side_effect=asyncio.CancelledError)
        with self.assertRaises(asyncio.CancelledError):
            asyncio.run(env.download_dir("/logs/verifier", "/unused"))
        env._cleanup_remote_exec_paths_sync.assert_called_once()


class ExecFeedbackTests(unittest.TestCase):
    def invoke(self, events):
        env = environment()
        env._command_url = "https://example.invalid/command"
        env._access_token = "fake"
        env._signed_headers = Mock(return_value={})
        response = Mock()
        response.text = "".join("data: " + json.dumps(e) + "\n\n" for e in events)
        session = Mock()
        session.post.return_value = response
        with patch.object(mod.requests, "Session", return_value=session):
            value = env._run_command_direct_sync("ignored", "/", {}, 120)
        self.assertEqual(session.post.call_count, 1)
        return value

    def test_missing_marker_reports_uncertain_completion_without_raw_error(self):
        v = self.invoke([{"type": "error", "error": {"ename": "CommandExecError", "evalue": "secret raw details"}}])
        self.assertEqual(v.return_code, 1)
        self.assertIn("CommandExecError", v.stderr)
        self.assertIn("completion was not confirmed", v.stderr)
        self.assertIn("configured timeout=120s", v.stderr)
        self.assertNotIn("secret", v.stderr)

    def test_normal_nonzero_exit_is_unchanged(self):
        v = self.invoke([{"type": "stdout", "text": "output\n" + mod.EXIT_MARKER + "7\n"}, {"type": "stderr", "text": "original error"}])
        self.assertEqual(v.return_code, 7)
        self.assertEqual(v.stderr, "original error")
        self.assertIn("output", v.stdout)

    def test_missing_marker_preserves_partial_output(self):
        v = self.invoke([{"type": "stdout", "text": "partial"}, {"type": "stderr", "text": "existing error"}])
        self.assertEqual(v.stdout, "partial")
        self.assertTrue(v.stderr.startswith("existing error\n"))


if __name__ == "__main__":
    unittest.main()
