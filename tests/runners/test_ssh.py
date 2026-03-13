"""Tests for SSH runner — run_id encoding, sentinel deploy, event fetching."""

import json
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from daedalus.runners.ssh import SSHRunner, SSHConfig
from daedalus.runners.base import RunStatus


@pytest.fixture
def ssh_config():
    return SSHConfig(
        host="10.0.0.1",
        user="ubuntu",
        remote_work_dir="/tmp/daedalus",
        python_path="python3",
    )


@pytest.fixture
def runner(ssh_config):
    return SSHRunner(ssh_config)


class TestRunIdEncoding:
    def test_encode_run_id(self, runner):
        run_id = runner._encode_run_id("/tmp/daedalus/exp_001")
        assert run_id == "ssh:10.0.0.1:/tmp/daedalus/exp_001"

    def test_decode_run_id(self):
        host, remote_dir = SSHRunner._decode_run_id("ssh:10.0.0.1:/tmp/daedalus/exp_001")
        assert host == "10.0.0.1"
        assert remote_dir == "/tmp/daedalus/exp_001"

    def test_decode_run_id_with_port_in_path(self):
        """Paths with colons should work (split on first 2 colons only)."""
        host, remote_dir = SSHRunner._decode_run_id("ssh:myhost:/data/exp:special")
        assert host == "myhost"
        assert remote_dir == "/data/exp:special"

    def test_decode_legacy_run_id_raises(self):
        with pytest.raises(ValueError, match="Cannot decode"):
            SSHRunner._decode_run_id("ssh_12345")

    def test_run_id_is_stateless(self, runner):
        """Run ID contains all info needed for cross-session recovery."""
        run_id = runner._encode_run_id("/tmp/daedalus/exp_007")
        host, remote_dir = SSHRunner._decode_run_id(run_id)

        # Can reconstruct everything from run_id alone
        assert host == runner.config.host
        assert remote_dir == "/tmp/daedalus/exp_007"


class TestPoll:
    def test_poll_running(self, runner):
        """Poll returns running when status.json says running."""
        run_id = "ssh:10.0.0.1:/tmp/daedalus/exp_001"

        with patch.object(runner, "_run_ssh") as mock_ssh:
            mock_ssh.return_value = MagicMock(
                returncode=0,
                stdout='{"state":"running","pid":1234}\n---SEPARATOR---\n'
                       '{"event":"METRIC","metric":"loss","value":0.5}\n'
                       '{"event":"PROGRESS","key":"epoch","value":"2/10"}\n',
            )

            status = runner.poll(run_id)
            assert status.state == "running"
            # Should extract progress from events
            assert status.progress is not None

    def test_poll_completed_from_events(self, runner):
        """Poll detects completion from sentinel COMPLETED event."""
        run_id = "ssh:10.0.0.1:/tmp/daedalus/exp_001"

        with patch.object(runner, "_run_ssh") as mock_ssh:
            mock_ssh.return_value = MagicMock(
                returncode=0,
                stdout='{"state":"running","pid":1234}\n---SEPARATOR---\n'
                       '{"event":"COMPLETED","pid":1234}\n',
            )

            status = runner.poll(run_id)
            assert status.state == "completed"

    def test_poll_completed_from_status(self, runner):
        """Poll detects completion from status.json."""
        run_id = "ssh:10.0.0.1:/tmp/daedalus/exp_001"

        with patch.object(runner, "_run_ssh") as mock_ssh:
            mock_ssh.return_value = MagicMock(
                returncode=0,
                stdout='{"state":"completed","pid":1234}\n---SEPARATOR---\n',
            )

            status = runner.poll(run_id)
            assert status.state == "completed"

    def test_poll_failed_from_events(self, runner):
        """Poll detects failure from sentinel FAILED event."""
        run_id = "ssh:10.0.0.1:/tmp/daedalus/exp_001"

        with patch.object(runner, "_run_ssh") as mock_ssh:
            mock_ssh.return_value = MagicMock(
                returncode=0,
                stdout='{"state":"running","pid":1234}\n---SEPARATOR---\n'
                       '{"event":"FAILED","pid":1234,"error":"OOM"}\n',
            )

            status = runner.poll(run_id)
            assert status.state == "failed"

    def test_poll_ssh_failure(self, runner):
        """Poll handles SSH connection failure."""
        run_id = "ssh:10.0.0.1:/tmp/daedalus/exp_001"

        with patch.object(runner, "_run_ssh") as mock_ssh:
            mock_ssh.return_value = MagicMock(
                returncode=1,
                stderr="Connection timed out",
            )

            status = runner.poll(run_id)
            assert status.state == "failed"
            assert "SSH" in status.error

    def test_poll_invalid_run_id(self, runner):
        """Poll handles invalid run_id."""
        status = runner.poll("ssh_12345")
        assert status.state == "failed"


class TestFetchEvents:
    def test_fetch_events(self, runner):
        """fetch_events returns parsed JSONL events."""
        run_id = "ssh:10.0.0.1:/tmp/daedalus/exp_001"

        with patch.object(runner, "_run_ssh") as mock_ssh:
            mock_ssh.return_value = MagicMock(
                returncode=0,
                stdout=(
                    '{"event":"SENTINEL_START","pid":1234}\n'
                    '{"event":"METRIC","metric":"loss","value":0.5}\n'
                    '{"event":"PROGRESS","key":"epoch","value":"1/10"}\n'
                ),
            )

            events = runner.fetch_events(run_id)
            assert len(events) == 3
            assert events[1]["event"] == "METRIC"
            assert events[1]["value"] == 0.5

    def test_fetch_events_since_line(self, runner):
        """fetch_events with since_line skips old events."""
        run_id = "ssh:10.0.0.1:/tmp/daedalus/exp_001"

        with patch.object(runner, "_run_ssh") as mock_ssh:
            mock_ssh.return_value = MagicMock(
                returncode=0,
                stdout='{"event":"METRIC","metric":"loss","value":0.3}\n',
            )

            events = runner.fetch_events(run_id, since_line=5)
            assert len(events) == 1

            # Verify tail command was used
            call_args = mock_ssh.call_args[0][0]
            assert "tail -n +6" in call_args

    def test_fetch_events_no_file(self, runner):
        """fetch_events returns empty list when no events file."""
        run_id = "ssh:10.0.0.1:/tmp/daedalus/exp_001"

        with patch.object(runner, "_run_ssh") as mock_ssh:
            mock_ssh.return_value = MagicMock(returncode=1, stdout="")

            events = runner.fetch_events(run_id)
            assert events == []

    def test_fetch_events_invalid_run_id(self, runner):
        """fetch_events handles invalid run_id."""
        events = runner.fetch_events("ssh_12345")
        assert events == []


class TestLogs:
    def test_logs(self, runner):
        run_id = "ssh:10.0.0.1:/tmp/daedalus/exp_001"

        with patch.object(runner, "_run_ssh") as mock_ssh:
            mock_ssh.return_value = MagicMock(
                returncode=0,
                stdout="Epoch 1/10\nloss=0.5\n",
            )

            output = runner.logs(run_id, tail=50)
            assert "Epoch 1/10" in output

    def test_logs_invalid_run_id(self, runner):
        output = runner.logs("ssh_12345")
        assert "Cannot decode" in output


class TestCancel:
    def test_cancel(self, runner):
        run_id = "ssh:10.0.0.1:/tmp/daedalus/exp_001"

        with patch.object(runner, "_run_ssh") as mock_ssh:
            mock_ssh.return_value = MagicMock(returncode=0)
            runner.cancel(run_id)
            mock_ssh.assert_called_once()

    def test_cancel_invalid_run_id(self, runner):
        """Cancel handles invalid run_id gracefully."""
        runner.cancel("ssh_12345")  # Should not raise
