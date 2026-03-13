"""Tests for runner factory."""

import pytest
import yaml

from daedalus.runners.factory import create_runner
from daedalus.runners.local import LocalRunner
from daedalus.runners.ssh import SSHRunner


class TestRunnerFactory:
    def test_default_local(self, tmp_path):
        """No config -> local runner."""
        runner = create_runner(tmp_path)
        assert isinstance(runner, LocalRunner)

    def test_explicit_local(self, tmp_path):
        (tmp_path / "daedalus.yaml").write_text(
            yaml.dump({"runner": {"type": "local"}})
        )
        (tmp_path / "runner_config.yaml").write_text(
            yaml.dump({"local": {"python": "/usr/bin/python3"}})
        )

        runner = create_runner(tmp_path)
        assert isinstance(runner, LocalRunner)
        assert runner.python == "/usr/bin/python3"

    def test_ssh_runner(self, tmp_path):
        (tmp_path / "daedalus.yaml").write_text(
            yaml.dump({"runner": {"type": "ssh", "config_key": "gpu_host"}})
        )
        (tmp_path / "runner_config.yaml").write_text(
            yaml.dump({"gpu_host": {
                "host": "10.0.0.1",
                "user": "ubuntu",
                "remote_work_dir": "/science/daedalus",
            }})
        )

        runner = create_runner(tmp_path)
        assert isinstance(runner, SSHRunner)
        assert runner.config.host == "10.0.0.1"
        assert runner.config.user == "ubuntu"

    def test_unknown_type_raises(self, tmp_path):
        (tmp_path / "daedalus.yaml").write_text(
            yaml.dump({"runner": {"type": "kubernetes"}})
        )

        with pytest.raises(ValueError, match="Unknown runner type"):
            create_runner(tmp_path)


class TestMultiHostSSH:
    def _setup_multi_host(self, tmp_path, default_host=None):
        """Helper to set up multi-host config."""
        runner_spec = {"type": "ssh"}
        if default_host:
            runner_spec["default_host"] = default_host

        (tmp_path / "daedalus.yaml").write_text(
            yaml.dump({"runner": runner_spec})
        )
        (tmp_path / "runner_config.yaml").write_text(
            yaml.dump({"hosts": {
                "gpu-a100": {
                    "host": "10.0.0.1",
                    "user": "ubuntu",
                    "remote_work_dir": "/data/daedalus",
                },
                "gpu-h100": {
                    "host": "10.0.0.2",
                    "user": "admin",
                    "remote_work_dir": "/opt/daedalus",
                },
            }})
        )

    def test_multi_host_default(self, tmp_path):
        self._setup_multi_host(tmp_path, default_host="gpu-h100")

        runner = create_runner(tmp_path)
        assert isinstance(runner, SSHRunner)
        assert runner.config.host == "10.0.0.2"
        assert runner.config.user == "admin"

    def test_multi_host_explicit(self, tmp_path):
        self._setup_multi_host(tmp_path, default_host="gpu-a100")

        runner = create_runner(tmp_path, host="gpu-h100")
        assert isinstance(runner, SSHRunner)
        assert runner.config.host == "10.0.0.2"

    def test_multi_host_unknown_raises(self, tmp_path):
        self._setup_multi_host(tmp_path)

        with pytest.raises(ValueError, match="Unknown host 'nonexistent'"):
            create_runner(tmp_path, host="nonexistent")

    def test_multi_host_no_default_uses_first(self, tmp_path):
        self._setup_multi_host(tmp_path)  # no default_host

        runner = create_runner(tmp_path)
        assert isinstance(runner, SSHRunner)
        assert runner.config.host == "10.0.0.1"  # first entry

    def test_multi_host_invalid_default_raises(self, tmp_path):
        self._setup_multi_host(tmp_path, default_host="nonexistent")

        with pytest.raises(ValueError, match="Default host 'nonexistent' not found"):
            create_runner(tmp_path)

    def test_legacy_ssh_still_works(self, tmp_path):
        """Old config_key format should still work."""
        (tmp_path / "daedalus.yaml").write_text(
            yaml.dump({"runner": {"type": "ssh", "config_key": "my_gpu"}})
        )
        (tmp_path / "runner_config.yaml").write_text(
            yaml.dump({"my_gpu": {
                "host": "10.0.0.99",
                "user": "legacy",
            }})
        )

        runner = create_runner(tmp_path)
        assert isinstance(runner, SSHRunner)
        assert runner.config.host == "10.0.0.99"
