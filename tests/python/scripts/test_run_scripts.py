"""Validate launch configuration without loading a model or touching devices."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVE_SCRIPT = REPO_ROOT / "scripts/vllm/run/serve-8gpu-striped.sh"
PROFILE_ENV = REPO_ROOT / "scripts/vllm/run/profile-env.sh"


@pytest.fixture
def launch_env(tmp_path):
    config = tmp_path / 'daemon with "quotes".yaml'
    config.write_text("nvmes:\n  - device_id: 0\n  - device_id: 1\n")
    return {
        **os.environ,
        "MODEL": "example/model",
        "TUTTI_PYTHON": sys.executable,
        "TUTTI_PROFILE_ROOT": str(tmp_path / "scratch"),
        "DAEMON_CONFIG": str(config),
        "DEVICE_GROUPS": "0;1",
        "KV_MOUNT": str(tmp_path / 'kv with "quotes"'),
        "TP_SIZE": "2",
        "CHUNK_TOKENS": "512",
        "BLOCK_SIZE": "512",
        "TOTAL_CAPACITY": "2 KiB",
        "POOL_TAG": "launch-test",
        "CUDA_VISIBLE_DEVICES": "3,5",
        "LOAD_FORMAT": "auto",
        "PHXLOADER_DIR": "",
        "NUM_QUEUES": "8",
    }


def preview(env):
    return subprocess.run(
        ["bash", str(SERVE_SCRIPT), "--dry-run"],
        env=env, text=True, capture_output=True, timeout=30,
    )


def test_serve_preview_has_real_connector_config(launch_env):
    result = preview(launch_env)
    assert result.returncode == 0, result.stderr
    config_line = next(line for line in result.stdout.splitlines()
                       if line.startswith("[serve] kv_config="))
    config = json.loads(config_line.split("=", 1)[1])
    assert config["kv_connector_module_path"] == "tutti.integration.vllm.connector"
    assert config["kv_load_failure_policy"] == "fail"
    options = config["kv_connector_extra_config"]["store"]["options"]
    assert options["capacity"] == "1024 B"
    assert "capacity_bytes" not in options and "num_chunks" not in options
    assert options["root"] == str(Path(launch_env["KV_MOUNT"]) /
                                   "tutti-kv-online-launch-test-{LOCAL_RANK}")
    assert options["preset"]["daemon_config"] == [launch_env["DAEMON_CONFIG"]]
    assert options["preset"]["device_groups"] == [[0], [1]]
    assert "--load-format auto" in result.stdout


@pytest.mark.parametrize("overrides, message", [
    ({"TP_SIZE": "0"}, "positive integers"),
    ({"TOTAL_CAPACITY": "1 B"}, "at least one byte per rank"),
    ({"TOTAL_CAPACITY": "0 GiB"}, "正整数字节"),
    ({"TOTAL_CAPACITY_BYTES": "2048"}, "已删除"),
    ({"CHUNK_TOKENS": "513"}, "divisible by BLOCK_SIZE"),
    ({"DEVICE_GROUPS": "0;1;0"}, "group count must divide TP_SIZE"),
    ({"DEVICE_GROUPS": "0;9"}, "devices missing"),
    ({"KV_MOUNT": "relative"}, "must be absolute"),
    ({"POOL_TAG": "../unsafe"}, "directory name"),
])
def test_serve_preview_rejects_invalid_config(launch_env, overrides, message):
    result = preview({**launch_env, **overrides})
    assert result.returncode != 0
    assert message in result.stderr


def test_serve_preview_applies_queue_override(launch_env):
    result = preview({**launch_env, "NUM_QUEUES": "4"})
    assert result.returncode == 0, result.stderr
    line = next(line for line in result.stdout.splitlines()
                if line.startswith("[serve] kv_config="))
    config = json.loads(line.split("=", 1)[1])
    assert config["kv_connector_extra_config"]["store"]["options"]["preset"]["num_queues"] == 4


@pytest.mark.parametrize("value", ["0", "-1", "many"])
def test_serve_preview_rejects_invalid_queue_count(launch_env, value):
    result = preview({**launch_env, "NUM_QUEUES": value})
    assert result.returncode != 0
    assert "positive integers" in result.stderr


def test_only_module_config_templates_exist():
    assert (REPO_ROOT / "config/device_manager/nvme_daemon.yaml").is_file()
    assert (REPO_ROOT / "config/storage_runtime/local_nvme.yaml").is_file()
    assert not (REPO_ROOT / "config/local_nvme_config.yaml").exists()
    assert not (REPO_ROOT / "config/tutti_config.yaml").exists()


def test_serve_rejects_unmounted_metadata_root(launch_env):
    result = subprocess.run(
        ["bash", str(SERVE_SCRIPT)], env=launch_env,
        text=True, capture_output=True, timeout=30,
    )
    assert result.returncode != 0
    assert "KV_MOUNT is not mounted" in result.stderr


def test_profile_env_preserves_gpu_and_toolchain(launch_env):
    env = {**launch_env, "CC": "chosen-cc", "CXX": "chosen-cxx"}
    command = '''source "$1"; "$TUTTI_PYTHON" -c 'import json,os
print(json.dumps({k:os.environ.get(k) for k in
("PATH","CC","CXX","CUDA_VISIBLE_DEVICES","TUTTI_PYTHON","TMPDIR")}))' '''
    result = subprocess.run(
        ["bash", "-e", "-c", command, "bash", str(PROFILE_ENV)],
        env=env, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    for key in ("PATH", "CC", "CXX", "CUDA_VISIBLE_DEVICES", "TUTTI_PYTHON"):
        assert output[key] == env[key]
    assert output["TMPDIR"] == str(Path(env["TUTTI_PROFILE_ROOT"]) / "tmp")


@pytest.fixture
def bench_runner(launch_env, tmp_path):
    capture = tmp_path / "driver-arguments.txt"
    interpreter = tmp_path / "capture-python"
    interpreter.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$TUTTI_TEST_ARGV\"\n")
    interpreter.chmod(0o755)
    tools = tmp_path / "tools"
    tools.mkdir()
    mountpoint = tools / "mountpoint"
    mountpoint.write_text("#!/bin/sh\nexit 0\n")
    mountpoint.chmod(0o755)
    env = {**launch_env, "TUTTI_PYTHON": str(interpreter),
           "TUTTI_TEST_ARGV": str(capture), "PATH": f"{tools}:{launch_env['PATH']}"}
    env.pop("CAPACITY", None)
    env.pop("MAX_IN_FLIGHT", None)

    def run(args, overrides=None):
        result = subprocess.run(
            ["bash", str(REPO_ROOT / "scripts/vllm/run/bench-8gpu-striped.sh"), *args],
            env={**env, **(overrides or {})}, text=True, capture_output=True, timeout=30,
        )
        arguments = capture.read_text().splitlines() if capture.exists() else []
        return result, arguments

    return run


@pytest.mark.parametrize("args, overrides", [
    (["--capacity", "64 GiB"], {}),
    ([], {"CAPACITY": "64 GiB"}),
])
def test_bench_passes_per_rank_capacity_without_loading_model(bench_runner, args, overrides):
    result, arguments = bench_runner(args, overrides)
    assert result.returncode == 0, result.stderr
    assert arguments[arguments.index("--capacity") + 1] == "64 GiB"
    assert "--direct-transfer-strict" not in arguments
    assert "--without-tutti" not in arguments


def test_bench_requires_capacity_for_tutti(bench_runner):
    result, arguments = bench_runner([])
    assert result.returncode != 0
    assert "CAPACITY" in result.stderr
    assert arguments == []


def test_bench_baseline_needs_no_storage_capacity(bench_runner):
    result, arguments = bench_runner(["--baseline"])
    assert result.returncode == 0, result.stderr
    assert "--without-tutti" in arguments
    assert "--capacity" not in arguments


@pytest.fixture
def memwatch():
    spec = importlib.util.spec_from_file_location(
        "tutti_memwatch_test", REPO_ROOT / "scripts/vllm/run/memwatch.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("text, expected", [
    ("", {}),
    ("unrelated resident=99", {}),
    ("[tutti] health resident=3 pending=1 / 20(used)",
     {"resident": 3, "pending": 1, "capacity": 20}),
    ("[tutti] health resident=3\nother line\n[tutti] health resident=7 hit_reqs=4",
     {"resident": 7, "hit_reqs": 4}),
    ("[tutti] health resident=3\n[tutti] health no counters", {}),
])
def test_memwatch_health_uses_last_line(memwatch, text, expected):
    assert memwatch.read_health(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("", {"ckpt_total": 0.0, "ckpt_failed_total": 0.0}),
    ("DIRECT_CHECKPOINT_WRITTEN", {"ckpt_total": 1.0, "ckpt_failed_total": 0.0}),
    ("DIRECT_CHECKPOINT_WRITTEN elapsed=1.25s\nDIRECT_CHECKPOINT_FAILED\n"
     "DIRECT_CHECKPOINT_WRITTEN elapsed=0.50s",
     {"ckpt_total": 2.0, "ckpt_failed_total": 1.0, "ckpt_last_elapsed_s": 0.5}),
])
def test_memwatch_storage_stats_output_unchanged(memwatch, text, expected):
    assert memwatch.read_storage_stats(text) == expected
