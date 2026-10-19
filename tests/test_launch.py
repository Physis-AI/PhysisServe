# Required Notice: Copyright 2026 Physis AI, Inc. and/or its affiliates
#
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# Licensed under the PolyForm Noncommercial License 1.0.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://polyformproject.org/licenses/noncommercial/1.0.0
#
# This software is provided "AS IS", without warranties or conditions.
# See the License for permissions, limitations, and liability terms.

"""Exercise the script launcher with independent torchrun agents on two nodes."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import textwrap
from time import monotonic

import pytest
import torch
import torch.distributed as dist

from physisserve import PhysisServe, SamplingParams
from tests.test_api import fake_config


def engine_state(engine):
    """An importable worker operation, just as bench.measure is importable."""
    return {
        "rank": engine.rank,
        "distributed_rank": dist.get_rank() if dist.is_initialized() else 0,
        "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
        "ranks": engine.config.ranks,
        "token": engine.config.launch_test_token,
        "loads": len(engine.config.loaded),
        "encoded": engine.config.encoded,
        "decoder_loads": engine.config.decoder_loads,
    }


def cpu_measure(engine, prompts, params, warmups, repeats, candidates=None, budgets=()):
    """Use wall time for the CPU fixture while exercising bench's real engines."""
    engine.prepare(prompts, params, candidates, budgets)
    for _ in range(warmups):
        engine.run(return_outputs=False)
    measurements = []
    for _ in range(repeats):
        engine._barrier()
        start, times, batches = monotonic(), {}, []

        def completed(index, chunk, latent):
            times.setdefault(index, []).append((chunk, (monotonic() - start) * 1000))

        assert engine.run(on_output=completed, on_batch=lambda index, rank, size: batches.append(size),
                          return_outputs=False) is None
        measurements.append({"times": times, "max_batch": max(batches, default=0)})
    return measurements


def cpu_devices(engine):
    return {"rank": engine.rank, "name": "CPU fixture", "free_bytes": 1 << 30, "total_bytes": 1 << 30}


def cpu_calibrate(engine, params, prompt, max_batch, warmups=3, samples=12):
    return {"rank": engine.rank, "error": None,
            "samples_ms": {f"{block}:{size}": [float(size)] * samples
                           for block in range(engine.rank, engine.config.num_blocks, engine.config.ranks)
                           for size in range(1, max_batch + 1)}}


def cpu_prepare_candidate(engine, prompts, params, candidates=None, budgets=(), memory_margin=1.2, memory_budgets=None):
    return {"rank": engine.rank, "status": "ready"}


def fail_worker_in_collective(engine):
    if engine.rank == 1:
        raise ValueError("test worker failure")
    Path("rank-zero-entered-barrier").touch()
    dist.barrier()


def _terminate_agents(agents):
    # torchrun puts workers in separate sessions, so killing only the agent's
    # process group can leak workers when a regression hangs a collective.
    table = subprocess.check_output(["ps", "-axo", "pid=,ppid=,pgid="], text=True)
    processes = [tuple(map(int, line.split())) for line in table.splitlines()]
    descendants = {agent.pid for agent in agents if agent.poll() is None}
    while True:
        children = {pid for pid, parent, _ in processes if parent in descendants}
        if children <= descendants:
            break
        descendants.update(children)
    groups = {group for pid, _, group in processes if pid in descendants}
    groups.discard(os.getpgrp())
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for group in groups:
            try:
                os.killpg(group, sig)
            except ProcessLookupError:
                pass
        for agent in agents:
            try:
                agent.wait(timeout=3 if sig == signal.SIGTERM else 5)
            except subprocess.TimeoutExpired:
                pass


def _run_script(tmp_path, source, *, processes_per_node=1, timeout=75):
    script = tmp_path / "user_script.py"
    script.write_text(textwrap.dedent(source))
    repository = str(Path(__file__).resolve().parents[1])
    env = {key: value for key, value in os.environ.items()
           if key not in {"RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK",
                          "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"}
           and not key.startswith("TORCHELASTIC_")}
    env.update(PYTHONPATH=os.pathsep.join(filter(None, [repository, env.get("PYTHONPATH")])),
               PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    agents, logs = [], []
    deadline = monotonic() + timeout
    with ExitStack() as stack:
        try:
            for node in range(2):
                command = [sys.executable, "-m", "torch.distributed.run", "--nnodes=2",
                           f"--nproc-per-node={processes_per_node}", f"--node-rank={node}",
                           "--master-addr=127.0.0.1", f"--master-port={port}",
                           "--max-restarts=0", "--monitor-interval=0.1",
                           "--module", "physisserve.launch", "--device", "cpu",
                           str(script), "forwarded value", "--script-option"]
                log = tmp_path / f"node-{node}.log"
                logs.append(log)
                stream = stack.enter_context(log.open("w"))
                agents.append(subprocess.Popen(command, cwd=tmp_path, env=env,
                                               stdout=stream, stderr=subprocess.STDOUT,
                                               start_new_session=True))
            for agent in agents:
                agent.wait(timeout=max(0.01, deadline - monotonic()))
        except subprocess.TimeoutExpired:
            _terminate_agents(agents)
            pytest.fail("launcher did not terminate within the test deadline:\n" +
                        "\n".join(log.read_text() for log in logs))
        finally:
            if any(agent.poll() is None for agent in agents):
                _terminate_agents(agents)
    return [agent.returncode for agent in agents], "\n".join(log.read_text() for log in logs)


_PREAMBLE = """
import json
import os
from pathlib import Path
import sys

import torch
import physisserve.serve as serve
from physisserve import PhysisServe, SamplingParams
from tests.test_api import fake_config, fail_worker
from tests.test_launch import engine_state, fail_worker_in_collective

assert __name__ == "__main__"
assert sys.argv[1:] == ["forwarded value", "--script-option"]
with Path("script-ran").open("x") as marker:
    marker.write(str(os.getpid()))
config_calls = []
def make_config(model, **kwargs):
    # Give every worker an ordinary local model path; fake weights stay in the fixture.
    Path(model).mkdir(parents=True, exist_ok=True)
    config_calls.append(kwargs["ranks"])
    config = fake_config(model, **kwargs)
    config.launch_test_token = "created in user script only"
    return config
serve.Config = make_config
"""


@pytest.mark.parametrize("processes_per_node", [1, 2], ids=["two-nodes", "four-ranks"])
def test_launcher_runs_unmodified_api_once_and_reuses_pool(tmp_path, monkeypatch, processes_per_node):
    source = _PREAMBLE + """
params = [SamplingParams(num_chunks=5, seed=seed, mode="dit_only") for seed in (17, 23)]
offline = [SamplingParams(num_chunks=5, seed=seed, mode="offline_decode",
                          latent_dir="latents") for seed in (17, 23)]
results, states = {}, {}
for strategy in ("stage_equal", "virtual_rank"):
    with PhysisServe("fake", pipeline_parallel_size=1, device="cpu", strategy=strategy) as service:
        first = service.generate(["cat", "dog"], params)
        again = service.generate(["cat", "dog"], params)
        for original, repeated in zip(first, again):
            torch.testing.assert_close(original["latents"], repeated["latents"], rtol=0, atol=0)
        outputs = service.generate(["cat", "dog"], offline)
        for original, output in zip(first, outputs):
            torch.testing.assert_close(original["latents"], output["latents"], rtol=0, atol=0)
        states[strategy] = service._call(engine_state)
        results[strategy] = outputs
    assert service._closed
    service.close()
torch.save(results, "outputs.pt")
Path("states.json").write_text(json.dumps({"states": states, "config_calls": config_calls}))
"""
    codes, logs = _run_script(tmp_path, source, processes_per_node=processes_per_node)
    assert all(code == 0 for code in codes), logs
    assert (tmp_path / "script-ran").is_file()
    world = processes_per_node * 2
    report = json.loads((tmp_path / "states.json").read_text())
    assert report["config_calls"] == [world, world]
    for states in report["states"].values():
        assert [state["rank"] for state in states] == list(range(world))
        assert [state["distributed_rank"] for state in states] == list(range(world))
        assert [state["local_rank"] for state in states] == [
            rank % processes_per_node for rank in range(world)]
        assert all(state["ranks"] == world and state["loads"] > 0 for state in states)
        assert all(state["token"] == "created in user script only" for state in states)
        assert states[0]["encoded"] == ["cat", "dog"] * 3
        assert states[0]["decoder_loads"] == 1
        assert all(state["encoded"] == [] and state["decoder_loads"] == 0 for state in states[1:])

    from safetensors.torch import load_file

    actual = torch.load(tmp_path / "outputs.pt", weights_only=True)
    assert len(list((tmp_path / "latents").glob("generate-*/*.safetensors"))) == 4
    monkeypatch.setattr("physisserve.serve.Config", fake_config)
    for strategy, outputs in actual.items():
        params = [SamplingParams(num_chunks=5, seed=seed, mode="dit_only") for seed in (17, 23)]
        with PhysisServe("fake", device="cpu", strategy=strategy) as local:
            expected = local.decode(local.generate(["cat", "dog"], params))
        for output, reference in zip(outputs, expected):
            assert output["prompt"] == reference["prompt"]
            torch.testing.assert_close(output["latents"], reference["latents"], rtol=0, atol=0)
            torch.testing.assert_close(output["video"], reference["video"], rtol=0, atol=0)
            torch.testing.assert_close(load_file(output["latent_path"])["latents"],
                                       reference["latents"], rtol=0, atol=0)


@pytest.mark.parametrize("operation", ["fail_worker", "fail_worker_in_collective"],
                         ids=["result-gather", "data-collective"])
def test_worker_exception_terminates_both_nodes_without_hanging(tmp_path, operation):
    source = _PREAMBLE + f"""
with PhysisServe("fake", device="cpu") as service:
    service._call({operation})
raise AssertionError("worker failure was silently ignored")
"""
    codes, logs = _run_script(tmp_path, source)
    assert all(code != 0 for code in codes), logs
    assert "test worker failure" in logs
    assert "worker failure was silently ignored" not in logs
    if operation == "fail_worker_in_collective":
        assert (tmp_path / "rank-zero-entered-barrier").exists()


def test_original_example_runs_without_edits_or_rank_guards(tmp_path):
    example = str(Path(__file__).resolve().parents[1] / "example.py")
    source = _PREAMBLE + f"""
import runpy
runpy.run_path({example!r}, run_name="__main__")
"""
    codes, logs = _run_script(tmp_path, source)
    assert all(code == 0 for code in codes), logs

    import imageio.v3 as iio
    from safetensors.torch import load_file

    assert sorted(path.name for path in tmp_path.glob("*.mp4")) == ["video-0.mp4", "video-1.mp4"]
    for index in range(2):
        assert iio.imread(tmp_path / f"video-{index}.mp4").shape == (13, 2, 2, 3)
    latents = sorted((tmp_path / "latents").glob("generate-*/*.safetensors"))
    assert len(latents) == 2
    assert all(load_file(str(path))["latents"].shape == (1, 2, 7, 2, 2) for path in latents)


def test_benchmark_reuses_pool_and_gathers_profile_and_delivery_measurements(tmp_path):
    source = _PREAMBLE + """
import bench
from tests.test_launch import cpu_measure, cpu_devices, cpu_calibrate, cpu_prepare_candidate

# Only GPU measurement/metadata operations are replaced; dispatch, inference,
# policy switching, result aggregation and JSON recording remain real.
bench.torch.cuda.is_available = lambda: True
bench.measure = cpu_measure
bench.devices = cpu_devices
bench.calibrate = cpu_calibrate
bench.prepare_candidate = cpu_prepare_candidate
bench.identity = lambda config, params, hardware: {"fixture": "cpu", "ranks": config.ranks}
sys.argv = ["bench.py", "fake", "--streams", "2",
            "--warmups", "1", "--repeats", "2", "--output", "benchmark"]
bench.main()
assert config_calls == [2]
"""
    codes, logs = _run_script(tmp_path, source)
    assert all(code == 0 for code in codes), logs
    report = json.loads((tmp_path / "benchmark/report.json").read_text())
    assert report["status"] in ("completed", "incomplete") and "error" not in report
    assert report["identity"] == {"fixture": "cpu", "ranks": 2}
    assert [row["rank"] for row in report["hardware"]] == [0, 1]
    calibration = json.loads((tmp_path / "benchmark/calibration.json").read_text())
    assert [row["rank"] for row in calibration] == [0, 1]
    assert [set(row["samples_ms"]) for row in calibration] == [
        {f"{block}:{size}" for size in range(1, 5)} for block in range(2)]
    assert json.loads((tmp_path / "benchmark/costs.json").read_text()) == [[1, 2, 3, 4]] * 2
    assert set(report["comparison"]) == {"stage_equal", "virtual_rank"}
    for method in report["comparison"]:
        point = json.loads((tmp_path / "benchmark" / f"{method}.json").read_text())
        assert [row["rank"] for row in point["preflight"]] == [0, 1]
        assert point["attempts"]
        for attempt in point["attempts"]:
            output_rank, worker_rank = attempt["raw"]
            assert len(output_rank) == len(worker_rank) == 2
            assert all(not trial["times"] for trial in worker_rank)
            for trial in output_rank:
                assert set(trial["times"]) == {"0", "1"}
                for index, count in enumerate(attempt["chunks"]):
                    assert [chunk for chunk, _ in trial["times"][str(index)]] == list(range(count))
    assert "Reports:" in logs
    assert not (tmp_path / "latents").exists()


@pytest.mark.parametrize("create_service", [False, True], ids=["before-service", "open-service"])
def test_script_exception_stops_waiting_workers(tmp_path, create_service):
    source = _PREAMBLE
    if create_service:
        source += '\nservice = PhysisServe("fake", device="cpu")\n'
    source += '\nraise RuntimeError("user script failure")\n'
    codes, logs = _run_script(tmp_path, source)
    assert any(code != 0 for code in codes), logs
    assert "user script failure" in logs


def test_conflicting_pipeline_size_fails_before_config_and_pool_remains_usable(tmp_path):
    source = _PREAMBLE + """
def forbidden_config(*args, **kwargs):
    raise AssertionError("invalid size reached Config")
serve.Config = forbidden_config
try:
    PhysisServe("fake", pipeline_parallel_size=3, device="cpu")
except ValueError:
    pass
else:
    raise AssertionError("conflicting pipeline size was accepted")
serve.Config = make_config
with PhysisServe("fake", device="cpu") as service:
    assert [state["rank"] for state in service._call(engine_state)] == [0, 1]
    try:
        PhysisServe("fake", device="cpu")
    except RuntimeError as error:
        assert "close" in str(error)
    else:
        raise AssertionError("a second active service was accepted")
    assert not service._closed
    assert [state["rank"] for state in service._call(engine_state)] == [0, 1]
Path("conflict-recovered").touch()
"""
    codes, logs = _run_script(tmp_path, source)
    assert all(code == 0 for code in codes), logs
    assert (tmp_path / "conflict-recovered").exists()
