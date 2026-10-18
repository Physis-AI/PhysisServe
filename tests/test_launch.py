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


def cpu_measure(engine, requests, warmups, repeats):
    """Use wall time for the CPU fixture while exercising bench's real engines."""
    engine.prepare(requests)
    for _ in range(warmups):
        engine.run()
    measurements = []
    for _ in range(repeats):
        engine._barrier()
        start, times, batches = monotonic(), {}, []

        def completed(index, chunk, latent):
            times.setdefault(index, []).append((chunk, (monotonic() - start) * 1000))

        engine.run(on_latent=completed, on_batch=lambda index, rank, size: batches.append(size))
        measurements.append({"times": times, "max_batch": max(batches, default=0)})
    return measurements


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


def test_original_benchmark_reopens_services_and_gathers_worker_measurements(tmp_path):
    source = _PREAMBLE + """
import bench
import runpy
from tests.test_launch import cpu_measure

# The benchmark remains CUDA-only in production; replace its timer for this CPU fixture.
bench.torch.cuda.is_available = lambda: True
bench.measure = cpu_measure
sys.argv = ["bench.py", "fake", "--streams", "2", "--chunks", "5", "--skip-chunks", "0",
            "--warmups", "1", "--repeats", "2"]
runpy.run_path(bench.__file__, run_name="__main__")
assert config_calls == [2, 2]
"""
    codes, logs = _run_script(tmp_path, source)
    assert all(code == 0 for code in codes), logs
    assert "stage_equal   completion" in logs
    assert "virtual_rank  completion" in logs
    assert "VirtualRank speedup:" in logs
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
