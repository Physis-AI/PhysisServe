"""Grouped P2P work mapping, single waits, and in-flight storage ownership."""
from datetime import timedelta
import multiprocessing
import os
from pathlib import Path
import socket
import time
from types import SimpleNamespace
import weakref

import pytest
import torch

from physisserve.engine import transport


@pytest.fixture
def fake_distributed(monkeypatch):
    waits, works, posted = [], [], []

    class Work:
        def __init__(self, index):
            self.index, self.complete, self.waited = index, False, False
            self.complete_on_wait = True

        def wait(self):
            # Gloo receive handles need not support a second wait even after completion.
            assert not self.waited, "the same raw work handle was waited twice"
            self.waited, self.complete = True, self.complete_on_wait
            waits.append(self.index)

        def is_completed(self):
            return self.complete

    def batch(operations):
        posted.extend(("send" if op is transport.dist.isend else "recv", peer, group)
                      for op, _, peer, group in operations)
        created = [Work(len(works) + index) for index in range(len(operations))]
        works.extend(created)
        return created

    def send(*args, **kwargs):
        pytest.fail("sends must be submitted as grouped P2P operations")

    def receive(*args, **kwargs):
        pytest.fail("receives must be submitted as grouped P2P operations")

    monkeypatch.setattr(transport.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(transport.dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(transport.dist, "get_world_size", lambda group=None: 3)
    monkeypatch.setattr(transport.dist, "P2POp", lambda op, tensor, peer, group: (op, tensor, peer, group))
    monkeypatch.setattr(transport.dist, "batch_isend_irecv", batch)
    monkeypatch.setattr(transport.dist, "isend", send)
    monkeypatch.setattr(transport.dist, "irecv", receive)
    return SimpleNamespace(bus=transport.Distributed("cpu"), waits=waits, works=works, posted=posted)


SPEC = [((1,), torch.float32)]


def test_received_item_waits_only_its_operations_and_flush_does_not_wait_twice(fake_distributed):
    state = fake_distributed
    pending = state.bus.transfer([
        ("first", 1, 0, SPEC * 2, None),
        ("sent", 0, 2, SPEC, (torch.ones(1),)),
        ("second", 2, 0, SPEC, None),
    ])
    pending["first"].wait()
    assert state.waits == [0, 1]
    pending["second"].wait()
    assert state.waits == [0, 1, 3]
    state.bus.flush()
    assert state.waits == [0, 1, 3, 2]
    assert state.bus.inflight == []


def test_flush_before_consumer_does_not_repeat_raw_wait(fake_distributed):
    state = fake_distributed
    pending = state.bus.transfer([("received", 1, 0, SPEC, None)])["received"]
    state.bus.flush()
    pending.wait()
    pending.wait()
    assert state.waits == [0]


def test_completed_work_still_gets_one_consumer_wait_after_inflight_pruning(fake_distributed):
    state = fake_distributed
    pending = state.bus.transfer([("received", 1, 0, SPEC, None)])["received"]
    # Completion polling permits releasing sender storage, but does not replace the consumer's
    # wait: backends such as NCCL use that wait to establish the current stream dependency.
    state.works[0].complete = True
    state.bus.transfer([])
    assert state.bus.inflight == []
    pending.wait()
    pending.wait()
    assert state.waits == [0]


def test_stream_wait_does_not_claim_device_work_has_completed(fake_distributed):
    state = fake_distributed
    pending = state.bus.transfer([("received", 1, 0, SPEC, None)])["received"]
    state.works[0].complete_on_wait = False
    pending.wait()
    state.bus.transfer([])
    assert len(state.bus.inflight) == 1
    state.works[0].complete = True
    state.bus.transfer([])
    assert state.bus.inflight == []
    assert state.waits == [0]


def test_aggregate_work_is_shared_by_every_receive_and_flush(fake_distributed, monkeypatch):
    state = fake_distributed
    make_work = transport.dist.batch_isend_irecv
    monkeypatch.setattr(transport.dist, "batch_isend_irecv", lambda operations: make_work(operations[:1]))
    pending = state.bus.transfer([
        ("first", 1, 0, SPEC * 2, None),
        ("sent", 0, 2, SPEC, (torch.ones(1),)),
        ("second", 2, 0, SPEC, None),
    ])
    # The second receive has no per-operation handle in this backend's result, but it must still
    # wait on the aggregate before exposing its tensors. Later consumers and cleanup share it.
    pending["second"].wait()
    assert state.waits == [0]
    pending["first"].wait()
    state.bus.flush()
    assert state.waits == [0]
    assert state.bus.inflight == []


def test_empty_and_local_transfers_need_no_backend_operations(fake_distributed):
    state = fake_distributed
    assert state.bus.transfer([]) == {}
    tensors = (torch.ones(1),)
    pending = state.bus.transfer([("local", 0, 0, SPEC, tensors)])
    assert pending["local"].wait() is tensors
    assert state.posted == state.works == state.bus.inflight == []


def test_backend_failure_propagates(fake_distributed, monkeypatch):
    def fail(operations):
        raise RuntimeError("backend failed")

    monkeypatch.setattr(transport.dist, "batch_isend_irecv", fail)
    with pytest.raises(RuntimeError, match="backend failed"):
        fake_distributed.bus.transfer([("received", 1, 0, SPEC, None)])


def test_grouped_submission_preserves_order_and_maps_receive_handles(fake_distributed):
    state = fake_distributed
    pending = state.bus.transfer([
        ("sent", 0, 2, SPEC, (torch.ones(1),)),
        ("received", 1, 0, SPEC * 2, None),
    ])["received"]
    pending.wait()
    assert state.posted == [("send", 2, None), ("recv", 1, None), ("recv", 1, None)]
    assert state.waits == [1, 2]
    assert not state.works[0].complete
    state.bus.transfer([])
    assert len(state.bus.inflight) == 1
    state.bus.flush()
    assert state.waits == [1, 2, 0]


def test_sender_storage_survives_until_backend_completion(fake_distributed):
    state = fake_distributed
    tensor = torch.ones(1)
    reference = weakref.ref(tensor)
    state.bus.transfer([("sent", 0, 2, SPEC, (tensor,))])
    del tensor
    assert reference() is not None
    state.works[0].complete = True
    state.bus.transfer([])
    assert reference() is None


def test_dropped_pending_keeps_receiver_storage_until_completion(fake_distributed):
    state = fake_distributed
    pending = state.bus.transfer([("received", 1, 0, SPEC, None)])["received"]
    reference = weakref.ref(pending.tensors[0])
    del pending
    assert reference() is not None
    state.works[0].complete = True
    state.bus.transfer([])
    assert reference() is None


def test_receive_packet_views_do_not_overlap_and_are_not_reused(fake_distributed):
    bus = fake_distributed.bus
    first = bus._receive_buffers([((2, 3), torch.float32), ((4,), torch.float32)])
    assert first[0].untyped_storage().data_ptr() == first[1].untyped_storage().data_ptr()
    first[0].fill_(2)
    first[1].fill_(3)
    second = bus._receive_buffers([((2, 3), torch.float32), ((4,), torch.float32)])
    second[0].fill_(4)
    second[1].fill_(5)
    assert torch.equal(first[0], torch.full((2, 3), 2.0))
    assert torch.equal(first[1], torch.full((4,), 3.0))
    mixed = bus._receive_buffers([((2,), torch.float32), ((3,), torch.int64)])
    assert tuple(x.dtype for x in mixed) == (torch.float32, torch.int64)


def _gloo_exchange_worker(rank, directory):
    torch.set_num_threads(1)
    interfaces = {name for _, name in socket.if_nameindex()}
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if "lo0" in interfaces else "lo"
    transport.dist.init_process_group("gloo", rank=rank, world_size=3,
        init_method=(Path(directory) / "rendezvous").as_uri(), timeout=timedelta(seconds=20))
    try:
        bus = transport.Distributed("cpu")
        retained = []
        spec = [((2,), torch.float32), ((3, 4), torch.float32)]
        for iteration in range(8):
            direction = 1 if iteration % 2 else -1
            items = []
            for source in range(3):
                value = iteration * 10 + source
                tensors = (torch.full((2,), float(value)), torch.full((4, 3), float(value + 1)).t())
                items.append((source, source, (source + direction) % 3, spec,
                              tensors if rank == source else None))
            if iteration % 2:
                # Runners may prefilter unrelated moves but must preserve the
                # relative global order of the remaining operations.
                items = [item for item in items if rank in item[1:3]]
            pending = bus.transfer(items)
            if iteration % 2:
                bus.flush()  # A later consumer still may wait its Pending once.
            source = (rank - direction) % 3
            received = pending[source].wait()
            retained.append((iteration, source, received))
        bus.flush()
        assert bus.inflight == []
        # Retaining inputs beyond later transfers is legal (e.g. a backend KV
        # page aliases an input). Completion must not imply buffer reuse.
        for iteration, source, tensors in retained:
            value = iteration * 10 + source
            assert torch.equal(tensors[0], torch.full((2,), float(value)))
            assert torch.equal(tensors[1], torch.full((3, 4), float(value + 1)))
    finally:
        transport.dist.destroy_process_group()


@pytest.mark.skipif(not transport.dist.is_gloo_available(), reason="Gloo is unavailable")
def test_three_rank_gloo_packet_exchange_and_retained_inputs(tmp_path):
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_gloo_exchange_worker, args=(rank, str(tmp_path)))
                 for rank in range(3)]
    deadline = time.monotonic() + 60
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))
        assert not any(process.is_alive() for process in processes), "Gloo exchange exceeded 60 seconds"
        assert [process.exitcode for process in processes] == [0, 0, 0]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
            if process.pid is not None:
                process.close()
