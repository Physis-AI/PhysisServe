"""Tensor movement: pass references in a single process, point-to-point torch.distributed across processes.

``transfer`` takes ordered items [(tag, src, dst, spec, tensors)]:
- every rank preserves the global item order; unrelated items may be filtered out in advance;
- a rank acts only where src/dst is itself, with matching message order at both ends;
- spec is [(shape, dtype), ...], from which the receiver pre-allocates buffers;
- tensors are required only when src is this process;
- the result {tag: Pending} contains only the items whose dst is this process; ``Pending.wait()``
  returns the tensors once they have arrived.
Transfers are posted asynchronously and waited for by their consumer, so a rank keeps computing while
its states are in flight. Sent tensors stay referenced until their transfer completes; ``flush`` waits
for everything still outstanding (the end of a request).
"""
from math import prod

import torch
import torch.distributed as dist


class _Work:
    """Share one backend wait between a received item and transport cleanup.

    Backend work handles need not allow repeated waits (in particular Gloo receives). Completion
    polling is separate: a consumer still waits once to establish its device stream dependency.
    """

    __slots__ = ("work", "waited")

    def __init__(self, work):
        self.work, self.waited = work, False

    def wait(self):
        if not self.waited:
            self.work.wait()
            self.waited = True

    def is_completed(self):
        # A device-stream wait can return before device completion; keep polling the real handle
        # so sender storage is retained until the backend reports that the operation completed.
        return self.work.is_completed()


class Pending:
    """Tensors of one received item; ``wait`` blocks until they hold the data."""

    __slots__ = ("tensors", "works")

    def __init__(self, tensors, works=()):
        self.tensors, self.works = tensors, tuple(works)

    def wait(self):
        for work in self.works:
            work.wait()
        self.works = ()
        return self.tensors


class Local:
    """Single process: every logical rank lives here, so a transfer is a reference."""

    def is_local(self, rank: int) -> bool:
        return True

    def transfer(self, items):
        return {tag: Pending(tensors) for tag, _, _, _, tensors in items}

    def flush(self):
        pass


class Distributed:
    """Multi-process: each logical rank is one process of the process group.

    Sends and receives are submitted together in the supplied item order through
    ``batch_isend_irecv``. All ranks must preserve the matching global order.
    """

    def __init__(self, device, group=None):
        self.device, self.group = torch.device(device), group
        self.rank, self.world = dist.get_rank(group), dist.get_world_size(group)
        self.inflight = []  # (works, tensors) retained until the backend reports completion

    def is_local(self, rank: int) -> bool:
        return rank == self.rank

    def _peer(self, rank):
        return dist.get_global_rank(self.group, rank) if self.group is not None else rank

    def _receive_buffers(self, spec):
        # One allocation per same-dtype packet, with non-overlapping contiguous
        # views. Never pool these buffers: callers may retain received tensors
        # or use them as KV pages after the transfer/request has completed.
        if spec and all(dtype == spec[0][1] for _, dtype in spec):
            sizes = [prod(shape) for shape, _ in spec]
            storage = torch.empty(sum(sizes), dtype=spec[0][1], device=self.device)
            buffers, offset = [], 0
            for (shape, _), size in zip(spec, sizes):
                buffers.append(storage.narrow(0, offset, size).view(shape))
                offset += size
            return tuple(buffers)
        return tuple(torch.empty(shape, dtype=dtype, device=self.device) for shape, dtype in spec)

    def _submit(self, operations):
        if not operations:
            return []
        works = [_Work(work) for work in dist.batch_isend_irecv(
            [dist.P2POp(op, tensor, peer, self.group) for op, tensor, peer in operations])]
        # Retain receiver storage too, even if its Pending is discarded before
        # completion. A stream wait alone does not establish device completion.
        self.inflight.append((works, tuple(tensor for _, tensor, _ in operations)))
        return works

    def transfer(self, items):
        self.inflight = [entry for entry in self.inflight if not all(work.is_completed() for work in entry[0])]
        received, result, operations = {}, {}, []
        for tag, src, dst, spec, tensors in items:
            if src == dst:
                if src == self.rank:
                    result[tag] = Pending(tensors)
                continue
            if dst == self.rank:
                buffers = self._receive_buffers(spec)
                first = len(operations)
                received[tag] = buffers, slice(first, first + len(buffers))
                peer = self._peer(src)
                operations.extend((dist.irecv, x, peer) for x in buffers)
            elif src == self.rank:
                kept = tuple(x.contiguous() for x in tensors)
                peer = self._peer(dst)
                operations.extend((dist.isend, x, peer) for x in kept)
        works = self._submit(operations)
        # Gloo returns one handle per operation. Coalesced NCCL may instead return aggregate
        # handles for the whole batch; every consumer must share those completion handles.
        per_operation = len(works) == len(operations)
        result.update({tag: Pending(buffers, works[selection] if per_operation else works)
                       for tag, (buffers, selection) in received.items()})
        return result

    def flush(self):
        for works, _ in self.inflight:
            for work in works:
                work.wait()
        self.inflight = []
