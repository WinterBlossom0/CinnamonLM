"""Step times and fused-kernel configurations on the current GPU: `python speed.py <arm> [<arm> ...]`.

For each arm: the wall time of the compiled training step at the full batch and at half of it (each process's share
when a run is split over 2 GPUs). Then the GPU time of the forward and the backward Hyperloop, GRT (with gate noise,
as in training) and MeSH kernels at the full batch of 81,920 tokens, for BLOCK_M 16/32/64/128 and 1/2/4 warps, and
the fastest configuration of each kernel. The MeSH times are the sum over the 9 steps of an x8 run.
"""
from __future__ import annotations

import sys
import time
from collections.abc import Callable

import torch
import triton.testing

import grt_kernel
import hyperloop_kernel
import mesh_kernel
import train

ROWS = train.BATCH_SIZE * train.CONTEXT_LENGTH
# BLOCK_M is a power of 2 and at least 16, because it is a tl.dot dimension in every kernel.
CONFIGS = [(block, warps) for block in (16, 32, 64, 128) for warps in (1, 2, 4)]
KERNELS = {"hyperloop": hyperloop_kernel, "grt": grt_kernel, "mesh": mesh_kernel}


def wall_ms(function, repeats: int) -> float:
    for _ in range(3):
        function()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        function()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats * 1e3


def gpu_ms(function) -> float:
    # A kernel call takes microseconds, so Python dispatch would dominate a wall-clock time; a CUDA graph replay
    # leaves only the GPU work.
    return triton.testing.do_bench_cudagraph(function, return_mode="median")


def step_times(arm: str, blocks: torch.Tensor) -> None:
    torch.manual_seed(0)
    torch._dynamo.reset()
    model = train.build_model(train.ARMS[arm], 512).cuda()
    compiled = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
    for rows in (train.BATCH_SIZE, train.BATCH_SIZE // 2):
        state = {"index": 0}

        def step():
            index = state["index"] = (state["index"] + 1) % 40
            batch = blocks[index * rows:(index + 1) * rows].long()
            compiled(input_ids=batch, labels=batch).loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        print(f"{torch.cuda.get_device_name()} step {arm} batch {rows}: {wall_ms(step, 30):.1f} ms", flush=True)


def kernel_calls() -> dict[str, tuple[str, Callable[[], object]]]:
    """Each timed call by name, with the kernel module whose BLOCK_M and WARPS it reads."""
    width, streams, slots = train.HIDDEN_SIZE, train.HYPERLOOP_STREAMS, 11
    flat = torch.randn(ROWS, streams * width, device="cuda")
    weight = torch.randn(3 * streams, streams * width, device="cuda") * 0.1
    gate, bias, norm = (torch.ones(3 * streams, device="cuda"), torch.zeros(3 * streams, device="cuda"),
                        torch.ones(streams * width, device="cuda"))
    grads = (torch.randn(ROWS, width, device="cuda"), torch.randn(ROWS, streams, device="cuda"),
             torch.randn(ROWS, streams, device="cuda"))
    state, output, normed = (torch.randn(ROWS, width, device="cuda") for _ in range(3))
    gains = (torch.ones(width, device="cuda"), torch.ones(2 * width, device="cuda"))
    w1, w2, b2 = (torch.randn(width, 2 * width, device="cuda") * 0.1, torch.randn(width, width, device="cuda") * 0.1,
                  torch.full((width,), train.GRT_GATE_BIAS_INIT, device="cuda"))
    gate_noise = torch.randn(ROWS, device="cuda") * train.GRT_NOISE_STD
    eps = train.LAYER_NORM_EPS
    kept = [torch.randn(ROWS, width, device="cuda") for _ in range(9)]
    writes = [torch.softmax(torch.randn(ROWS, slots, device="cuda"), -1) for _ in range(9)]
    read = torch.softmax(torch.randn(ROWS, slots, device="cuda"), -1)
    router_w, router_b = torch.randn(2 * slots, width, device="cuda") * 0.1, torch.zeros(2 * slots, device="cuda")
    mesh_steps = []
    for count in range(1, 10):  # the 9 steps of an x8 run
        has_next = count < 9
        outputs = mesh_kernel.mesh_step(state, kept[:count], writes[:count], read, router_w, router_b, slots, has_next)
        grad_slots = torch.randn(ROWS if has_next else 0, slots, device="cuda")
        mesh_steps.append((count, has_next, outputs, grad_slots))

    def mesh_forward():
        for count, has_next, _, _ in mesh_steps:
            mesh_kernel.mesh_step(state, kept[:count], writes[:count], read, router_w, router_b, slots, has_next)

    def mesh_backward():
        for count, has_next, outputs, grad_slots in mesh_steps:
            mesh_kernel.mesh_step_backward(state, kept[:count], writes[:count], read, router_w, *outputs, state,
                                           grad_slots, grad_slots, slots, has_next)

    return {
        "hyperloop forward": ("hyperloop", lambda: hyperloop_kernel.hyperloop_read(flat, weight, gate, bias, norm,
                                                                                   streams, train.RMS_NORM_EPS)),
        "hyperloop backward": ("hyperloop", lambda: hyperloop_kernel.hyperloop_read_backward(
            flat, weight, gate, bias, norm, *grads, streams, train.RMS_NORM_EPS)),
        "grt forward": ("grt", lambda: grt_kernel.grt_update(state, output, normed, *gains, w1, w2, b2, eps,
                                                             gate_noise)),
        "grt backward": ("grt", lambda: grt_kernel.grt_update_backward(state, output, normed, *gains, w1, w2, b2,
                                                                       state, eps, gate_noise)),
        "mesh forward": ("mesh", mesh_forward),
        "mesh backward": ("mesh", mesh_backward),
    }


def kernel_configs() -> None:
    calls = kernel_calls()
    times = {name: {} for name in calls}
    device = torch.cuda.get_device_name()
    for block, warps in CONFIGS:
        for module in KERNELS.values():
            module.BLOCK_M, module.WARPS = block, warps
        for name, (_, function) in calls.items():
            times[name][block, warps] = gpu_ms(function)
        print(f"{device} BLOCK_M {block:3d} warps {warps}: "
              + "  ".join(f"{name} {times[name][block, warps]:.4f}" for name in calls) + " ms", flush=True)
    for name, by_config in times.items():
        (block, warps), best = min(by_config.items(), key=lambda item: item[1])
        print(f"{device} fastest {name}: BLOCK_M {block} warps {warps} ({best:.4f} ms)", flush=True)
    # Forward and backward of one module share its BLOCK_M and WARPS, so their sum decides the setting.
    for kernel in KERNELS:
        names = [name for name, (owner, _) in calls.items() if owner == kernel]
        (block, warps), best = min(((config, sum(times[name][config] for name in names)) for config in CONFIGS),
                                   key=lambda item: item[1])
        print(f"{device} fastest {kernel} forward + backward: BLOCK_M {block} warps {warps} ({best:.4f} ms)",
              flush=True)


def main() -> None:
    blocks = train.to_blocks(train.load_packed("validation")).cuda()
    for arm in sys.argv[1:]:
        step_times(arm, blocks)
    kernel_configs()


if __name__ == "__main__":
    main()
