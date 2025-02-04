import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config(
            kwargs=dict(
                BLOCK_SIZE_ROWS=BLOCK_SIZE_ROWS,
                num_stages=num_stages,
            ),
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for BLOCK_SIZE_ROWS in (16, 32, 64, 128)
        for num_stages in (2, 3, 4)
        for num_warps in (2, 4, 8)
    ],
    key=["N_COLS"],
)
@triton.heuristics(
    values=dict(BLOCK_SIZE_COLS=lambda args: triton.next_power_of_2(args["N_COLS"]))
)
@triton.jit
def softmax_kernel(
    input_ptr: tl.tensor,
    output_ptr: tl.tensor,
    input_row_stride: int,
    output_row_stride: int,
    n_rows: int,
    N_COLS: tl.constexpr,
    BLOCK_SIZE_ROWS: tl.constexpr,
    BLOCK_SIZE_COLS: tl.constexpr,
    num_stages: tl.constexpr,
):
    input_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(n_rows, N_COLS),
        strides=(input_row_stride, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_ROWS, BLOCK_SIZE_COLS),
        order=(1, 0),
    )

    output_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(n_rows, N_COLS),
        strides=(output_row_stride, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_ROWS, BLOCK_SIZE_COLS),
        order=(1, 0),
    )

    cols_mask = tl.arange(0, BLOCK_SIZE_COLS) < N_COLS

    row_idx = tl.program_id(0) * BLOCK_SIZE_ROWS
    in_tile_ptr = tl.advance(input_ptr, (row_idx, 0))
    row = tl.load(pointer=in_tile_ptr, boundary_check=(0, 1))

    # Subtract maximum for numerical stability
    row_minus_max = row - tl.max(row, axis=1, keep_dims=True)
    row_minus_max = tl.where(cols_mask, row_minus_max, -float("inf"))

    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=1, keep_dims=True)
    softmax_output = numerator / denominator

    out_tile_ptr = tl.advance(output_ptr, (row_idx, 0))
    tl.store(out_tile_ptr, softmax_output, boundary_check=(0, 1))


def softmax(x: torch.Tensor):
    x_orig_shape = x.shape
    x = x.view(-1, x_orig_shape[-1])
    n_rows, n_cols = x.shape

    y = torch.empty_like(x, memory_format=torch.contiguous_format)

    grid = lambda args: (triton.cdiv(n_rows, args["BLOCK_SIZE_ROWS"]), 1, 1)

    softmax_kernel[grid](
        input_ptr=x,
        output_ptr=y,
        input_row_stride=x.stride(0),
        output_row_stride=y.stride(0),
        n_rows=n_rows,
        N_COLS=n_cols,
    )
    return y.view(*x_orig_shape)


def softmax_naive(x: torch.Tensor):
    row_minus_max = x - x.max(-1, keepdim=True).values
    numerator = torch.exp(row_minus_max)
    denominator = torch.sum(numerator, dim=-1, keepdim=True)
    return numerator / denominator


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],  # argument names to use as an x-axis for the plot
        x_vals=[
            128 * i for i in range(2, 20)
        ],  # different possible values for `x_name`
        line_arg="provider",  # argument name whose value corresponds to a different line in the plot
        line_vals=["triton", "torch", "torch-naive"],  # possible values for `line_arg``
        line_names=[
            "Triton",
            "Torch",
            "Torch-naive",
        ],
        styles=[("red", "-"), ("green", "-"), ("blue", "-")],  # line styles
        ylabel="GB/s",
        plot_name="softmax-performance",
        args={"M": 128},
    )
)
def benchmark(N, M, provider):
    device = torch.device("cuda")
    x = torch.randn(N, M, device=device, dtype=torch.float32)
    stream = getattr(torch, device.type).Stream()
    getattr(torch, device.type).set_stream(stream)

    if provider == "torch":
        ms = triton.testing.do_bench(lambda: F.softmax(x, dim=-1))
    elif provider == "torch-naive":
        ms = triton.testing.do_bench(lambda: softmax_naive(x))
    elif provider == "triton":
        ms = triton.testing.do_bench(lambda: softmax(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


if __name__ == "__main__":
    x = torch.ones(13, 10, 19, device="cuda")

    tri_out = softmax(x)
    torch_out = F.softmax(x, dim=-1)

    torch.testing.assert_close(tri_out, torch_out)

    benchmark.run(print_data=True, save_path="./")
