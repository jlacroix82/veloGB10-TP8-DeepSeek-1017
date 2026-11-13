# Build the experimental TP8 code

The benchmark builds used AArch64 Linux, Rust 1.98.1, CUDA 13.0.88,
and the NVIDIA 580.178.04 driver. Build on a GB10 host with the CUDA toolkit,
a C compiler and the libibverbs and liblzma development libraries. For
a toolkit under `/usr/local/cuda`:

```sh
CUDA_HOME=/usr/local/cuda \
LIBRARY_PATH=/usr/local/cuda/lib64 \
LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
cargo build --locked --release --example dsv41_resident_atomic_http --example dsv41_hc_graph_oracle

LD_LIBRARY_PATH=/usr/local/cuda/lib64 target/release/examples/dsv41_hc_graph_oracle
```

The model-free oracle uses one GPU and compares eager and graph output
at widths 1, 2, 8, and 16, including a sticky status check. It opens no
RoCE links and does not need the checkpoint. Use an idle GB10 for this
check. The resident executable uses `--decode-hc-scratch off|reuse` and
`--decode-hc-graph off|on`. Graphs require reuse. Both default to off.

The resident diagnostic also needs the pinned checkpoint, its tokenizer
and compressed token map, verified CUDA rotary frequency tables, and
a controller that coordinates all-rank startup and request release.
Supply eight management IPv4 addresses via `--management-ips` and
eight RoCE addresses via `--peer-ips`, both in rank order. Bind HTTP
to the configured rank-zero management address on a trusted private
network. The port is a bounded text diagnostic; it is not an
authenticated public service or a turnkey cluster installer.

Tokenizer-backed CPU tests look for the checkpoint's `tokenizer.json`
at `test-data/tokenizer.json` (ignored by Git). The expected SHA-256 is
`c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b`.
The CPU replay example can instead use the `DSV41_TOKENIZER` environment
variable. The 573-request archive replay remains opt-in and requires the
operator's original request manifest via `DSV41_PRIVATE_REQUEST_AUDIT`.

The source derives from [veloGB10 at 438176b](https://github.com/sf-stav/veloGB10/commit/438176b68436295a0a9bbda11355e10dbf92be01)
under its [Apache 2.0 license](../LICENSE). See the upstream
[credits](../CREDITS.md), [DeepSeek numerical code license](../src/dsv41/DEEPSEEK_LICENSE)
and [vendored toolkit license](../vendor/deepseek-recipe/LICENSE).
