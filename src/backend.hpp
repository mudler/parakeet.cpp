#pragma once
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;
struct ggml_backend;
typedef struct ggml_backend* ggml_backend_t;

namespace pk {

class ModelLoader;

// Persistent CPU compute backend + reusable graph allocator.
//
// parakeet's inference path builds hundreds of tiny ggml graphs per utterance
// (272 on the 110m for a 10 s clip). The old `run_graph` did a full
// `ggml_init` -> build -> `ggml_graph_compute_with_ctx` -> `ggml_free` on every
// call, so ~40% of wall time was allocator churn (init/alloc + free of the
// per-call compute buffer), and the disposable threadpool was respawned each
// time. This mirrors the wall rt-detr.cpp hit; the fix is the same: a
// persistent `ggml_backend_t` (CPU) + a persistent `ggml_gallocr_t` reused
// across every graph computation.
//
// Backend::compute() builds the graph in a `no_alloc=true` context (metadata
// only), allocates it via the persistent gallocr (which packs intermediates
// with lifetime-aware reuse AND keeps the underlying compute buffer alive
// across calls, so the steady-state loop sees no mmap/munmap traffic), pushes
// the host input data AFTER alloc, runs it on the persistent backend, and reads
// the output back.
//
// CORRECTNESS-CRITICAL ordering: with `no_alloc=true`, a tensor's `->data` is
// NULL until `ggml_gallocr_alloc_graph`. So input tensor data MUST be written
// AFTER alloc (via `ggml_backend_tensor_set`), never inline in the build
// lambda. The build lambda registers each host-backed input by calling
// `pk::add_graph_input(...)`; Backend defers the copy until after alloc.
class Backend {
public:
    // Construct a CPU backend with `n_threads` worker threads (<=0 -> 1). The
    // gallocr is created lazily on the first compute and reused afterwards.
    explicit Backend(int n_threads);
    ~Backend();

    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;

    // Update the worker-thread count (cheap; just calls
    // ggml_backend_cpu_set_n_threads). <=0 is clamped to 1.
    void set_n_threads(int n_threads);
    int  n_threads() const { return n_threads_; }

    // Name of the selected compute device ("cpu" for the CPU backend, or the
    // registry device name for a GPU backend, e.g. the CUDA device name).
    const char* device_name() const { return device_name_.c_str(); }

    // True iff the active backend is a non-CPU (GPU/IGPU) device. The graph-
    // capture/replay decode optimisation only helps GPU (it is launch-overhead
    // bound there); on CPU the per-step set_input + capture-readback overhead
    // it adds is a net regression, so callers gate on this.
    bool is_gpu() const;

    // The underlying CPU ggml backend. Exposed so the loader can give its weight
    // tensors a backend buffer over the SAME backend graphs run on (see
    // ModelLoader::realize_weights). Any CPU buffer is compatible with the CPU
    // backend, but agreeing on one keeps the contract explicit.
    ggml_backend_t handle() const;

    // Build + run a single graph on the persistent backend/gallocr and copy the
    // output tensor's f32 contents into `out`.
    //
    //   build(ctx): create the graph in `ctx` (a no_alloc=true context).
    //     - Input tensors: `ggml_new_tensor*` then register their host data via
    //       `pk::add_graph_input(t, host_ptr, nbytes)`. Do NOT write `t->data`
    //       (it is NULL until alloc). add_graph_input marks the tensor as a
    //       graph input so the gallocr keeps it.
    //     - Weights: reference loader tensors directly (their `->data` is set by
    //       the mmap'd loader context, so the gallocr treats them as already
    //       allocated and never touches them) OR build small host-computed
    //       constants as inputs via add_graph_input.
    //     - Returns the output tensor (Backend marks it as an output).
    //
    // Returns true on success.
    bool compute(const std::function<ggml_tensor*(ggml_context*)>& build,
                 std::vector<float>& out);

    // Backend-internal hook used by add_graph_input(). Registers a host->device
    // input copy to be performed after ggml_gallocr_alloc_graph. Public so the
    // free helper can reach it, but callers should use add_graph_input().
    void register_input(ggml_tensor* t, const void* host, size_t nbytes);

    // Backend-internal hook used by capture_graph_output(). Marks `t` as an
    // output kept by the gallocr and reads its f32 contents into `*dst` after
    // compute. Lets one graph yield several tensors (e.g. the fused encoder's
    // per-layer captures) without separate compute calls.
    void register_capture(ggml_tensor* t, std::vector<float>* dst);

private:
    struct Impl;
    Impl* impl_;
    int   n_threads_ = 1;
    std::string device_name_ = "cpu";

    friend class ReplayGraph;
};

// Register a host-backed graph input for the currently-active Backend::compute
// build phase. Marks `t` as a ggml graph input and records that `nbytes` from
// `host` must be copied into `t` AFTER the graph is allocated. Must be called
// from inside a build lambda passed to Backend::compute (it routes to the
// Backend driving the current compute via a thread-local pointer).
//
// `host` must stay valid until Backend::compute returns (it does for all
// callers: the data is the caller's input vector, alive across the call).
void add_graph_input(ggml_tensor* t, const void* host, size_t nbytes);

// Create a graph input tensor of the given type and ne[] inside `ctx`, mark it
// as a graph input, and register a host->device copy of `nbytes` from `host`
// to be performed after the gallocr allocates it. The data is NOT written
// inline (the tensor's ->data is NULL until alloc); use this instead of
// `ggml_new_tensor*` + memcpy in build lambdas. Returns the new tensor.
ggml_tensor* graph_input_tensor(ggml_context* ctx, int type, int n_dims,
                                const int64_t* ne, const void* host,
                                size_t nbytes);

// Capture an intermediate graph tensor for readback after Backend::compute.
// `*dst` is resized to the tensor's element count and filled with its f32
// contents once the graph has run. Must be called from inside a build lambda;
// `dst` must stay valid until Backend::compute returns. Used by the fused
// encoder to pull per-layer outputs out of the single graph.
void capture_graph_output(ggml_tensor* t, std::vector<float>* dst);

// Reference a weight tensor from the loader DIRECTLY as a graph leaf (ZERO
// per-call copy). The loader gives every weight a CPU backend buffer ONCE at
// load (ModelLoader::realize_weights, lazily ensured here on first use against
// the process-global Backend). With the weight living in a backend buffer and
// having ->data set, the gallocr treats it as already-allocated and never
// touches it, and reshapes/views resolve their data at build time — so the
// SAME bytes are reused across every utterance instead of being re-copied.
// Allowlisted linears may be f16/q8_0; ggml_mul_mat dequantizes src0 on the
// fly. Returns nullptr iff `name` is absent (clone_weight_opt); asserts present
// otherwise (clone_weight). `ctx` is unused (kept for call-site compatibility).
ggml_tensor* clone_weight(ggml_context* ctx, const ModelLoader& ml,
                          const char* name);
ggml_tensor* clone_weight_opt(ggml_context* ctx, const ModelLoader& ml,
                              const char* name);

// Ensure the loader's weights have a backend buffer (zero-copy) on the
// process-global Backend. Idempotent; called automatically by clone_weight, but
// exposed so the model/test load path can realize weights up front.
void ensure_weights_realized(const ModelLoader& ml);

// Copy a weight tensor's f32 contents into `out` on the host, regardless of
// whether the weight lives in CPU or device (e.g. CUDA) memory. Ensures weights
// are realized first. Use this for host-side computation that needs raw floats
// (preprocessing, batch-norm folding) — NOT for graph leaves (use clone_weight).
void weight_to_host_f32(const ModelLoader& ml, const char* name, std::vector<float>& out);

// A graph built ONCE and recomputed many times, keeping the SAME ggml context /
// cgraph / input tensors alive across calls. The C++ analogue of megapar's
// CUDA-graph step capture, realized through ggml's OWN capture:
//
// ggml-cuda keys its internal CUDA graph on `cgraph->nodes[0]` (a tensor pointer
// owned by the compute context). Backend::compute does ggml_init + ggml_free
// EVERY call, so every per-step graph gets a NEW context -> NEW node pointers ->
// a different key -> CUDA-graph capture NEVER warms up and every tiny per-step
// op is launched directly (the launch-overhead regime that dominates the GPU
// transducer decode loop). ReplayGraph keeps the context + cgraph alive across
// calls, so nodes[0] is a STABLE pointer: ggml-cuda warms up after one extra
// direct eval and then replays the captured graph, collapsing the per-step
// launch storm. On CPU this still wins by skipping the per-call ggml_init /
// gallocr re-plan / ggml_free.
//
// The replayed graph's input tensors live in the (persistent) ggml context; the
// caller feeds fresh data into them each step via the returned input handles.
//
// Usage:
//   ReplayGraph rg(backend, [&](ggml_context* ctx){
//       ggml_tensor* a = pk::graph_input_tensor(ctx, ...);
//       return some_op(ctx, a, weight);
//   });
//   // build() recorded `a`'s handle; feed it:
//   rg.set_input(0, host_a, nbytes_a);
//   std::vector<float> out; rg.compute(out);
class ReplayGraph {
public:
    // Build the graph now: runs `build(ctx)` in a no_alloc context (same
    // contract as Backend::compute's build lambda — register host inputs via
    // pk::add_graph_input / pk::graph_input_tensor), allocates the result via
    // the backend's persistent gallocr, and remembers the input-tensor handles
    // (in registration order) for set_input(). `backend` must outlive this
    // object (it owns the gallocr the graph is allocated in).
    ReplayGraph(Backend& backend,
                const std::function<ggml_tensor*(ggml_context*)>& build);
    ~ReplayGraph();

    ReplayGraph(const ReplayGraph&) = delete;
    ReplayGraph& operator=(const ReplayGraph&) = delete;

    // Feed `nbytes` from `host` into input #`i` (the i-th tensor registered
    // during build). The data lands in the persistent input tensor; it must
    // stay valid until compute() returns.
    void set_input(size_t i, const void* host, size_t nbytes);

    // Recompute the graph (with whatever data set_input wrote) and read the
    // output tensor's f32 contents into `out`. Returns true on success.
    bool compute(std::vector<float>& out);

    // Number of input tensors registered during build().
    size_t n_inputs() const { return inputs_.size(); }

    // Recompute the graph and read BOTH the output tensor (into `out`) and every
    // tensor registered via pk::capture_graph_output() during build() (into the
    // caller's stable dst vectors). The captures are remembered across calls
    // (unlike Backend::compute, which clears them), so the same dst vectors are
    // re-filled each step. Used by the prediction net to pull each layer's new
    // (h', c') state out of the replayed graph.
    bool compute_with_captures(std::vector<float>& out);

private:
    Backend& backend_;
    ggml_context* ctx_ = nullptr;
    ggml_cgraph*  gf_  = nullptr;
    ggml_tensor*  out_ = nullptr;
    // Input tensors recorded in build() order; set_input(i) writes into these.
    std::vector<ggml_tensor*> inputs_;
    // Capture tensors + their stable dst vectors (recorded from
    // pk::capture_graph_output() during build). compute_with_captures() re-fills
    // each dst after every replay.
    std::vector<std::pair<ggml_tensor*, std::vector<float>*>> captures_;
    // Whether gf_ was allocated via the sched fallback (true) or the fast
    // gallocr path (false). Set once in alloc_internal(); read in compute().
    bool need_sched_ = false;

    // Allocate gf_ on the persistent gallocr / sched (called once in the ctor).
    bool alloc_internal();
};

} // namespace pk
