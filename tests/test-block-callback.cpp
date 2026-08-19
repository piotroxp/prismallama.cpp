// Per-block eval callback test (JAISIU-2319-A, prismalama fork)
//
// Verifies that llama_block_callback_install() + llama_block_callback_dispatch()
// fire the user-supplied llama_block_cb exactly once per transformer block
// during graph compute.
//
// The test does NOT load a real model — it builds a synthetic ggml_cgraph
// with mock tensors named like llama.cpp's real graph ("ffn_inp-0",
// "attn_out-3", ...) so the dispatch wrapper's boundary detection sees the
// same naming convention it sees in production.

#ifdef NDEBUG
#undef NDEBUG
#endif

#include "llama.h"
#include "llama-block-callback.h"

#include <ggml-backend.h>
#include <ggml-cpu.h>
#include <ggml-impl.h>
#include <ggml.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <memory>
#include <set>
#include <vector>

namespace {

// Recorded events from the test callback.
struct cb_record {
    int  block_idx;
    bool aborted; // true if callback returned 0
};

// Capture state shared with the test callback.
struct cb_state {
    std::vector<cb_record>      records;
    std::atomic<int>            call_count{0};
    int                         fail_at = -1; // abort (return 0) when block_idx == fail_at
};

// C-linkage callback exposed via the public llama_block_cb typedef.
int test_block_cb(void * user_data, int block_idx) {
    auto * st = static_cast<cb_state *>(user_data);
    st->call_count.fetch_add(1, std::memory_order_relaxed);
    const bool abort = (st->fail_at >= 0 && block_idx == st->fail_at);
    st->records.push_back({block_idx, abort});
    return abort ? 0 : 1;
}

// Build a synthetic ggml_cgraph that looks like a tiny llama forward pass:
//   for each block i in [0, n_blocks):
//       3 ops named "<base>-i"   e.g. "ffn_inp-i", "attn_out-i", "ffn_out-i"
//   plus 1 final "logits-N" tensor (no suffix; should not be a boundary)
//
// Returns the graph. Caller owns the ggml_context.
ggml_cgraph * build_fake_graph(ggml_context * ctx, int n_blocks, int ops_per_block = 3) {
    ggml_cgraph * gf = ggml_new_graph(ctx);

    ggml_tensor * prev = 0;
    for (int b = 0; b < n_blocks; b++) {
        for (int op = 0; op < ops_per_block; op++) {
            ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
            // Name format: "<base>-<il>" with a base that varies by op.
            // ggml_set_name copies the string, so we must use a buffer.
            char name[64];
            const char * base = (op == 0) ? "ffn_inp" :
                               (op == 1) ? "attn_out" : "ffn_out";
            std::snprintf(name, sizeof(name), "%s-%d", base, b);
            ggml_set_name(t, name);

            if (prev != nullptr) {
                // Tie them together so ggml sees a real graph (not a forest).
                ggml_add(ctx, prev, t);
            }
            prev = t;
        }
    }

    // Final logits tensor without suffix — should NOT be a boundary.
    ggml_tensor * logits = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(logits, "logits");
    if (prev != nullptr) {
        ggml_add(ctx, prev, logits);
        prev = logits;
    }

    ggml_build_forward_expand(gf, prev);
    return gf;
}

// ---------------------------------------------------------------------------
// Test 1 — pure dispatch (no scheduler).
//
// Manually drive the dispatch wrapper: for every node in the graph, ask
// "should I observe?" with ask=true, then if it returns true call with
// ask=false. Verify the user callback fires once per block boundary, with
// the expected indices, in graph order.
// ---------------------------------------------------------------------------
void test_dispatch_pure() {
    printf("test-block-callback: dispatch_pure\n");

    constexpr int N_BLOCKS = 4;
    constexpr int OPS_PER_BLOCK = 3;

    struct ggml_init_params iparams = {
        /*.mem_size   =*/ 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(iparams);
    assert(ctx != nullptr);

    ggml_cgraph * gf = build_fake_graph(ctx, N_BLOCKS, OPS_PER_BLOCK);
    assert(ggml_graph_n_nodes(gf) == N_BLOCKS * OPS_PER_BLOCK + 1);

    cb_state st;
    std::shared_ptr<llama_block_callback_state> state;

    const bool has_boundaries = llama_block_callback_install(
            /*sched=*/nullptr, // scheduler not needed for this test
            gf,
            &test_block_cb,
            &st,
            /*node_cb=*/nullptr,
            /*node_cb_data=*/nullptr,
            state);

    // install() returns false when sched == nullptr (early-out), so we
    // need to scan manually and dispatch manually here.
    assert(state != nullptr);
    assert(!state->block_node_to_idx.empty());

    // Manually scan + dispatch: simulate what the GGML scheduler does.
    auto * gf_mut = gf; // already non-const
    const int n_nodes = ggml_graph_n_nodes(gf_mut);
    int boundary_count = 0;
    for (int i = 0; i < n_nodes; i++) {
        ggml_tensor * node = ggml_graph_node(gf_mut, i);
        const bool should_observe = llama_block_callback_dispatch(node, /*ask=*/true, state.get());
        if (!should_observe) {
            continue;
        }
        const bool cont = llama_block_callback_dispatch(node, /*ask=*/false, state.get());
        // Only boundary nodes trigger the block_cb. Per-node cb is NULL here.
        (void) cont;
        if (state->block_node_to_idx.find(node) != state->block_node_to_idx.end()) {
            boundary_count++;
        }
    }

    // Assertions:
    assert(st.call_count.load() == N_BLOCKS);
    assert(st.records.size() == static_cast<size_t>(N_BLOCKS));
    for (int i = 0; i < N_BLOCKS; i++) {
        assert(st.records[i].block_idx == i);
        assert(st.records[i].aborted == false);
    }
    assert(boundary_count == N_BLOCKS);
    (void) has_boundaries;

    ggml_free(ctx);
    printf("  ok (%d blocks, %d fires)\n", N_BLOCKS, st.call_count.load());
}

// ---------------------------------------------------------------------------
// Test 2 — abort path.
//
// Same as test 1, but fail_at == 2. The dispatch wrapper must return
// false (abort) at block 2 and the user callback must see block 2 with
// aborted == true. Block 3 should NOT fire because compute aborted.
// ---------------------------------------------------------------------------
void test_dispatch_abort() {
    printf("test-block-callback: dispatch_abort\n");

    constexpr int N_BLOCKS = 5;

    struct ggml_init_params iparams = {
        /*.mem_size   =*/ 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(iparams);
    assert(ctx != nullptr);

    ggml_cgraph * gf = build_fake_graph(ctx, N_BLOCKS);
    cb_state st;
    st.fail_at = 2;

    std::shared_ptr<llama_block_callback_state> state;
    llama_block_callback_install(/*sched=*/nullptr, gf, &test_block_cb, &st, nullptr, nullptr, state);
    assert(state != nullptr);

    auto * gf_mut = gf;
    const int n_nodes = ggml_graph_n_nodes(gf_mut);
    bool aborted = false;
    for (int i = 0; i < n_nodes && !aborted; i++) {
        ggml_tensor * node = ggml_graph_node(gf_mut, i);
        if (!llama_block_callback_dispatch(node, true, state.get())) {
            continue;
        }
        const bool cont = llama_block_callback_dispatch(node, false, state.get());
        if (!cont) {
            aborted = true;
        }
    }

    assert(st.call_count.load() == 3); // 0, 1, 2 — aborted at 2
    assert(st.records.size() == 3);
    assert(st.records[0].block_idx == 0 && !st.records[0].aborted);
    assert(st.records[1].block_idx == 1 && !st.records[1].aborted);
    assert(st.records[2].block_idx == 2 &&  st.records[2].aborted);
    assert(aborted);

    ggml_free(ctx);
    printf("  ok (aborted at block %d)\n", st.fail_at);
}

// ---------------------------------------------------------------------------
// Test 3 — name parser unit tests.
// ---------------------------------------------------------------------------
void test_parser() {
    printf("test-block-callback: parser\n");
    assert(llama_block_callback_parse_layer_suffix("ffn_inp-0") == 0);
    assert(llama_block_callback_parse_layer_suffix("attn_out-12") == 12);
    assert(llama_block_callback_parse_layer_suffix("ffn_out-31") == 31);
    assert(llama_block_callback_parse_layer_suffix("logits") == -1);
    assert(llama_block_callback_parse_layer_suffix("ffn_inp-") == -1);
    assert(llama_block_callback_parse_layer_suffix("ffn_inp-12x") == -1);
    assert(llama_block_callback_parse_layer_suffix("") == -1);
    assert(llama_block_callback_parse_layer_suffix(nullptr) == -1);
    assert(llama_block_callback_parse_layer_suffix("a-0-b") == -1);
    printf("  ok\n");
}

// ---------------------------------------------------------------------------
// Test 4 — end-to-end through the real ggml scheduler.
//
// Builds a tiny graph, installs the wrapper on a CPU scheduler, runs
// ggml_backend_sched_graph_compute, and asserts the user callback
// fires once per block during compute.
// ---------------------------------------------------------------------------
void test_scheduler_end_to_end() {
    printf("test-block-callback: scheduler_end_to_end\n");

    constexpr int N_BLOCKS = 4;

    // Load CPU backend (and any other registered backends) so the
    // scheduler can find a CPU device. Idempotent.
    ggml_backend_load_all();

    struct ggml_init_params iparams = {
        /*.mem_size   =*/ 4 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true, // scheduler will allocate
    };
    ggml_context * ctx = ggml_init(iparams);
    assert(ctx != nullptr);

    ggml_cgraph * gf = build_fake_graph(ctx, N_BLOCKS);

    // CPU backend + scheduler.
    ggml_backend_t cpu = ggml_backend_cpu_init();
    assert(cpu != nullptr);
    ggml_backend_sched_t sched = ggml_backend_sched_new(&cpu, /*bufts=*/nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, /*parallel=*/false, /*op_offload=*/false);
    assert(sched != nullptr);

    cb_state st;
    std::shared_ptr<llama_block_callback_state> state;
    const bool has_boundaries = llama_block_callback_install(sched, gf, &test_block_cb, &st, nullptr, nullptr, state);
    assert(has_boundaries);
    assert(state != nullptr);

    const enum ggml_status rc = ggml_backend_sched_graph_compute(sched, gf);
    assert(rc == GGML_STATUS_SUCCESS);

    assert(st.call_count.load() == N_BLOCKS);
    assert(st.records.size() == static_cast<size_t>(N_BLOCKS));
    std::set<int> seen;
    for (const auto & r : st.records) {
        assert(r.block_idx >= 0 && r.block_idx < N_BLOCKS);
        seen.insert(r.block_idx);
    }
    assert(static_cast<int>(seen.size()) == N_BLOCKS);

    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);
    ggml_free(ctx);
    printf("  ok (%d blocks, %d fires)\n", N_BLOCKS, st.call_count.load());
}

} // namespace

int main() {
    test_parser();
    test_dispatch_pure();
    test_dispatch_abort();
    test_scheduler_end_to_end();
    printf("test-block-callback: PASSED\n");
    return 0;
}