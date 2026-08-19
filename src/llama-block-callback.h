// Per-block eval callback dispatch (JAISIU-2319-A, prismalama fork)
//
// Layered on top of ggml_backend_sched_set_eval_callback. The upstream
// ggml eval callback fires per tensor node; this helper instead fires
// the user-supplied `llama_block_cb` exactly once per transformer block
// boundary while still respecting any per-node cb_eval the caller
// installed via llama_context_params.cb_eval.
//
// The public C API surface lives in include/llama.h
// (`llama_block_cb` typedef and `llama_set_block_callback`).
//
// Internal layering:
//   - llama_block_callback_state holds { node -> block_idx } map,
//     the user_cb, user_data, and the optional fallback per-node cb.
//   - llama_block_callback_dispatch is the C function pointer we hand
//     to ggml_backend_sched_set_eval_callback. The GGML scheduler calls
//     it with `ask==true` to learn which nodes to observe, and with
//     `ask==false` to hand a computed node back to us.
//
// This file is independent of llama_context and does NOT depend on a
// loaded model — that lets tests/test-block-callback.cpp build a fake
// graph and exercise the dispatch helper directly.

#pragma once

#include "llama.h"

#include <ggml-backend.h>
#include <ggml.h>

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

struct llama_block_callback_state {
    // node pointer -> block_idx (0-based, model layer order).
    // Empty when no block_cb is installed (dispatch is a passthrough).
    std::unordered_map<ggml_tensor *, int> block_node_to_idx;

    // User-installed per-block callback + user data. Mirrors cparams.block_cb.
    llama_block_cb user_block_cb = nullptr;
    void *         user_block_data = nullptr;

    // Optional per-node callback from cparams.cb_eval. We delegate to it
    // for nodes that are NOT block boundaries, so existing per-node
    // observability keeps working alongside the new per-block one.
    ggml_backend_sched_eval_callback user_node_cb = nullptr;
    void *                           user_node_cb_data = nullptr;
};

// Parse the trailing -<il> suffix that llama.cpp graph_get_cb() applies
// to per-layer tensor names ("ffn_inp-3", "attn_out-12", ...).
// Returns -1 if the name does not have a parseable layer suffix.
int llama_block_callback_parse_layer_suffix(const char * name);

// Build block_node_to_idx from a ggml_cgraph: walk every node, parse its
// name suffix, and keep the LAST node seen for each layer. The graph is
// expected to have been built with llama.cpp's standard graph_get_cb
// naming convention; other layouts will fall back to "no boundaries".
void llama_block_callback_scan(
        const ggml_cgraph * gf,
        llama_block_callback_state & state);

// C-compatible dispatch function we hand to ggml_backend_sched_set_eval_callback.
// Mirrors ggml_backend_sched_eval_callback semantics.
bool llama_block_callback_dispatch(ggml_tensor * t, bool ask, void * user_data);

// Convenience: scan the graph, then install the dispatch wrapper on the
// scheduler. Clears any prior block callback. Pass NULL user_block_cb to
// install a passthrough that defers only to user_node_cb.
//
// Returns true if any block boundaries were registered (i.e. the user_cb
// will fire during the next graph compute). Returns false if no block
// boundaries were found (graph uses non-<name>-<il> naming, or empty).
//
// The state object is heap-allocated; the returned shared_ptr keeps it
// alive for as long as the scheduler holds the callback.
bool llama_block_callback_install(
        ggml_backend_sched_t sched,
        const ggml_cgraph * gf,
        llama_block_cb user_block_cb,
        void *         user_block_data,
        ggml_backend_sched_eval_callback user_node_cb,
        void *         user_node_cb_data,
        std::shared_ptr<llama_block_callback_state> & out_state);

// Clear a previously installed block callback on the scheduler. Safe to
// call multiple times.
void llama_block_callback_clear(ggml_backend_sched_t sched);