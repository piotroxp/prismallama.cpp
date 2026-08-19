// Per-block eval callback dispatch — implementation.
// See llama-block-callback.h for the contract.

#include "llama-block-callback.h"

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

// Forward decl: ggml_format_name returns ggml_tensor * — not used here,
// but we use ggml_get_name (public).

int llama_block_callback_parse_layer_suffix(const char * name) {
    if (name == nullptr || name[0] == '\0') {
        return -1;
    }
    // Walk to the end of the string, then scan back to find a '-' followed
    // by digits. We require at least one digit and stop at the first
    // non-digit or at the start of the string.
    const size_t n = std::strlen(name);
    if (n < 2) {
        return -1;
    }
    size_t i = n;
    // Walk back over trailing digits.
    while (i > 0 && name[i - 1] >= '0' && name[i - 1] <= '9') {
        i--;
    }
    // Need at least one digit AND a '-' separator right before it.
    if (i == n || i == 0 || name[i - 1] != '-') {
        return -1;
    }
    const char * digits = name + i;
    char * end = nullptr;
    long val = std::strtol(digits, &end, 10);
    if (end == digits || *end != '\0') {
        return -1;
    }
    if (val < 0 || val > INT32_MAX) {
        return -1;
    }
    return static_cast<int>(val);
}

void llama_block_callback_scan(
        const ggml_cgraph * gf,
        llama_block_callback_state & state) {
    state.block_node_to_idx.clear();
    if (gf == nullptr) {
        return;
    }

    // ggml's API takes a non-const pointer but only reads from it.
    auto * gf_mut = const_cast<ggml_cgraph *>(gf);
    const int n_nodes = ggml_graph_n_nodes(gf_mut);

    // Pass 1: collect the last node index per layer. There can be many
    // tensors per layer (ffn_inp-i, attn_out-i, ffn_out-i, ...) but only
    // ONE boundary fire per block. We pick the last one seen in graph
    // node order.
    std::unordered_map<int, ggml_tensor *> last_per_layer;
    int max_il = -1;
    for (int i = 0; i < n_nodes; i++) {
        ggml_tensor * node = ggml_graph_node(gf_mut, i);
        if (node == nullptr) {
            continue;
        }
        const char * name = ggml_get_name(node);
        const int il = llama_block_callback_parse_layer_suffix(name);
        if (il < 0) {
            continue;
        }
        last_per_layer[il] = node; // overwrite: keeps the latest
        if (il > max_il) {
            max_il = il;
        }
    }

    // Pass 2: insert boundary map entries only for the LAST node per layer.
    // Iterating last_per_layer keeps insertion order stable for tests.
    state.block_node_to_idx.clear();
    for (int il = 0; il <= max_il; il++) {
        auto it = last_per_layer.find(il);
        if (it == last_per_layer.end()) {
            continue;
        }
        state.block_node_to_idx[it->second] = il;
    }
}

bool llama_block_callback_dispatch(ggml_tensor * t, bool ask, void * user_data) {
    auto * state = static_cast<llama_block_callback_state *>(user_data);
    if (state == nullptr) {
        return true;
    }

    const bool is_boundary = state->user_block_cb != nullptr
            && state->block_node_to_idx.find(t) != state->block_node_to_idx.end();

    if (ask) {
        // Scheduler is asking whether it should observe this node.
        // We want to observe boundary nodes (to fire block_cb) and any
        // node the user-supplied per-node callback wants to observe.
        if (is_boundary) {
            return true;
        }
        if (state->user_node_cb != nullptr) {
            return state->user_node_cb(t, true, state->user_node_cb_data);
        }
        return false;
    }

    // Scheduler is delivering a computed node.
    if (is_boundary) {
        const int blk = state->block_node_to_idx[t];
        // User contract: non-zero = continue compute, zero = abort.
        // The GGML eval-callback contract is "return true to continue,
        // false to cancel" — translate.
        const int rc = state->user_block_cb(state->user_block_data, blk);
        return rc != 0;
    }
    if (state->user_node_cb != nullptr) {
        return state->user_node_cb(t, false, state->user_node_cb_data);
    }
    return true;
}

bool llama_block_callback_install(
        ggml_backend_sched_t sched,
        const ggml_cgraph * gf,
        llama_block_cb user_block_cb,
        void *         user_block_data,
        ggml_backend_sched_eval_callback user_node_cb,
        void *         user_node_cb_data,
        std::shared_ptr<llama_block_callback_state> & out_state) {
    // Always populate out_state so callers (and tests) can drive the
    // dispatch wrapper manually even without a scheduler. Pass
    // sched == nullptr for "scan + state only, do not install".
    auto state = std::make_shared<llama_block_callback_state>();
    state->user_block_cb    = user_block_cb;
    state->user_block_data  = user_block_data;
    state->user_node_cb     = user_node_cb;
    state->user_node_cb_data = user_node_cb_data;
    llama_block_callback_scan(gf, *state);

    if (sched != nullptr) {
        ggml_backend_sched_set_eval_callback(
                sched,
                &llama_block_callback_dispatch,
                state.get());
    }

    out_state = std::move(state);
    return !out_state->block_node_to_idx.empty();
}

void llama_block_callback_clear(ggml_backend_sched_t sched) {
    if (sched == nullptr) {
        return;
    }
    ggml_backend_sched_set_eval_callback(sched, nullptr, nullptr);
}