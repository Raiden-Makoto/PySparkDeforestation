//
//  EGNNKernel.metal
//  MSLGraphDiffusion
//
//  Created by Raiden Makoto on 2026-02-05.
//

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

// Metal cannot see Swift structs
struct Node {
    float3 pos;
    float atomType;
};

inline float silu(float x){
    return x / (1.0f + exp(-x));
}

kernel void compute_message(
    device const Node* nodes [[buffer(0)]], // from nodes.bin
    device const float* h [[buffer(1)]], // node features (hidden_dim)
    device const int2* edge_index [[buffer(2)]], // from edges.bin
    device const float* weights_l1 [[buffer(3)]], // [hidden_dim, hidden_dim*2+1]
    device const float* bias_l1 [[buffer(4)]], // [hidden_dim]
    device float* message_out [[buffer(5)]], // num_edges * hidden_dim
    constant uint& hidden_dim [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
){
    int idx = edge_index[gid].x;
    int jdx = edge_index[gid].y;
    // compute radial distances (equivarian feature)
    float3 pos_i = nodes[idx].pos;
    float3 pos_j = nodes[jdx].pos;
    float3 diff = pos_i - pos_j;
    float radial = dot(diff, diff) + 1e-8f;
    // Linear Layer 1
    uint input_dim = 2 * hidden_dim + 1;
    for (uint row = 0; row < hidden_dim; row++){
        float accumulated = bias_l1[row];
        for (uint col = 0; col < input_dim; col++){
            float val;
            if (col < hidden_dim){
                val = h[idx *  hidden_dim + col];
            } else if (col < 2 * hidden_dim){
                val = h[jdx * hidden_dim + (col - hidden_dim)];
            } else {
                val = radial;
            }
            accumulated += val * weights_l1[row * input_dim + col];
        }
        message_out[gid * hidden_dim + row] = silu(accumulated);
    }
}

kernel void aggregate_message(
    device const float* messages [[buffer(0)]],
    device const int2* edge_index [[buffer(1)]],
    device atomic_float* node_agg_out [[buffer(2)]],
    constant uint& hidden_dim [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
){
    int target_node = edge_index[gid].y;
    for (uint h = 0; h < hidden_dim; h++){
        float msg = messages[gid * hidden_dim + h];
        atomic_fetch_add_explicit(
            &node_agg_out[target_node * hidden_dim + h],
            msg,
            memory_order_relaxed
        );
    }
}

kernel void update_coords(
    device const Node* nodes [[buffer(0)]], // Original positions
    device const float* messages [[buffer(1)]], // From compute_message [E * H]
    device const int2* edge_index [[buffer(2)]], // [E * 2]
    device const float* coord_weights [[buffer(3)]], // [1 * H] (coord_mlp)
    device const float* coord_bias [[buffer(4)]], // [1]
    device atomic_float* pos_agg_out [[buffer(5)]], // [N * 3] (x, y, z updates)
    constant uint& hidden_dim [[buffer(6)]],
    uint gid [[thread_position_in_grid]] // Thread per Edge
){
    int idx = edge_index[gid].x;
    int jdx = edge_index[gid].y;
    float3 pos_i = nodes[idx].pos;
    float3 pos_j = nodes[jdx].pos;
    float3 diff = pos_i - pos_j;
    // multi-layer perceptron
    float weight = coord_bias[0];
    for (uint h = 0; h < hidden_dim; h++) {
        weight += messages[gid * hidden_dim + h] * coord_weights[h];
    }
    // scale the displacement
    float3 displacement = diff * weight;
    // Atomic Aggregation into Node Positions
    // We update x, y, and z separately using atomic_fetch_add
    atomic_fetch_add_explicit(&pos_agg_out[idx * 3 + 0], displacement.x, memory_order_relaxed);
    atomic_fetch_add_explicit(&pos_agg_out[idx * 3 + 1], displacement.y, memory_order_relaxed);
    atomic_fetch_add_explicit(&pos_agg_out[idx * 3 + 2], displacement.z, memory_order_relaxed);
}

// for the sinusoidal timestep embedding in diffusion
kernel void inject_timestep(
    device float* h [[buffer(0)]], // Node features [N * H]
    constant float* t_emb [[buffer(1)]], // Time embedding [H]
    constant uint& hidden_dim [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
){
    for (uint i = 0; i < hidden_dim; i++) {
        h[gid * hidden_dim + i] += t_emb[i]; // h = h + t_emb
    }
}

// mimics an embedding layer (like from PyTorch)
kernel void embed_atoms(
    device const Node* nodes [[buffer(0)]],
    device const float* embed_table [[buffer(1)]], // [num_types * hidden_dim]
    device float* h_out [[buffer(2)]], // [N * hidden_dim]
    constant uint& hidden_dim [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
){
    int type_idx = (int) nodes[gid].atomType;
    for (uint i = 0; i < hidden_dim; i++) {
        h_out[gid * hidden_dim + i] = embed_table[type_idx * hidden_dim + i];
    }
}

kernel void apply_updates(
    device Node* nodes              [[buffer(0)]], // Mutable positions
		  device float* h                 [[buffer(1)]], // Mutable features
		  device const float* pos_delta   [[buffer(2)]], // From update_coords
		  device const float* h_agg       [[buffer(3)]], // Aggregated messages
		  device const float* node_w      [[buffer(4)]], // Node MLP Weights [H, 2H]
		  device const float* node_b      [[buffer(5)]], // Node MLP Bias [H]
		  constant uint& hidden_dim       [[buffer(6)]],
		  uint gid [[thread_position_in_grid]]){
				// 1. Move Atoms
				nodes[gid].pos += float3(pos_delta[gid*3], pos_delta[gid*3+1], pos_delta[gid*3+2]);
				// 2. Node MLP
				// We concatenate h and h_agg virtually here
				for (uint row = 0; row < hidden_dim; row++) {
								float activation = node_b[row];
								for (uint col = 0; col < 2 * hidden_dim; col++) {
												float val = (col < hidden_dim) ? h[gid * hidden_dim + col] : h_agg[gid * hidden_dim + (col - hidden_dim)];
												activation += val * node_w[row * (2 * hidden_dim) + col];
								}
								// 3. Final Residual Update: h = h + SiLU(MLP(h, m_agg))
								h[gid * hidden_dim + row] += silu(activation);
				}
}
