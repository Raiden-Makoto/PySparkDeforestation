//
//  EGNNKernel.metal
//  MSLGraphDiffusion
//
//  Created by Raiden Makoto on 2026-02-05.
//

#include <metal_stdlib>
#include <metal_atomic>
#include "ShaderTypes.h"
using namespace metal;

inline float silu(float x){
    return x / (1.0f + exp(-x));
}

kernel void compute_message(
    device const Node* nodes [[buffer(0)]],
    device const float* h [[buffer(1)]],
    device const int2* edge_index [[buffer(2)]],
    device const float* weights_l1 [[buffer(3)]],
    device const float* bias_l1 [[buffer(4)]],
    device float* message_out [[buffer(5)]],
    constant uint& hidden_dim [[buffer(6)]],
    device float* inputs_out     [[buffer(7)]],
    device float* pre_activ_out  [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
){
    int idx = edge_index[gid].x;
    int jdx = edge_index[gid].y;
    
    float3 diff = nodes[idx].pos - nodes[jdx].pos;
    float dist = sqrt(dot(diff, diff) + 1e-8f);
    float radial = 1.0f / (1.0f + dist + 1e-6f); // Guarded denominator
    
    uint input_dim = 2 * hidden_dim + 1;

    // --- SAVE INPUTS FOR BACKPROP ---
    // We save h_i, h_j, and radial so backward_message can use them
    for (uint i = 0; i < input_dim; i++) {
        float val;
        if (i < hidden_dim) val = h[idx * hidden_dim + i];
        else if (i < 2 * hidden_dim) val = h[jdx * hidden_dim + (i - hidden_dim)];
        else val = radial;
        inputs_out[gid * input_dim + i] = val;
    }

    // --- FORWARD PASS ---
    for (uint row = 0; row < hidden_dim; row++){
        float accumulated = bias_l1[row];
        for (uint col = 0; col < input_dim; col++){
            // Use the already-saved inputs to ensure forward/backward consistency
            float val = inputs_out[gid * input_dim + col];
            accumulated += val * weights_l1[row * input_dim + col];
        }
        
        // Use 'row' instead of 'h' and 'accumulated' instead of 'activation'
        pre_activ_out[gid * hidden_dim + row] = accumulated;
        message_out[gid * hidden_dim + row] = accumulated / (1.0f + exp(-accumulated)); // Manual SiLU
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

