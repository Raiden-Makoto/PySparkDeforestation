//
//  StateUpdate.metal
//  MSLGraphDiffusion
//
//  Created by Raiden Makoto on 2026-02-08.
//

#include <metal_stdlib>
#include "ShaderTypes.h"
#include <metal_atomic>

using namespace metal;

kernel void aggregate(
    device const float* msg          [[buffer(0)]], // [E * 128]
    device const float* trans        [[buffer(1)]], // [E * 3]
    device const int2* edge_index    [[buffer(2)]], // [E]
    device atomic_float* m_agg       [[buffer(3)]], // [N * 128]
    device atomic_float* pos_agg     [[buffer(4)]], // [N * 3]
    constant uint& hidden_dim        [[buffer(5)]],
    constant uint& num_edges         [[buffer(6)]], // ADDED
    uint gid [[thread_position_in_grid]]
){
    if (gid >= num_edges){ return; }
    int i = edge_index[gid].x;
    // Aggregate messages into node i
    for (uint h = 0; h < hidden_dim; h++){
        atomic_fetch_add_explicit(&m_agg[i * hidden_dim + h], msg[gid * hidden_dim + h], memory_order_relaxed);
    }
    // Aggregate coordinate translations into node i
    for (uint c = 0; c < 3; c++){
        atomic_fetch_add_explicit(&pos_agg[i * 3 + c], ((device float*)&trans[gid])[c], memory_order_relaxed);
    }
}

// STABILIZED UPDATE KERNEL
kernel void apply_update(
    device Node* nodes               [[buffer(0)]],
    device float* h                  [[buffer(1)]],
    device const float* h_update     [[buffer(2)]],
    device const float* pos_agg      [[buffer(3)]],
    constant uint& hidden_dim        [[buffer(4)]],
    constant uint& num_nodes         [[buffer(5)]],
    constant float& current_t        [[buffer(6)]], // Pass Float(t) from Swift
    uint gid [[thread_position_in_grid]])
{
    if (gid >= num_nodes) return;

    uint pBase = gid * 3;
    float3 move = float3(pos_agg[pBase], pos_agg[pBase + 1], pos_agg[pBase + 2]);

    // DIFFUSION SCALING: Updates should be proportional to the current timestep
    // As t goes from 500 -> 1, this factor scales from 1.0 down to 0.002.
    float t_normalized = current_t / 500.0f;
    float t_scale = exp(5.0f * (t_normalized - 1.0f));
    
    // Applying the 1/(N-1) normalization explicitly here
    float norm = 1.0f / (float)(num_nodes - 1);
    
    float3 final_update = move * norm * t_scale;

    // HARD CLAMP: Absolute safety to prevent teleportation
    if (length(final_update) > 0.03f) final_update = normalize(final_update) * 0.03f;

    nodes[gid].pos += final_update;

    for (uint i = 0; i < hidden_dim; i++) {
        h[gid * hidden_dim + i] += h_update[gid * hidden_dim + i];
    }
}
