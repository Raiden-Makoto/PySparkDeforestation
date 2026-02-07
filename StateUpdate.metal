//
//  StateUpdate.metal
//  MSLGraphDiffusion
//
//  Created by Raiden Makoto on 2026-02-06.
//

#include "ShaderTypes.h"
#include <metal_stdlib>
#include <metal_atomic>

inline float silu(float x){
    return x / (1.0f + exp(-x));
}


kernel void apply_updates(
    device Node* nodes              [[buffer(0)]],
    device float* h                 [[buffer(1)]],
    device const float* pos_delta   [[buffer(2)]],
    device const float* h_agg       [[buffer(3)]],
    device const float* node_w      [[buffer(4)]],
    device const float* node_b      [[buffer(5)]],
    constant uint& hidden_dim       [[buffer(6)]],
    device float* node_inputs_out   [[buffer(7)]], // New: Save for backprop
    device float* node_z_out        [[buffer(8)]], // New: Save for backprop
    uint gid [[thread_position_in_grid]]
){
    float coord_scale = 0.01f;
    nodes[gid].pos += float3(pos_delta[gid*3], pos_delta[gid*3+1], pos_delta[gid*3+2]) * coord_scale;

    for (uint row = 0; row < hidden_dim; row++) {
        float activation = node_b[row];
        
        // Save inputs once per node
        if (row == 0) {
            for (uint i = 0; i < 2 * hidden_dim; i++) {
                node_inputs_out[gid * 2 * hidden_dim + i] = (i < hidden_dim) ? h[gid * hidden_dim + i] : h_agg[gid * hidden_dim + (i - hidden_dim)];
            }
        }

        for (uint col = 0; col < 2 * hidden_dim; col++) {
            float val = node_inputs_out[gid * 2 * hidden_dim + col];
            activation += val * node_w[row * (2 * hidden_dim) + col];
        }
        
        node_z_out[gid * hidden_dim + row] = activation; // Save Z
        h[gid * hidden_dim + row] += (activation / (1.0f + exp(-activation))) * 0.1f;
    }
}

kernel void compute_cog(
                device const Node* nodes [[buffer(0)]],
                device atomic_float* cog_sum [[buffer(1)]], // [3] (x, y, z)
                uint gid [[thread_position_in_grid]]
){
                // Atomic sum of all atom positions
                atomic_fetch_add_explicit(&cog_sum[0], nodes[gid].pos.x, memory_order_relaxed);
                atomic_fetch_add_explicit(&cog_sum[1], nodes[gid].pos.y, memory_order_relaxed);
                atomic_fetch_add_explicit(&cog_sum[2], nodes[gid].pos.z, memory_order_relaxed);
}

kernel void apply_cog_normalization(
                device Node* nodes              [[buffer(0)]],
                device const float* cog_sum     [[buffer(1)]],
                constant uint& node_count       [[buffer(2)]],
                uint gid [[thread_position_in_grid]]
){
                // Calculate mean on the fly to avoid CPU round-trips
                float3 mean_pos = float3(cog_sum[0], cog_sum[1], cog_sum[2]) / (float) node_count;
                
                // Centering the molecule
                nodes[gid].pos -= mean_pos;
}
