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
        atomic_fetch_add_explicit(&pos_agg[i * 3 + c], trans[gid * 3 + c], memory_order_relaxed);
    }
}

// STABILIZED UPDATE KERNEL
kernel void apply_diffusion(
                            device Node* nodes               [[buffer(0)]],
                            device const float* alphas       [[buffer(1)]],
                            device const float* alphas_cp    [[buffer(2)]],
                            device const float* pos_agg      [[buffer(3)]], // Model epsilon prediction
                            constant uint& current_t         [[buffer(4)]],
                            constant uint& num_nodes         [[buffer(5)]],
                            uint gid [[thread_position_in_grid]]
                            ){
    if (gid >= num_nodes){ return; }
    float a_t = alphas[current_t];
    float a_bar_t = max(alphas_cp[current_t], 1e-6);
    
    // 2. Extract model output (epsilon)
    uint pBase = gid * 3;
    float3 epsilon = float3(pos_agg[pBase], pos_agg[pBase + 1], pos_agg[pBase + 2]);
    
    // 3. DDPM Reverse Step: x_{t-1} calculation
    // This formula removes the predicted noise from the current position
    float coeff = (1.0f - a_t) / sqrt(1.0f - a_bar_t + 1e-6f);
    float3 x_next = (1.0f / sqrt(a_t + 1e-6f)) * (nodes[gid].pos - coeff * epsilon);

    // 4. Update position
    nodes[gid].pos = x_next;
}
