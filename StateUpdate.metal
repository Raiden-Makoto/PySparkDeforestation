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
    device Node* nodes               [[buffer(0)]], // [N]
    device float* h                  [[buffer(1)]], // [N * 128]
    device const float* h_update     [[buffer(2)]], // [N * 128]
    device const float* pos_agg      [[buffer(3)]], // [N * 3] (Summed forces)
    constant uint& hidden_dim        [[buffer(4)]],
    constant uint& num_nodes         [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= num_nodes) return;

    // --- 1. STABILIZE POSITION UPDATE ---
    
    // Read the aggregated force vector (Sum of all neighbors)
    uint pBase = gid * 3;
    float3 agg_force = float3(pos_agg[pBase], pos_agg[pBase + 1], pos_agg[pBase + 2]);

    // A. Normalization: Divide by number of neighbors (N-1)
    // For Methane (N=5), we divide by 4.0. This keeps the scale consistent.
    float neighbor_scale = 1.0f / max(1.0f, (float)num_nodes - 1.0f);
    
    // B. Damping: Reduce the step size slightly to prevent oscillation
    float damping = 0.5f;
    
    float3 move_vec = agg_force * neighbor_scale * damping;

    // C. Hard Speed Limit (Clamping)
    // Don't let any atom move more than 0.5 Angstroms in a single step.
    // This prevents the "Teleportation" to coordinates like -600.
    float dist = length(move_vec);
    if (dist > 0.5f) {
        move_vec = normalize(move_vec) * 0.5f;
    }

    // Apply the safe update
    nodes[gid].pos += move_vec;

    // --- 2. UPDATE FEATURES (Standard Residual) ---
    for (uint i = 0; i < hidden_dim; i++) {
        h[gid * hidden_dim + i] += h_update[gid * hidden_dim + i];
    }
}
