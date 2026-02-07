//
//  Embedding.metal
//  MSLGraphDiffusion
//
//  Created by Raiden-Makoto on 2026-02-06.
//

#include "ShaderTypes.h"
#include <metal_stdlib>

// for the sinusoidal timestep embedding in diffusion
kernel void inject_timestep(
    device float* h [[buffer(0)]], // Node features [N * H]
    constant float* t_emb [[buffer(1)]], // Time embedding [H]
    constant uint& hidden_dim [[buffer(2)]],
    constant uint& node_count [[buffer(3)]], // N
    uint gid [[thread_position_in_grid]]
){
    if (gid >= node_count) {
        return;
    }
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
    constant uint& node_count [[buffer(4)]],
    constant uint& num_types [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
){
    // Ensure thread index is within the number of nodes
    if (gid >= node_count) {
        return;
    }

    // Safely read and validate atom type
    int type_idx = (int)nodes[gid].atomType;
    if (type_idx < 0) {
        type_idx = 0;
    }
    if ((uint)type_idx >= num_types) {
        // Out-of-range type; skip to avoid out-of-bounds table access
        return;
    }

    for (uint i = 0; i < hidden_dim; i++) {
        h_out[gid * hidden_dim + i] = embed_table[type_idx * hidden_dim + i];
    }
}
