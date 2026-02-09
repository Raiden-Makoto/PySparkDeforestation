//
//  Geometry.metal
//  MSLGraphDiffusion
//
//  Created by Raiden-Makoto on 2026-02-08.
//

#include <metal_stdlib>
#include <metal_atomic>
#include "ShaderTypes.h"

using namespace metal;

kernel void force_zero_center(
    device Node* nodes        [[buffer(0)]],
    constant uint& num_nodes  [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    // Only one thread executes this to ensure perfect precision
    if (gid != 0) return;

    float3 sum = float3(0.0f);
    for (uint i = 0; i < num_nodes; i++) {
        sum += nodes[i].pos;
    }
    
    float3 centroid = sum / (float)num_nodes;
    
    // Shift every node to be centered at (0,0,0)
    for (uint i = 0; i < num_nodes; i++) {
        nodes[i].pos -= centroid;
    }
}
