//
//  Optimization.metal
//  MSLGraphDiffusion
//
//  Created by Raiden Makoto on 2026-02-06.
//

#include <metal_stdlib>
#include "ShaderTypes.h"

kernel void compute_grad_norm_sq(
    device const float* gradients [[buffer(0)]],
    device atomic_float* sum_sq    [[buffer(1)]],
    constant uint& count           [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
){
    if (gid >= count) return;
    float g = gradients[gid];
    atomic_fetch_add_explicit(sum_sq, g * g, memory_order_relaxed);
}

kernel void apply_clipping(
    device float* gradients      [[buffer(0)]],
    device const float* norm_sq  [[buffer(1)]],
    constant float& max_norm     [[buffer(2)]],
    constant uint& count         [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
){
    if (gid >= count) return;
    
    float norm = sqrt(*norm_sq);
    if (norm > max_norm) {
        // Scale factor: max_norm / actual_norm
        gradients[gid] *= (max_norm / (norm + 1e-6f));
    }
}

float silu_derivative(float x) {
    float sig = 1.0f / (1.0f + exp(-x));
    return sig * (1.0f + x * (1.0f - sig));
}

kernel void backward_message(
    device const float* grad_msg       [[buffer(0)]], // Incoming gradient from aggregation
    device const float* weights        [[buffer(1)]], // Forward weights [H, 2H+1]
    device const float* inputs         [[buffer(2)]], // Original inputs to msg MLP (h_i, h_j, radial)
    device const float* pre_activations[[buffer(3)]], // Z values before SiLU
    device float* grad_weights_out     [[buffer(4)]], // Weight gradients to accumulate
    device float* grad_inputs_out      [[buffer(5)]], // Gradients for h_i, h_j (for next backprop step)
    constant uint& hidden_dim          [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
){
    // threads per edge
    uint in_dim = 2 * hidden_dim + 1;
    for (uint h = 0; h < hidden_dim; h++) {
        // 1. Backprop through SiLU
        float d_activ = grad_msg[gid * hidden_dim + h] * silu_derivative(pre_activations[gid * hidden_dim + h]);
        for (uint i = 0; i < in_dim; i++) {
            float x = inputs[gid * in_dim + i];
            // 2. Gradient w.r.t Weight: dL/dW = d_activ * input
            // Use atomic_add because multiple edges contribute to weight gradients
            atomic_fetch_add_explicit(
                (device atomic_float*) &grad_weights_out[h * in_dim + i],
                d_activ * x,
                memory_order_relaxed
            );
            // 3. Gradient w.r.t Input (for propagating to features h)
            if (i < 2 * hidden_dim) { // Skip radial for h backprop
                grad_inputs_out[gid * in_dim + i] = d_activ * weights[h * in_dim + i];
            }
        }
    }
}

kernel void backward_node(
    device const float* grad_h_next    [[buffer(0)]], // dL/dh from next layer
    device const float* node_weights   [[buffer(1)]], // nodeWBuffer [H, 2H]
    device const float* node_inputs    [[buffer(2)]], // Saved (h_i, m_agg)
    device const float* node_z         [[buffer(3)]], // Saved pre-activations
    device float* grad_node_w_out      [[buffer(4)]], // OUTPUT: weight grads
    device float* grad_node_b_out      [[buffer(5)]], // OUTPUT: bias grads
    device float* grad_h_prev_out      [[buffer(6)]], // OUTPUT: dL/dh for prev layer
    constant uint& hidden_dim          [[buffer(7)]],
    uint gid [[thread_position_in_grid]]
){
    for (uint h = 0; h < hidden_dim; h++) {
        // Derivative of SiLU residual
        float z = node_z[gid * hidden_dim + h];
        float sig = 1.0f / (1.0f + exp(-z));
        float d_silu = (sig * (1.0f + z * (1.0f - sig))) * 0.1f; // Matching forward scaling
        float d_activ = grad_h_next[gid * hidden_dim + h] * d_silu;
        
        // 1. Bias Gradient
        atomic_fetch_add_explicit((device atomic_float*)&grad_node_b_out[h], d_activ, memory_order_relaxed);
        
        for (uint i = 0; i < 2 * hidden_dim; i++) {
            float x = node_inputs[gid * 2 * hidden_dim + i];
            
            // 2. Weight Gradient
            atomic_fetch_add_explicit(
                (device atomic_float*)&grad_node_w_out[h * 2 * hidden_dim + i],
                d_activ * x, memory_order_relaxed
            );
            
            // 3. Propagate to previous hidden state
            if (i < hidden_dim) {
                atomic_fetch_add_explicit(
                    (device atomic_float*)&grad_h_prev_out[gid * hidden_dim + i],
                    d_activ * node_weights[h * 2 * hidden_dim + i], memory_order_relaxed
                );
            }
        }
    }
}

kernel void backward_coordinate(
    device const float* grad_pos_next   [[buffer(0)]], // dL/dpos [N*3]
    device const float* coord_weights   [[buffer(1)]], // coordWeightBuffer [H]
    device const float* msg_inputs      [[buffer(2)]], // msgBuffer [E*H]
    device const int2* edge_index       [[buffer(3)]], // From edges.bin
    device float* grad_coord_w_out      [[buffer(4)]], // OUTPUT: weight grads
    device float* grad_msg_out          [[buffer(5)]], // OUTPUT: dL/dmsg
    constant uint& hidden_dim           [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) // Thread per edge
{
    int idx = edge_index[gid].x;
    int jdx = edge_index[gid].y;

    for (uint h = 0; h < hidden_dim; h++) {
        float m_h = msg_inputs[gid * hidden_dim + h];
        float w_h = coord_weights[h];
        
        // Error from the node's position change
        float3 d_pos = float3(grad_pos_next[idx * 3], grad_pos_next[idx * 3 + 1], grad_pos_next[idx * 3 + 2]);
        
        // Weight Gradient: dL/dW = dL/dpos * m_h
        atomic_fetch_add_explicit((device atomic_float*)&grad_coord_w_out[h], dot(d_pos, d_pos) * m_h, memory_order_relaxed);
        
        // Propagate to message: dL/dm = dL/dpos * w_h
        grad_msg_out[gid * hidden_dim + h] += dot(d_pos, d_pos) * w_h;
    }
}

kernel void apply_adam_update(
    device float* weights         [[buffer(0)]], // Your W or Bias
    device float* m               [[buffer(1)]], // First moment (Mean)
    device float* v               [[buffer(2)]], // Second moment (Variance)
    device const float* gradients [[buffer(3)]], // Clipped gradients
    constant float& lr            [[buffer(4)]], // Learning rate (e.g., 1e-4)
    constant float& beta1         [[buffer(5)]], // Decay (0.9)
    constant float& beta2         [[buffer(6)]], // Decay (0.999)
    constant float& epsilon       [[buffer(7)]], // Stability (1e-8)
    constant uint& t              [[buffer(8)]], // Current Epoch
    uint gid [[thread_position_in_grid]]
){
    float g = gradients[gid];

    // 1. Update biased moment estimates
    m[gid] = beta1 * m[gid] + (1.0f - beta1) * g;
    v[gid] = beta2 * v[gid] + (1.0f - beta2) * (g * g);

    // 2. Compute bias-corrected moment estimates
    // These account for the moments starting at zero in early epochs
    float m_hat = m[gid] / (1.0f - pow(beta1, (float)t));
    float v_hat = v[gid] / (1.0f - pow(beta2, (float)t));

    // 3. Update weights
    // This is where the magic happens: the update is scaled by 1/sqrt(v)
    weights[gid] -= lr * m_hat / (sqrt(v_hat) + epsilon);
}

