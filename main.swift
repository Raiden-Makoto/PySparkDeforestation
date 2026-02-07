//
//  main.swift
//  MSLGraphDiffusion
//
//  Created by Raiden Makoto on 2026-02-04.
//

import Metal
import Foundation

// --- UTILITIES ---

func KaimingInit(_ buffer: MTLBuffer, count: Int, fanIn: Int) {
    let ptr = buffer.contents().bindMemory(to: Float.self, capacity: count)
    let std = sqrt(2.0 / Double(fanIn))
    for i in 0..<count {
        let u1 = max(Float.leastNonzeroMagnitude, Float.random(in: 0..<1))
        let u2 = Float.random(in: 0..<1)
        let z = Float(sqrt(-2.0 * log(Double(u1))) * cos(2.0 * Double.pi * Double(u2)))
        ptr[i] = z * Float(std)
    }
}

func ZeroInit(_ buffer: MTLBuffer) {
    memset(buffer.contents(), 0, buffer.length)
}

func TimestepEmbedding(t: Float, dim: Int) -> [Float]{
    var embedding = [Float](repeating: 0, count: dim)
    let halfDim = dim / 2
    let exponent = log(10000.0) / Double(halfDim - 1)
    for i in 0..<halfDim {
        let freq = exp(-exponent * Double(i))
        let arg = Double(t) * freq
        embedding[i] = Float(sin(arg))
        embedding[i + halfDim] = Float(cos(arg))
    }
    return embedding
}

func SaveModelWeights(buffers: [(String, MTLBuffer)], path: String) {
    for (name, buffer) in buffers {
        let fileURL = URL(fileURLWithPath: path).appendingPathComponent("\(name).bin")
        let data = Data(bytes: buffer.contents(), count: buffer.length)
        do {
            try data.write(to: fileURL)
            print("Saved \(name) to \(fileURL.lastPathComponent)")
        } catch {
            print("Failed to save \(name): \(error)")
        }
    }
}

// --- INITIALIZATION ---

guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("Metal is not supported on this device!")
}

print("Engine initialized on: \(device.name)")
let commandQueue = device.makeCommandQueue()!
let loader = QM9Loader(device: device)
let sourceDirURL = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
let datapath = sourceDirURL.path

do {
    try loader.load(from: datapath)
} catch {
    fatalError("Data Load Failed: \(error.localizedDescription)")
}

// --- DATASET SIZING & SPLITTING ---

let totalGraphs = 3662
let graphMetadata = loader.graphdataBuffer!.contents().bindMemory(to: GraphData.self, capacity: totalGraphs)

var totalNodes = 0
var totalEdges = 0
for i in 0..<totalGraphs {
    totalNodes += Int(graphMetadata[i].nodeCount)
    totalEdges += Int(graphMetadata[i].edgeCount)
}

let trainSplitIdx = Int(Double(totalGraphs) * 0.8)
var trainNodeCount = 0
var trainEdgeCount = 0
for i in 0..<trainSplitIdx {
    trainNodeCount += Int(graphMetadata[i].nodeCount)
    trainEdgeCount += Int(graphMetadata[i].edgeCount)
}

let valNodeCount = totalNodes - trainNodeCount
let valEdgeCount = totalEdges - trainEdgeCount
let hiddenDim = 64

// --- BUFFER ALLOCATIONS ---

// Parameters
let embedTableBuffer = device.makeBuffer(length: 10 * hiddenDim * 4, options: .storageModeShared)!
let weightsBuffer = device.makeBuffer(length: hiddenDim * (2 * hiddenDim + 1) * 4, options: .storageModeShared)!
let biasBuffer = device.makeBuffer(length: hiddenDim * 4, options: .storageModeShared)!
let nodeWBuffer = device.makeBuffer(length: hiddenDim * (2 * hiddenDim) * 4, options: .storageModeShared)!
let nodeBBuffer = device.makeBuffer(length: hiddenDim * 4, options: .storageModeShared)!
let coordWeightBuffer = device.makeBuffer(length: hiddenDim * 4, options: .storageModeShared)!
let coordBiasBuffer = device.makeBuffer(length: 4, options: .storageModeShared)!

// Training Features
let hBufferTrain = device.makeBuffer(length: trainNodeCount * hiddenDim * 4, options: .storageModeShared)!
let msgBufferTrain = device.makeBuffer(length: trainEdgeCount * hiddenDim * 4, options: .storageModeShared)!
let aggBufferTrain = device.makeBuffer(length: trainNodeCount * hiddenDim * 4, options: .storageModeShared)!
let posUpdateTrain = device.makeBuffer(length: trainNodeCount * 3 * 4, options: .storageModeShared)!
let targetNoiseTrain = device.makeBuffer(length: trainNodeCount * 3 * 4, options: .storageModeShared)!

// Validation Features
let hBufferVal = device.makeBuffer(length: valNodeCount * hiddenDim * 4, options: .storageModeShared)!
let msgBufferVal = device.makeBuffer(length: valEdgeCount * hiddenDim * 4, options: .storageModeShared)!
let aggBufferVal = device.makeBuffer(length: valNodeCount * hiddenDim * 4, options: .storageModeShared)!
let posUpdateVal = device.makeBuffer(length: valNodeCount * 3 * 4, options: .storageModeShared)!
let targetNoiseVal = device.makeBuffer(length: valNodeCount * 3 * 4, options: .storageModeShared)!

// Backprop State
let msgInputBuffer = device.makeBuffer(length: trainEdgeCount * (2 * hiddenDim + 1) * 4, options: .storageModeShared)!
let preActivBuffer = device.makeBuffer(length: trainEdgeCount * hiddenDim * 4, options: .storageModeShared)!
let nodeActivBuffer = device.makeBuffer(length: trainNodeCount * 2 * hiddenDim * 4, options: .storageModeShared)!
let preActivNodeBuffer = device.makeBuffer(length: trainNodeCount * hiddenDim * 4, options: .storageModeShared)!

// Gradients
let gradWeightsBuffer = device.makeBuffer(length: weightsBuffer.length, options: .storageModeShared)!
let gradBiasBuffer = device.makeBuffer(length: biasBuffer.length, options: .storageModeShared)!
let gradNodeWBuffer = device.makeBuffer(length: nodeWBuffer.length, options: .storageModeShared)!
let gradNodeBBuffer = device.makeBuffer(length: nodeBBuffer.length, options: .storageModeShared)!
let gradHBuffer = device.makeBuffer(length: hBufferTrain.length, options: .storageModeShared)!
let gradMsgBuffer = device.makeBuffer(length: msgBufferTrain.length, options: .storageModeShared)!
let gradPosBuffer = device.makeBuffer(length: posUpdateTrain.length, options: .storageModeShared)!
let gradInputBuffer = device.makeBuffer(length: msgInputBuffer.length, options: .storageModeShared)!

// Adam States
let weightsM = device.makeBuffer(length: weightsBuffer.length, options: .storageModeShared)!
let weightsV = device.makeBuffer(length: weightsBuffer.length, options: .storageModeShared)!
let nodeWM = device.makeBuffer(length: nodeWBuffer.length, options: .storageModeShared)!
let nodeWV = device.makeBuffer(length: nodeWBuffer.length, options: .storageModeShared)!

// Utils
let t_vector = TimestepEmbedding(t: 0.5, dim: hiddenDim)
let tEmbBuffer = device.makeBuffer(bytes: t_vector, length: hiddenDim * 4, options: .storageModeShared)!
let cogSumBuffer = device.makeBuffer(length: 3 * 4, options: .storageModeShared)!
let globalNormSqBuffer = device.makeBuffer(length: 4, options: .storageModeShared)!
let lossBufferTrain = device.makeBuffer(length: 4, options: .storageModeShared)!
let lossBufferVal = device.makeBuffer(length: 4, options: .storageModeShared)!

// --- INITIALIZATION & DATA RESET ---

[gradWeightsBuffer, gradBiasBuffer, gradNodeWBuffer, gradNodeBBuffer, gradInputBuffer, gradMsgBuffer, gradHBuffer, gradPosBuffer,
 weightsM, weightsV, nodeWM, nodeWV, globalNormSqBuffer, lossBufferTrain, lossBufferVal, cogSumBuffer].forEach { ZeroInit($0) }

KaimingInit(embedTableBuffer, count: 10 * hiddenDim, fanIn: 10)
KaimingInit(weightsBuffer, count: hiddenDim * (2 * hiddenDim + 1), fanIn: (2 * hiddenDim + 1))
KaimingInit(nodeWBuffer, count: hiddenDim * 2 * hiddenDim, fanIn: 2 * hiddenDim)

// Stabilizer: Start with zero coordinate movement
ZeroInit(coordWeightBuffer)
ZeroInit(biasBuffer); ZeroInit(nodeBBuffer); ZeroInit(coordBiasBuffer)

// Generate target noise
let targetPtr = targetNoiseTrain.contents().bindMemory(to: Float.self, capacity: trainNodeCount * 3)
for i in 0..<(trainNodeCount * 3) { targetPtr[i] = Float.random(in: -0.1...0.1) }

// --- PIPELINES ---

let lib = device.makeDefaultLibrary()!
let names = ["embed_atoms", "compute_message", "aggregate_message", "update_coords", "apply_updates", "compute_mse_loss", "backward_node", "backward_coordinate", "backward_message", "compute_grad_norm_sq", "apply_clipping", "apply_adam_update"]
var p = [String: MTLComputePipelineState]()
for n in names { p[n] = try! device.makeComputePipelineState(function: lib.makeFunction(name: n)!) }

// --- TRAINING LOOP ---

var timestep: UInt32 = 1
let totalEpochs = 100
var lr: Float = 1e-4

for epoch in 1...totalEpochs {
    let cb = commandQueue.makeCommandBuffer()!
    var hDim = UInt32(hiddenDim)
    
    // RESET BUFFERS
    let resetBlit = cb.makeBlitCommandEncoder()!
    [gradWeightsBuffer, gradBiasBuffer, gradNodeWBuffer, gradNodeBBuffer, gradHBuffer, gradMsgBuffer, gradPosBuffer, hBufferTrain, aggBufferTrain, posUpdateTrain].forEach {
        resetBlit.fill(buffer: $0, range: 0..<$0.length, value: 0)
    }
    resetBlit.endEncoding()

    let enc = cb.makeComputeCommandEncoder()!
    
    // 1. FORWARD PASS
    enc.setComputePipelineState(p["embed_atoms"]!)
    enc.setBuffer(loader.nodeBuffer!, offset: 0, index: 0); enc.setBuffer(embedTableBuffer, offset: 0, index: 1)
    enc.setBuffer(hBufferTrain, offset: 0, index: 2); enc.setBytes(&hDim, length: 4, index: 3)
    enc.dispatchThreads(MTLSize(width: trainNodeCount, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    
    for _ in 0..<4 {
        enc.setComputePipelineState(p["compute_message"]!)
        enc.setBuffer(loader.nodeBuffer!, offset: 0, index: 0); enc.setBuffer(hBufferTrain, offset: 0, index: 1)
        enc.setBuffer(loader.edgeBuffer!, offset: 0, index: 2); enc.setBuffer(weightsBuffer, offset: 0, index: 3)
        enc.setBuffer(biasBuffer, offset: 0, index: 4); enc.setBuffer(msgBufferTrain, offset: 0, index: 5)
        enc.setBytes(&hDim, length: 4, index: 6); enc.setBuffer(msgInputBuffer, offset: 0, index: 7); enc.setBuffer(preActivBuffer, offset: 0, index: 8)
        enc.dispatchThreads(MTLSize(width: trainEdgeCount, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        
        enc.setComputePipelineState(p["aggregate_message"]!)
        enc.setBuffer(msgBufferTrain, offset: 0, index: 0); enc.setBuffer(loader.edgeBuffer!, offset: 0, index: 1)
        enc.setBuffer(aggBufferTrain, offset: 0, index: 2); enc.setBytes(&hDim, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: trainEdgeCount, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        
        enc.setComputePipelineState(p["update_coords"]!)
        enc.setBuffer(loader.nodeBuffer!, offset: 0, index: 0); enc.setBuffer(msgBufferTrain, offset: 0, index: 1)
        enc.setBuffer(loader.edgeBuffer!, offset: 0, index: 2); enc.setBuffer(coordWeightBuffer, offset: 0, index: 3)
        enc.setBuffer(coordBiasBuffer, offset: 0, index: 4); enc.setBuffer(posUpdateTrain, offset: 0, index: 5)
        enc.setBytes(&hDim, length: 4, index: 6)
        enc.dispatchThreads(MTLSize(width: trainEdgeCount, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        
        enc.setComputePipelineState(p["apply_updates"]!)
        enc.setBuffer(loader.nodeBuffer!, offset: 0, index: 0); enc.setBuffer(hBufferTrain, offset: 0, index: 1)
        enc.setBuffer(posUpdateTrain, offset: 0, index: 2); enc.setBuffer(aggBufferTrain, offset: 0, index: 3)
        enc.setBuffer(nodeWBuffer, offset: 0, index: 4); enc.setBuffer(nodeBBuffer, offset: 0, index: 5)
        enc.setBytes(&hDim, length: 4, index: 6); enc.setBuffer(nodeActivBuffer, offset: 0, index: 7); enc.setBuffer(preActivNodeBuffer, offset: 0, index: 8)
        enc.dispatchThreads(MTLSize(width: trainNodeCount, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    }
    
    // 2. LOSS
    enc.setComputePipelineState(p["compute_mse_loss"]!)
    enc.setBuffer(posUpdateTrain, offset: 0, index: 0); enc.setBuffer(targetNoiseTrain, offset: 0, index: 1); enc.setBuffer(lossBufferTrain, offset: 0, index: 2)
    var trainElems = UInt32(trainNodeCount * 3); enc.setBytes(&trainElems, length: 4, index: 3)
    enc.dispatchThreads(MTLSize(width: Int(trainElems), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    
    // 3. BACKWARD PASS (Full Chain)
    enc.setComputePipelineState(p["backward_node"]!)
    enc.setBuffer(gradHBuffer, offset: 0, index: 0); enc.setBuffer(nodeWBuffer, offset: 0, index: 1)
    enc.setBuffer(nodeActivBuffer, offset: 0, index: 2); enc.setBuffer(preActivNodeBuffer, offset: 0, index: 3)
    enc.setBuffer(gradNodeWBuffer, offset: 0, index: 4); enc.setBuffer(gradNodeBBuffer, offset: 0, index: 5)
    enc.setBuffer(gradHBuffer, offset: 0, index: 6); enc.setBytes(&hDim, length: 4, index: 7)
    enc.dispatchThreads(MTLSize(width: trainNodeCount, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

    enc.setComputePipelineState(p["backward_coordinate"]!)
    enc.setBuffer(gradPosBuffer, offset: 0, index: 0); enc.setBuffer(coordWeightBuffer, offset: 0, index: 1)
    enc.setBuffer(msgBufferTrain, offset: 0, index: 2); enc.setBuffer(loader.edgeBuffer!, offset: 0, index: 3)
    enc.setBuffer(gradWeightsBuffer, offset: 0, index: 4); enc.setBuffer(gradMsgBuffer, offset: 0, index: 5)
    enc.setBytes(&hDim, length: 4, index: 6); enc.dispatchThreads(MTLSize(width: trainEdgeCount, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

    enc.setComputePipelineState(p["backward_message"]!)
    enc.setBuffer(gradMsgBuffer, offset: 0, index: 0); enc.setBuffer(weightsBuffer, offset: 0, index: 1)
    enc.setBuffer(msgInputBuffer, offset: 0, index: 2); enc.setBuffer(preActivBuffer, offset: 0, index: 3)
    enc.setBuffer(gradWeightsBuffer, offset: 0, index: 4); enc.setBuffer(gradInputBuffer, offset: 0, index: 5)
    enc.setBytes(&hDim, length: 4, index: 6); enc.dispatchThreads(MTLSize(width: trainEdgeCount, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

    // 4. CLIPPING & ADAM
    enc.setComputePipelineState(p["compute_grad_norm_sq"]!)
    enc.setBuffer(gradWeightsBuffer, offset: 0, index: 0); enc.setBuffer(globalNormSqBuffer, offset: 0, index: 1)
    var totW = UInt32(weightsBuffer.length / 4); enc.setBytes(&totW, length: 4, index: 2)
    enc.dispatchThreads(MTLSize(width: Int(totW), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

    enc.setComputePipelineState(p["apply_clipping"]!)
    enc.setBuffer(gradWeightsBuffer, offset: 0, index: 0); enc.setBuffer(globalNormSqBuffer, offset: 0, index: 1)
    var maxN: Float = 0.1; enc.setBytes(&maxN, length: 4, index: 2); enc.setBytes(&totW, length: 4, index: 3)
    enc.dispatchThreads(MTLSize(width: Int(totW), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))

    enc.setComputePipelineState(p["apply_adam_update"]!)
    enc.setBuffer(weightsBuffer, offset: 0, index: 0); enc.setBuffer(weightsM, offset: 0, index: 1); enc.setBuffer(weightsV, offset: 0, index: 2)
    enc.setBuffer(gradWeightsBuffer, offset: 0, index: 3); enc.setBytes(&lr, length: 4, index: 4)
    var b1: Float = 0.9; var b2: Float = 0.999; var eps: Float = 1e-8; var t = timestep
    enc.setBytes(&b1, length: 4, index: 5); enc.setBytes(&b2, length: 4, index: 6); enc.setBytes(&eps, length: 4, index: 7); enc.setBytes(&t, length: 4, index: 8)
    enc.dispatchThreads(MTLSize(width: Int(totW), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    
    enc.endEncoding()
    
    cb.addCompletedHandler { _ in
        let mse = lossBufferTrain.contents().bindMemory(to: Float.self, capacity: 1).pointee / Float(trainNodeCount * 3)
        print("Epoch \(epoch) | Train MSE: \(mse)")
    }
    cb.commit(); cb.waitUntilCompleted(); timestep += 1
}

print("Saving Weights...")
SaveModelWeights(buffers: [("weights_msg", weightsBuffer), ("weights_node", nodeWBuffer)], path: datapath)
