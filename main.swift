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
    precondition(count > 0 && fanIn > 0, "count and fanIn must be positive")
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
    let fileManager = FileManager.default
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
    print("Attempting to load data from \(datapath)...")
    try loader.load(from: datapath)
} catch {
    fatalError("Data Load Failed: \(error.localizedDescription)")
}

guard let nodeBuffer = loader.nodeBuffer else {
    fatalError("loader.nodeBuffer is nil after loading data")
}
let firstGraphMetadata = loader.graphdataBuffer!.contents().bindMemory(to: GraphData.self, capacity: 1).pointee
let activeEdgeCount = Int(firstGraphMetadata.edgeCount)
let activeNodeCount = Int(firstGraphMetadata.nodeCount)
let hiddenDim = 64

// --- BUFFER ALLOCATIONS ---

let embedTableBuffer = device.makeBuffer(length: 10 * hiddenDim * MemoryLayout<Float>.size, options: .storageModeShared)!
let hBuffer = device.makeBuffer(length: activeNodeCount * hiddenDim * MemoryLayout<Float>.size, options: .storageModeShared)!
let t_vector = TimestepEmbedding(t: 0.5, dim: hiddenDim)
let tEmbBuffer = device.makeBuffer(bytes: t_vector, length: hiddenDim * MemoryLayout<Float>.size, options: .storageModeShared)!

let weightsBuffer = device.makeBuffer(length: hiddenDim * (2 * hiddenDim + 1) * MemoryLayout<Float>.size, options: .storageModeShared)!
let biasBuffer = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.size, options: .storageModeShared)!
let msgBuffer = device.makeBuffer(length: activeEdgeCount * hiddenDim * MemoryLayout<Float>.size, options: .storageModeShared)!

let aggBuffer = device.makeBuffer(length: activeNodeCount * hiddenDim * MemoryLayout<Float>.size, options: .storageModeShared)!
let coordWeightBuffer = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.size, options: .storageModeShared)!
let coordBiasBuffer = device.makeBuffer(length: MemoryLayout<Float>.size, options: .storageModeShared)!
let posUpdateBuffer = device.makeBuffer(length: activeNodeCount * 3 * MemoryLayout<Float>.size, options: .storageModeShared)!

let nodeWBuffer = device.makeBuffer(length: hiddenDim * (2 * hiddenDim) * MemoryLayout<Float>.size, options: .storageModeShared)!
let nodeBBuffer = device.makeBuffer(length: hiddenDim * MemoryLayout<Float>.size, options: .storageModeShared)!
let cogSumBuffer = device.makeBuffer(length: 3 * MemoryLayout<Float>.size, options: .storageModeShared)!

// --- BACKPROPAGATION & OPTIMIZATION STORAGE ---

let msgInputBuffer = device.makeBuffer(length: activeEdgeCount * (2 * hiddenDim + 1) * MemoryLayout<Float>.size, options: .storageModeShared)!
let preActivBuffer = device.makeBuffer(length: activeEdgeCount * hiddenDim * MemoryLayout<Float>.size, options: .storageModeShared)!
let nodeActivBuffer = device.makeBuffer(length: activeNodeCount * 2 * hiddenDim * MemoryLayout<Float>.size, options: .storageModeShared)!
let preActivNodeBuffer = device.makeBuffer(length: activeNodeCount * hiddenDim * MemoryLayout<Float>.size, options: .storageModeShared)!

let gradWeightsBuffer = device.makeBuffer(length: weightsBuffer.length, options: .storageModeShared)!
let gradBiasBuffer = device.makeBuffer(length: biasBuffer.length, options: .storageModeShared)!
let gradNodeWBuffer = device.makeBuffer(length: nodeWBuffer.length, options: .storageModeShared)!
let gradNodeBBuffer = device.makeBuffer(length: nodeBBuffer.length, options: .storageModeShared)!
let gradInputBuffer = device.makeBuffer(length: activeEdgeCount * (2 * hiddenDim + 1) * MemoryLayout<Float>.size, options: .storageModeShared)!
let gradMsgBuffer = device.makeBuffer(length: msgBuffer.length, options: .storageModeShared)!
let gradHBuffer = device.makeBuffer(length: hBuffer.length, options: .storageModeShared)!
let gradPosBuffer = device.makeBuffer(length: activeNodeCount * 3 * MemoryLayout<Float>.size, options: .storageModeShared)!

let weightsM = device.makeBuffer(length: weightsBuffer.length, options: .storageModeShared)!
let weightsV = device.makeBuffer(length: weightsBuffer.length, options: .storageModeShared)!
let nodeWM = device.makeBuffer(length: nodeWBuffer.length, options: .storageModeShared)!
let nodeWV = device.makeBuffer(length: nodeWBuffer.length, options: .storageModeShared)!

let globalNormSqBuffer = device.makeBuffer(length: MemoryLayout<Float>.size, options: .storageModeShared)!
let lossBuffer = device.makeBuffer(length: MemoryLayout<Float>.size, options: .storageModeShared)!
let targetNoiseBuffer = device.makeBuffer(length: activeNodeCount * 3 * MemoryLayout<Float>.size, options: .storageModeShared)!

// Reset all training/temp buffers
[cogSumBuffer, msgInputBuffer, preActivBuffer, nodeActivBuffer, preActivNodeBuffer,
 gradWeightsBuffer, gradBiasBuffer, gradNodeWBuffer, gradNodeBBuffer, gradInputBuffer,
 gradMsgBuffer, gradHBuffer, gradPosBuffer, weightsM, weightsV, nodeWM, nodeWV,
 globalNormSqBuffer, lossBuffer].forEach { ZeroInit($0) }

// --- WEIGHT INITIALIZATION ---

KaimingInit(embedTableBuffer, count: 10 * hiddenDim, fanIn: 10)
KaimingInit(weightsBuffer, count: hiddenDim * (2 * hiddenDim + 1), fanIn: (2 * hiddenDim + 1))
ZeroInit(biasBuffer)
KaimingInit(coordWeightBuffer, count: hiddenDim, fanIn: hiddenDim)
KaimingInit(nodeWBuffer, count: hiddenDim * 2 * hiddenDim, fanIn: 2 * hiddenDim)
ZeroInit(nodeBBuffer)

// --- PIPELINES ---

let lib = device.makeDefaultLibrary()!
let embedPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "embed_atoms")!)
let timePipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "inject_timestep")!)
let msgPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "compute_message")!)
let aggPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "aggregate_message")!)
let coordPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "update_coords")!)
let applyPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "apply_updates")!)
let cogPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "compute_cog")!)
let normPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "apply_cog_normalization")!)

let lossPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "compute_mse_loss")!)
let bwdNodePipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "backward_node")!)
let bwdMsgPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "backward_message")!)
let bwdCoordPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "backward_coordinate")!)
let gradNormPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "compute_grad_norm_sq")!)
let clipPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "apply_clipping")!)
let adamPipeline = try device.makeComputePipelineState(function: lib.makeFunction(name: "apply_adam_update")!)

// --- PRE-TRAINING NOISE INJECTION ---

let targetPtr = targetNoiseBuffer.contents().bindMemory(to: Float.self, capacity: activeNodeCount * 3)
for i in 0..<(activeNodeCount * 3) {
    let u1 = Float.random(in: 0...1), u2 = Float.random(in: 0...1)
    let mag = sqrt(-2.0 * log(u1))
    targetPtr[i] = mag * cos(2.0 * Float.pi * u2)
}

let nodePtrStart = nodeBuffer.contents().bindMemory(to: Node.self, capacity: activeNodeCount)
for i in 0..<activeNodeCount {
    nodePtrStart[i].pos.x += targetPtr[i*3]
    nodePtrStart[i].pos.y += targetPtr[i*3+1]
    nodePtrStart[i].pos.z += targetPtr[i*3+2]
}

// --- EXECUTION ---

let commandBuffer = commandQueue.makeCommandBuffer()!

// 1. FORWARD PASS
let embedEncoder = commandBuffer.makeComputeCommandEncoder()!
embedEncoder.setComputePipelineState(embedPipeline)
embedEncoder.setBuffer(nodeBuffer, offset: 0, index: 0); embedEncoder.setBuffer(embedTableBuffer, offset: 0, index: 1)
embedEncoder.setBuffer(hBuffer, offset: 0, index: 2); var hDimVal = UInt32(hiddenDim)
embedEncoder.setBytes(&hDimVal, length: MemoryLayout<UInt32>.size, index: 3)
embedEncoder.dispatchThreads(MTLSize(width: activeNodeCount, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
embedEncoder.endEncoding()

let timeEncoder = commandBuffer.makeComputeCommandEncoder()!
timeEncoder.setComputePipelineState(timePipeline); timeEncoder.setBuffer(hBuffer, offset: 0, index: 0)
timeEncoder.setBuffer(tEmbBuffer, offset: 0, index: 1); timeEncoder.setBytes(&hDimVal, length: MemoryLayout<UInt32>.size, index: 2)
timeEncoder.dispatchThreads(MTLSize(width: activeNodeCount, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
timeEncoder.endEncoding()

for _ in 0..<4 {
    let msgEncoder = commandBuffer.makeComputeCommandEncoder()!
    msgEncoder.setComputePipelineState(msgPipeline)
    msgEncoder.setBuffer(nodeBuffer, offset: 0, index: 0); msgEncoder.setBuffer(hBuffer, offset: 0, index: 1)
    msgEncoder.setBuffer(loader.edgeBuffer, offset: 0, index: 2); msgEncoder.setBuffer(weightsBuffer, offset: 0, index: 3)
    msgEncoder.setBuffer(biasBuffer, offset: 0, index: 4); msgEncoder.setBuffer(msgBuffer, offset: 0, index: 5)
    msgEncoder.setBytes(&hDimVal, length: MemoryLayout<UInt32>.size, index: 6)
    msgEncoder.setBuffer(msgInputBuffer, offset: 0, index: 7); msgEncoder.setBuffer(preActivBuffer, offset: 0, index: 8)
    msgEncoder.dispatchThreads(MTLSize(width: activeEdgeCount, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    msgEncoder.endEncoding()

    let blit = commandBuffer.makeBlitCommandEncoder()!
    blit.fill(buffer: aggBuffer, range: 0..<aggBuffer.length, value: 0)
    blit.endEncoding()

    let aggEncoder = commandBuffer.makeComputeCommandEncoder()!
    aggEncoder.setComputePipelineState(aggPipeline); aggEncoder.setBuffer(msgBuffer, offset: 0, index: 0)
    aggEncoder.setBuffer(loader.edgeBuffer, offset: 0, index: 1); aggEncoder.setBuffer(aggBuffer, offset: 0, index: 2)
    aggEncoder.setBytes(&hDimVal, length: MemoryLayout<UInt32>.size, index: 3)
    aggEncoder.dispatchThreads(MTLSize(width: activeEdgeCount, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    aggEncoder.endEncoding()

    let coordEncoder = commandBuffer.makeComputeCommandEncoder()!
    coordEncoder.setComputePipelineState(coordPipeline)
    coordEncoder.setBuffer(nodeBuffer, offset: 0, index: 0); coordEncoder.setBuffer(msgBuffer, offset: 0, index: 1)
    coordEncoder.setBuffer(loader.edgeBuffer, offset: 0, index: 2); coordEncoder.setBuffer(coordWeightBuffer, offset: 0, index: 3)
    coordEncoder.setBuffer(coordBiasBuffer, offset: 0, index: 4); coordEncoder.setBuffer(posUpdateBuffer, offset: 0, index: 5)
    coordEncoder.setBytes(&hDimVal, length: MemoryLayout<UInt32>.size, index: 6)
    coordEncoder.dispatchThreads(MTLSize(width: activeEdgeCount, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    coordEncoder.endEncoding()

    let applyEncoder = commandBuffer.makeComputeCommandEncoder()!
    applyEncoder.setComputePipelineState(applyPipeline); applyEncoder.setBuffer(nodeBuffer, offset: 0, index: 0)
    applyEncoder.setBuffer(hBuffer, offset: 0, index: 1); applyEncoder.setBuffer(posUpdateBuffer, offset: 0, index: 2)
    applyEncoder.setBuffer(aggBuffer, offset: 0, index: 3); applyEncoder.setBuffer(nodeWBuffer, offset: 0, index: 4)
    applyEncoder.setBuffer(nodeBBuffer, offset: 0, index: 5); applyEncoder.setBytes(&hDimVal, length: MemoryLayout<UInt32>.size, index: 6)
    applyEncoder.setBuffer(nodeActivBuffer, offset: 0, index: 7); applyEncoder.setBuffer(preActivNodeBuffer, offset: 0, index: 8)
    applyEncoder.dispatchThreads(MTLSize(width: activeNodeCount, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    applyEncoder.endEncoding()
    
    let blitCog = commandBuffer.makeBlitCommandEncoder()!
    blitCog.fill(buffer: cogSumBuffer, range: 0..<cogSumBuffer.length, value: 0)
    blitCog.endEncoding()

    let cogEncoder = commandBuffer.makeComputeCommandEncoder()!
    cogEncoder.setComputePipelineState(cogPipeline); cogEncoder.setBuffer(loader.nodeBuffer, offset: 0, index: 0)
    cogEncoder.setBuffer(cogSumBuffer, offset: 0, index: 1)
    cogEncoder.dispatchThreads(MTLSize(width: activeNodeCount, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    cogEncoder.endEncoding()

    let normEncoder = commandBuffer.makeComputeCommandEncoder()!
    normEncoder.setComputePipelineState(normPipeline); normEncoder.setBuffer(loader.nodeBuffer, offset: 0, index: 0)
    normEncoder.setBuffer(cogSumBuffer, offset: 0, index: 1); var nodeCountU32 = UInt32(activeNodeCount)
    normEncoder.setBytes(&nodeCountU32, length: MemoryLayout<UInt32>.size, index: 2)
    normEncoder.dispatchThreads(MTLSize(width: activeNodeCount, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
    normEncoder.endEncoding()
}

// 2. LOSS
let lossEncoder = commandBuffer.makeComputeCommandEncoder()!
lossEncoder.setComputePipelineState(lossPipeline)
lossEncoder.setBuffer(posUpdateBuffer, offset: 0, index: 0); lossEncoder.setBuffer(targetNoiseBuffer, offset: 0, index: 1)
lossEncoder.setBuffer(lossBuffer, offset: 0, index: 2); var totalElements = UInt32(activeNodeCount * 3)
lossEncoder.setBytes(&totalElements, length: MemoryLayout<UInt32>.size, index: 3)
lossEncoder.dispatchThreads(MTLSize(width: Int(totalElements), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
lossEncoder.endEncoding()

// 3. BACKWARD
let bwdNodeEncoder = commandBuffer.makeComputeCommandEncoder()!
bwdNodeEncoder.setComputePipelineState(bwdNodePipeline)
bwdNodeEncoder.setBuffer(gradHBuffer, offset: 0, index: 0); bwdNodeEncoder.setBuffer(nodeWBuffer, offset: 0, index: 1)
bwdNodeEncoder.setBuffer(nodeActivBuffer, offset: 0, index: 2); bwdNodeEncoder.setBuffer(preActivNodeBuffer, offset: 0, index: 3)
bwdNodeEncoder.setBuffer(gradNodeWBuffer, offset: 0, index: 4); bwdNodeEncoder.setBuffer(gradNodeBBuffer, offset: 0, index: 5)
bwdNodeEncoder.setBuffer(gradHBuffer, offset: 0, index: 6); bwdNodeEncoder.setBytes(&hDimVal, length: MemoryLayout<UInt32>.size, index: 7)
bwdNodeEncoder.dispatchThreads(MTLSize(width: activeNodeCount, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
bwdNodeEncoder.endEncoding()

let bwdCoordEncoder = commandBuffer.makeComputeCommandEncoder()!
bwdCoordEncoder.setComputePipelineState(bwdCoordPipeline)
bwdCoordEncoder.setBuffer(gradPosBuffer, offset: 0, index: 0); bwdCoordEncoder.setBuffer(coordWeightBuffer, offset: 0, index: 1)
bwdCoordEncoder.setBuffer(msgBuffer, offset: 0, index: 2); bwdCoordEncoder.setBuffer(loader.edgeBuffer, offset: 0, index: 3)
bwdCoordEncoder.setBuffer(gradWeightsBuffer, offset: 0, index: 4); bwdCoordEncoder.setBuffer(gradMsgBuffer, offset: 0, index: 5)
bwdCoordEncoder.setBytes(&hDimVal, length: MemoryLayout<UInt32>.size, index: 6)
bwdCoordEncoder.dispatchThreads(MTLSize(width: activeEdgeCount, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
bwdCoordEncoder.endEncoding()

let bwdMsgEncoder = commandBuffer.makeComputeCommandEncoder()!
bwdMsgEncoder.setComputePipelineState(bwdMsgPipeline)
bwdMsgEncoder.setBuffer(gradMsgBuffer, offset: 0, index: 0); bwdMsgEncoder.setBuffer(weightsBuffer, offset: 0, index: 1)
bwdMsgEncoder.setBuffer(msgInputBuffer, offset: 0, index: 2); bwdMsgEncoder.setBuffer(preActivBuffer, offset: 0, index: 3)
bwdMsgEncoder.setBuffer(gradWeightsBuffer, offset: 0, index: 4); bwdMsgEncoder.setBuffer(gradInputBuffer, offset: 0, index: 5)
bwdMsgEncoder.setBytes(&hDimVal, length: MemoryLayout<UInt32>.size, index: 6)
bwdMsgEncoder.dispatchThreads(MTLSize(width: activeEdgeCount, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
bwdMsgEncoder.endEncoding()

// 4. CLIPPING & ADAM
let normBlit = commandBuffer.makeBlitCommandEncoder()!
normBlit.fill(buffer: globalNormSqBuffer, range: 0..<MemoryLayout<Float>.size, value: 0); normBlit.endEncoding()

let normEncoder = commandBuffer.makeComputeCommandEncoder()!
normEncoder.setComputePipelineState(gradNormPipeline); normEncoder.setBuffer(gradWeightsBuffer, offset: 0, index: 0)
normEncoder.setBuffer(globalNormSqBuffer, offset: 0, index: 1); var totalGradElements = UInt32(weightsBuffer.length / MemoryLayout<Float>.size)
normEncoder.setBytes(&totalGradElements, length: MemoryLayout<UInt32>.size, index: 2)
normEncoder.dispatchThreads(MTLSize(width: Int(totalGradElements), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
normEncoder.endEncoding()

let clipEncoder = commandBuffer.makeComputeCommandEncoder()!
clipEncoder.setComputePipelineState(clipPipeline); clipEncoder.setBuffer(gradWeightsBuffer, offset: 0, index: 0)
clipEncoder.setBuffer(globalNormSqBuffer, offset: 0, index: 1); var maxNorm: Float = 1.0
clipEncoder.setBytes(&maxNorm, length: MemoryLayout<Float>.size, index: 2); clipEncoder.setBytes(&totalGradElements, length: MemoryLayout<UInt32>.size, index: 3)
clipEncoder.dispatchThreads(MTLSize(width: Int(totalGradElements), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
clipEncoder.endEncoding()

let adamEncoder = commandBuffer.makeComputeCommandEncoder()!
adamEncoder.setComputePipelineState(adamPipeline); adamEncoder.setBuffer(weightsBuffer, offset: 0, index: 0)
adamEncoder.setBuffer(weightsM, offset: 0, index: 1); adamEncoder.setBuffer(weightsV, offset: 0, index: 2)
adamEncoder.setBuffer(gradWeightsBuffer, offset: 0, index: 3); var lr: Float = 1e-4; var b1: Float = 0.9
var b2: Float = 0.999; var eps: Float = 1e-8; var timestep: UInt32 = 1
adamEncoder.setBytes(&lr, length: MemoryLayout<Float>.size, index: 4); adamEncoder.setBytes(&b1, length: MemoryLayout<Float>.size, index: 5)
adamEncoder.setBytes(&b2, length: MemoryLayout<Float>.size, index: 6); adamEncoder.setBytes(&eps, length: MemoryLayout<Float>.size, index: 7)
adamEncoder.setBytes(&timestep, length: MemoryLayout<UInt32>.size, index: 8)
adamEncoder.dispatchThreads(MTLSize(width: Int(totalGradElements), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
adamEncoder.endEncoding()

commandBuffer.addCompletedHandler { _ in
    let mse = lossBuffer.contents().bindMemory(to: Float.self, capacity: 1).pointee / Float(activeNodeCount * 3)
    print("Training Step Complete - MSE Loss: \(mse)")
}

commandBuffer.commit()
commandBuffer.waitUntilCompleted()

let finalNodes = nodeBuffer.contents().bindMemory(to: Node.self, capacity: activeNodeCount)
print("Final Position Node 0: \(finalNodes[0].pos)")

let weightBuffersToSave = [
    ("weights_msg", weightsBuffer),
    ("bias_msg", biasBuffer),
    ("weights_node", nodeWBuffer),
    ("bias_node", nodeBBuffer),
    ("weights_coord", coordWeightBuffer),
    ("embed_table", embedTableBuffer)
]

print("Saving Model Weights...")
SaveModelWeights(buffers: weightBuffersToSave, path: datapath)
