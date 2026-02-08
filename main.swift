//
//  main.swift
//  MSLGraphDiffusion
//
//  Created by Raiden Makoto on 2026-02-07.
//

import Metal
import Foundation

let sectionBreak = String(repeating: "=", count: 50)

// --- CONFIGURATION ---
let hiddenDim = 128
let numLayers = 3
let numTypes = 10

// --- BUFFER STORAGE ---
var weights: [String: MTLBuffer] = [:]

guard let device = MTLCreateSystemDefaultDevice() else { fatalError("Metal not supported") }

// --- UTILITIES ---
func ZeroInit(_ buffer: MTLBuffer) {
    // Sets Buffer contents to 0
    memset(buffer.contents(), 0, buffer.length)
}


func loadAndVerify(_ name: String, rows: Int, cols: Int, path: String) {
    let expectedBytes = rows * cols * 4 // float32 is 4 bytes
    let buffer = device.makeBuffer(length: expectedBytes, options: .storageModeShared)!
    ZeroInit(buffer) // initialize buffer to 0
    
    // Explicitly targeting the /weights subfolder
    let fileURL = URL(fileURLWithPath: path)
        .appendingPathComponent("weights")
        .appendingPathComponent("\(name).bin")
    
    do {
        let data = try Data(contentsOf: fileURL)
        if data.count == expectedBytes {
            buffer.contents().copyMemory(from: (data as NSData).bytes, byteCount: data.count)
            weights[name] = buffer
            print("VERIFIED: \(name).bin (\(data.count) bytes)")
        } else if data.count < expectedBytes {
            print("PARTIAL LOAD: \(data.count)/\(expectedBytes) loaded. Padding with zeros")
        } else {
            print("SIZE MISMATCH: \(name) - File: \(data.count), Buffer Needs: \(expectedBytes)")
        }
    } catch {
        print("NOT FOUND: \(name).bin at \(fileURL.path)")
    }
}

// --- EXECUTION CHUNK ---

// 1. Get the local path
let datapath = URL(fileURLWithPath: #filePath).deletingLastPathComponent().path
print(sectionBreak)
print("LOADING MODEL WEIGHTS")

loadAndVerify("embedding.weight", rows: numTypes, cols: hiddenDim, path: datapath)

// 3. Timestep MLP (2-stage)
loadAndVerify("timestep_mlp.0.weight", rows: hiddenDim, cols: hiddenDim, path: datapath)
loadAndVerify("timestep_mlp.0.bias",   rows: 1,         cols: hiddenDim, path: datapath)
loadAndVerify("timestep_mlp.2.weight", rows: hiddenDim, cols: hiddenDim, path: datapath)
loadAndVerify("timestep_mlp.2.bias",   rows: 1,         cols: hiddenDim, path: datapath)

// 4. Recursive Layers (0-3)
for i in 0..<numLayers {
    print("Layer \(i):")
    // Message MLP
    loadAndVerify("layers.\(i).message_mlp.0.weight", rows: hiddenDim, cols: 2 * hiddenDim + 1, path: datapath)
    loadAndVerify("layers.\(i).message_mlp.0.bias",   rows: 1,         cols: hiddenDim, path: datapath)
    loadAndVerify("layers.\(i).message_mlp.2.weight", rows: hiddenDim, cols: hiddenDim, path: datapath)
    loadAndVerify("layers.\(i).message_mlp.2.bias",   rows: 1,         cols: hiddenDim, path: datapath)
    
    // Coordination MLP
    loadAndVerify("layers.\(i).coord_mlp.0.weight", rows: hiddenDim, cols: hiddenDim, path: datapath)
    loadAndVerify("layers.\(i).coord_mlp.0.bias",   rows: 1,         cols: hiddenDim, path: datapath)
    loadAndVerify("layers.\(i).coord_mlp.2.weight", rows: 1,         cols: hiddenDim, path: datapath)
    loadAndVerify("layers.\(i).coord_mlp.2.bias",   rows: 1,         cols: 1,         path: datapath)
    
    // Node MLP
    loadAndVerify("layers.\(i).node_mlp.0.weight", rows: hiddenDim, cols: 2 * hiddenDim, path: datapath)
    loadAndVerify("layers.\(i).node_mlp.0.bias",   rows: 1,         cols: hiddenDim, path: datapath)
    loadAndVerify("layers.\(i).node_mlp.2.weight", rows: hiddenDim, cols: hiddenDim, path: datapath)
    loadAndVerify("layers.\(i).node_mlp.2.bias",   rows: 1,         cols: hiddenDim, path: datapath)
}

print("Chunk Verification: Loaded \(weights.count) total weight buffers.")
print(sectionBreak)
