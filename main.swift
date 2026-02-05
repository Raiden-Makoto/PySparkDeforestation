//
//  main.swift
//  MSLGraphDiffusion
//
//  Created by Raiden Makoto on 2026-02-04.
//

import Metal
import Foundation

guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("Metal is not supported on this device!")
}

print("Engine initialized on: \(device.name)")
let loader = QM9Loader(device: device)
let sourceDirURL = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
let datapath = sourceDirURL.path

do {
    print("Attempting to load the data from \(datapath)...")
    try loader.load(from: datapath)
    if let metaBuffer = loader.graphdataBuffer {
        let metaPtr = metaBuffer.contents().bindMemory(to: GraphData.self, capacity: loader.graphCount)
        let sample = metaPtr.pointee
        print("Data Load Success:")
        print("Total Molecules Loaded: \(loader.graphCount)")
        print("Sample Molecule Stats")
        print(" - Node Data: \(sample.nodeStart)")
        print(" - Atom Count: \(sample.nodeCount)")
        print(" - Edge Count: \(sample.edgeCount)")
    }
}
catch {
    print("Data Load Failed:")
    print("Error \(error.localizedDescription)")
    print("Check if the datapath exists and try again")
}
