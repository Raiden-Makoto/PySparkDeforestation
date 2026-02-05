//
//  GraphTypes.swift
//  MSLGraphDiffusion
//
//  Created by Raiden-Makoto on 2026-02-05.
//

import Metal
import Foundation

// avoid using unsigned ints for GPU programming

// matches the [x, y, z, atom_type] 16-byte packing
struct Node{
    let x: Float
    let y: Float
    let z: Float
    let atomType: Float
}

// matches the [row, col] 8-byte packing
struct Edge{
    let row: Int32
    let col: Int32
}

// matches the  [node_start, n_nodes, edge_start, n_edges] 16-byte packing
struct GraphData{
    let nodeStart: Int32
    let nodeCount: Int32
    let edgeStart: Int32
    let edgeCount: Int32
}
