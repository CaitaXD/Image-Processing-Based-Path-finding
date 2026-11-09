package main

import "core:fmt"
import "maps"
import "grid"

SEED :: 42

main :: proc() {
    caverns()
}


caverns :: proc() {
    c := maps.cellular_automata({40, 80}, SEED)
    for i in 0..<c.width {
        for j in 0..<c.height {
            if grid.at(c, i, j)^ == 1 {
            
            }
            else {
            
            }
        }
        fmt.println()
    }
}
