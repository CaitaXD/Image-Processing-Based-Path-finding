package maps

import "base:runtime"
import "core:math/rand"
import "../grid"

OBSTACLE :: u8(0)
WALKABLE :: u8(1)

cellular_automata :: proc(dims: [2]int, seed: u64, obstacle_rate:f32=0.45, relaxation_steps := 5) -> (g: grid.Grid2D(u8), err: runtime.Allocator_Error) #optional_allocator_error {
    runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
    r := rand.create(u64(seed))
    gen := runtime.default_random_generator(&r)
    g = grid.make(u8, dims[0], dims[1]) or_return
    for i in 0..<dims[0] {
        for j in 0..<dims[1] {
            noise := rand.float32(gen)
            if noise >= obstacle_rate {
                grid.colmaj(g, i, j)^ = WALKABLE
            }
        }
    }

    tmp := grid.make(u8, dims[0], dims[1], context.temp_allocator) or_return
    
    src := &g
    dst := &tmp

    for _ in 0..<relaxation_steps {
        for x in 0..<dims[0] {
            for y in 0..<dims[1] {
                neighbors := kernel_sum(src^, x, y)
                current   := grid.at(src^, x, y)^
                
                if neighbors > 4 {
                    grid.at(dst^, x, y)^ = WALKABLE
                }
                else if neighbors < 4 {
                    grid.at(dst^, x, y)^ = OBSTACLE
                } 
                else {
                    grid.at(dst^, x, y)^ = current 
                }
            }
        }
        src, dst = dst, src
    }

    if src != &g do copy(g.data, src.data)

    return
}

@(private)
kernel_sum :: proc(g: grid.Grid2D(u8), x, y: int, radious := 1) -> int {
    sum := 0
    for i in -radious..=radious {
        for j in -radious..=radious {
            if i == 0 && j == 0 do continue
            nx := x + i
            ny := y + j
            if nx >= 0 && nx < g.width && ny >= 0 && ny < g.height {
                sum += int(grid.at(g, nx, ny)^)
            }
        }
    }
    return sum
}
