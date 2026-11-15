package maps

import "base:runtime"
import "core:math/rand"
import cv "../odin-opencv"

OBSTACLE :: f32(0)
WALKABLE :: f32(1)

cellular_automata :: proc(dims: [2]int, seed: u64, obstacle_rate:f32=0.45, relaxation_steps := 5) -> (g: cv.Mat(f32), err: runtime.Allocator_Error) #optional_allocator_error {
    runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
    r := rand.create(u64(seed))
    gen := runtime.default_random_generator(&r)
    g = cv.mat_new(f32, dims[1], dims[0]) or_return
    for x in 0..<dims[0] {
        for y in 0..<dims[1] {
            noise := rand.float32(gen)
            if noise >= obstacle_rate {
                cv.at(&g, x, y)^ = WALKABLE
            }
        }
    }

    tmp := cv.mat_new(f32, dims[1], dims[0], context.temp_allocator) or_return
    
    src := &g
    dst := &tmp

    for _ in 0..<relaxation_steps {
        for x in 0..<dims[0] {
            for y in 0..<dims[1] {
                neighbors := kernel_sum(src, x, y)
                current   := cv.at(src, x, y)^
                
                if neighbors > 4 {
                    cv.at(dst, x, y)^ = WALKABLE
                }
                else if neighbors < 4 {
                    cv.at(dst, x, y)^ = OBSTACLE
                } 
                else {
                    cv.at(dst, x, y)^ = current 
                }
            }
        }
        src, dst = dst, src
    }

    if src != &g {
        n := int(g.rows * g.cols)
        copy(g.data[:n], src.data[:n])
    }

    return
}

@(private)
kernel_sum :: proc(g: ^cv.Mat(f32), x, y: int, radious := 1) -> f32 {
    sum: f32
    for i in -radious..=radious {
        for j in -radious..=radious {
            if i == 0 && j == 0 do continue
            nx := x + i
            ny := y + j
            if nx >= 0 && nx < int(g.cols) && ny >= 0 && ny < int(g.rows) {
                sum += cv.at(g, nx, ny)^
            }
        }
    }
    return sum
}
