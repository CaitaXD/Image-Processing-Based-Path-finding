package units

import "base:runtime"
import "core:math/rand"
import "../grid"
import "../maps"

Unit :: struct {
    pos: [2]f32,
}

Goal :: struct {
    pos: [2]int,
}

sample :: proc(obstacles: grid.Grid2D(u8), n: int, seed: u64, allocator := context.allocator) -> (u: []Unit, err: runtime.Allocator_Error) #optional_allocator_error {
    runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()

    cells := walkable_cells(obstacles, context.temp_allocator) or_return

    picked := min(n, len(cells))
    u = make([]Unit, picked, allocator) or_return

    state := rand.create(seed)
    gen := runtime.default_random_generator(&state)

    for i in 0..<picked {
        j := rand.int_range(i, len(cells), gen)
        cells[i], cells[j] = cells[j], cells[i]
        u[i].pos = {f32(cells[i][0]) + 0.5, f32(cells[i][1]) + 0.5}
    }
    return
}

sample_goal :: proc(obstacles: grid.Grid2D(u8), seed: u64) -> (goal: Goal, ok: bool) {
    count := 0
    for x in 0..<obstacles.width {
        for y in 0..<obstacles.height {
            if grid.at(obstacles, x, y)^ == maps.WALKABLE {
                count += 1
            }
        }
    }
    if count == 0 {
        return {}, false
    }

    state := rand.create(seed)
    gen := runtime.default_random_generator(&state)
    pick := rand.int_range(0, count, gen)

    i := 0
    for x in 0..<obstacles.width {
        for y in 0..<obstacles.height {
            if grid.at(obstacles, x, y)^ == maps.WALKABLE {
                if i == pick {
                    return Goal{pos = {x, y}}, true
                }
                i += 1
            }
        }
    }
    return {}, false
}

@(private)
walkable_cells :: proc(obstacles: grid.Grid2D(u8), allocator := context.allocator) -> (cells: [dynamic][2]int, err: runtime.Allocator_Error) #optional_allocator_error {
    cells = make([dynamic][2]int, 0, obstacles.width*obstacles.height, allocator) or_return
    for x in 0..<obstacles.width {
        for y in 0..<obstacles.height {
            if grid.at(obstacles, x, y)^ == maps.WALKABLE {
                append(&cells, [2]int{x, y})
            }
        }
    }
    return
}
