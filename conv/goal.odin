package conv

import "base:runtime"
import "core:math"
import gd "../grid"
import "../maps"

Goal :: struct {
    pos:    [2]f32,
    weight: f32,
}

HeatSolver :: struct {
    grid:            gd.Grid2D(u8),
    field:           gd.Grid2D(f32),
    tmp:             gd.Grid2D(f32),
    rate:            f32,
    decay:           f32,
    source_radius:   f32,
    source_strength: f32,
}

heat :: proc(walls: gd.Grid2D(u8), rate, decay, source_radius, source_strength: f32, allocator := context.allocator) -> (s: HeatSolver, err: runtime.Allocator_Error) #optional_allocator_error {
    s.grid           = walls
    s.rate            = rate
    s.decay           = decay
    s.source_radius   = source_radius
    s.source_strength = source_strength

    s.field, err = gd.make(f32, walls.width, walls.height, allocator)
    if err != nil {
        return
    }

    s.tmp, err = gd.make(f32, walls.width, walls.height, allocator)
    if err != nil {
        delete(s.field.data)
        s.field = {}
        return
    }

    return
}

step :: proc(s: ^HeatSolver, goals: []Goal, substeps: int) {
    if substeps <= 0 {
        return
    }

    for _ in 0..<substeps {
        diffuse(s)

        for goal in goals {
            add_source(&s.tmp, s.grid, goal, s.source_radius, s.source_strength)
        }

        s.field, s.tmp = s.tmp, s.field
    }
}

// Builds the source term once so it can be added every step without recomputing
// the Gaussian. Useful when a broad source is needed for navigation fields.
make_source :: proc(walls: gd.Grid2D(u8), goal: Goal, sigma, strength: f32, allocator := context.allocator) -> (src: gd.Grid2D(f32), err: runtime.Allocator_Error) #optional_allocator_error {
    src = gd.make(f32, walls.width, walls.height, allocator) or_return

    weight := max(goal.weight, f32(0))
    if weight == 0 || strength <= 0 {
        return
    }

    if sigma <= 0 {
        x := int(goal.pos[0])
        y := int(goal.pos[1])
        if x >= 0 && x < src.width && y >= 0 && y < src.height &&
           gd.at(walls, x, y)^ == maps.WALKABLE {
            gd.at(src, x, y)^ = weight * strength
        }
        return
    }

    radius := sigma * 3
    x0 := max(int(goal.pos[0] - radius), 0)
    x1 := min(int(goal.pos[0] + radius) + 1, src.width - 1)
    y0 := max(int(goal.pos[1] - radius), 0)
    y1 := min(int(goal.pos[1] + radius) + 1, src.height - 1)
    if x0 > x1 || y0 > y1 {
        return
    }

    two_sigma_sq := 2 * sigma * sigma

    kernel_sum: f32
    for y in y0..=y1 {
        for x in x0..=x1 {
            if gd.at(walls, x, y)^ != maps.WALKABLE {
                continue
            }
            dx := (f32(x) + 0.5) - goal.pos[0]
            dy := (f32(y) + 0.5) - goal.pos[1]
            kernel_sum += math.exp(-(dx*dx + dy*dy) / two_sigma_sq)
        }
    }
    if kernel_sum <= 0 {
        return
    }

    scale := weight * strength / kernel_sum
    for y in y0..=y1 {
        for x in x0..=x1 {
            if gd.at(walls, x, y)^ != maps.WALKABLE {
                continue
            }
            dx := (f32(x) + 0.5) - goal.pos[0]
            dy := (f32(y) + 0.5) - goal.pos[1]
            gd.at(src, x, y)^ = scale * math.exp(-(dx*dx + dy*dy) / two_sigma_sq)
        }
    }

    return
}

seed_source :: proc(s: ^HeatSolver, source: gd.Grid2D(f32)) {
    assert(source.width == s.field.width && source.height == s.field.height)

    if s.decay > 0 {
        inv_decay := 1 / s.decay
        for i in 0..<len(s.field.data) {
            s.field.data[i] = source.data[i] * inv_decay
        }
    } else {
        copy(s.field.data, source.data)
    }
}

step_with_source :: proc(s: ^HeatSolver, source: gd.Grid2D(f32), substeps: int) {
    if substeps <= 0 {
        return
    }
    assert(source.width == s.field.width && source.height == s.field.height)

    for _ in 0..<substeps {
        diffuse(s)

        for i in 0..<len(s.tmp.data) {
            s.tmp.data[i] += source.data[i]
        }

        s.field, s.tmp = s.tmp, s.field
    }
}

@(private)
diffuse :: proc(s: ^HeatSolver) {
    rate  := clamp(s.rate, 0, 1)
    decay := clamp(s.decay, 0, 1)

    for y in 0..<s.grid.height {
        for x in 0..<s.grid.width {
            if gd.at(s.grid, x, y)^ != maps.WALKABLE {
                gd.at(s.tmp, x, y)^ = 0
                continue
            }

            neighbour_sum: f32
            neighbour_count: int

            if x > 0 && gd.at(s.grid, x-1, y)^ == maps.WALKABLE {
                neighbour_sum += gd.at(s.field, x-1, y)^
                neighbour_count += 1
            }
            if x+1 < s.grid.width && gd.at(s.grid, x+1, y)^ == maps.WALKABLE {
                neighbour_sum += gd.at(s.field, x+1, y)^
                neighbour_count += 1
            }
            if y > 0 && gd.at(s.grid, x, y-1)^ == maps.WALKABLE {
                neighbour_sum += gd.at(s.field, x, y-1)^
                neighbour_count += 1
            }
            if y+1 < s.grid.height && gd.at(s.grid, x, y+1)^ == maps.WALKABLE {
                neighbour_sum += gd.at(s.field, x, y+1)^
                neighbour_count += 1
            }

            value := gd.at(s.field, x, y)^
            if neighbour_count > 0 {
                average := neighbour_sum / f32(neighbour_count)
                value += rate * (average - value)
            }
            value *= 1 - decay
            gd.at(s.tmp, x, y)^ = max(value, f32(0))
        }
    }
}

@(private)
add_source :: proc(dst: ^gd.Grid2D(f32), walls: gd.Grid2D(u8), goal: Goal, sigma, strength: f32) {
    weight := max(goal.weight, f32(0))
    if weight == 0 || strength <= 0 {
        return
    }

    if sigma <= 0 {
        x := int(goal.pos[0])
        y := int(goal.pos[1])
        if x >= 0 && x < dst.width && y >= 0 && y < dst.height &&
           gd.at(walls, x, y)^ == maps.WALKABLE {
            gd.at(dst^, x, y)^ += weight * strength
        }
        return
    }

    radius := sigma * 3
    x0 := max(int(goal.pos[0] - radius), 0)
    x1 := min(int(goal.pos[0] + radius) + 1, dst.width - 1)
    y0 := max(int(goal.pos[1] - radius), 0)
    y1 := min(int(goal.pos[1] + radius) + 1, dst.height - 1)
    if x0 > x1 || y0 > y1 {
        return
    }

    two_sigma_sq := 2 * sigma * sigma

    kernel_sum: f32
    for y in y0..=y1 {
        for x in x0..=x1 {
            if gd.at(walls, x, y)^ != maps.WALKABLE {
                continue
            }
            dx := (f32(x) + 0.5) - goal.pos[0]
            dy := (f32(y) + 0.5) - goal.pos[1]
            kernel_sum += math.exp(-(dx*dx + dy*dy) / two_sigma_sq)
        }
    }
    if kernel_sum <= 0 {
        return
    }

    scale := weight * strength / kernel_sum
    for y in y0..=y1 {
        for x in x0..=x1 {
            if gd.at(walls, x, y)^ != maps.WALKABLE {
                continue
            }
            dx := (f32(x) + 0.5) - goal.pos[0]
            dy := (f32(y) + 0.5) - goal.pos[1]
            gd.at(dst^, x, y)^ += scale * math.exp(-(dx*dx + dy*dy) / two_sigma_sq)
        }
    }
}
