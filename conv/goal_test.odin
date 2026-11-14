package conv

import "core:testing"
import "../grid"
import "../maps"

@(private)
make_walls :: proc(w, h: int) -> grid.Grid2D(u8) {
    g, err := grid.make(u8, w, h)
    assert(err == .None)
    for i in 0..<len(g.data) {
        g.data[i] = maps.WALKABLE
    }
    return g
}

@(private)
total_heat :: proc(g: grid.Grid2D(f32)) -> f32 {
    total: f32
    for value in g.data {
        total += value
    }
    return total
}

@(test)
sources_add_linearly :: proc(t: ^testing.T) {
    walls := make_walls(48, 48)
    defer delete(walls.data)

    a, err_a := heat(walls, 0.5, 0.01, 3, 1, context.temp_allocator)
    assert(err_a == .None)

    b, err_b := heat(walls, 0.5, 0.01, 3, 1, context.temp_allocator)
    assert(err_b == .None)

    both, err_both := heat(walls, 0.5, 0.01, 3, 1, context.temp_allocator)
    assert(err_both == .None)

    goal_a := Goal{pos = {12.5, 24.5}, weight = 1}
    goal_b := Goal{pos = {36.5, 24.5}, weight = 1}

    for _ in 0..<30 {
        step(&a, []Goal{goal_a}, 1)
        step(&b, []Goal{goal_b}, 1)
        step(&both, []Goal{goal_a, goal_b}, 1)
    }

    for i in 0..<len(both.field.data) {
        expected := a.field.data[i] + b.field.data[i]
        assert(abs(both.field.data[i] - expected) < 1e-4,
            "multiple goals must superpose linearly")
    }
}

@(test)
weights_scale_field :: proc(t: ^testing.T) {
    walls := make_walls(48, 48)
    defer delete(walls.data)

    one, err_one := heat(walls, 0.5, 0.01, 3, 1, context.temp_allocator)
    assert(err_one == .None)

    two, err_two := heat(walls, 0.5, 0.01, 3, 1, context.temp_allocator)
    assert(err_two == .None)

    goal_one := Goal{pos = {24.5, 24.5}, weight = 1}
    goal_two := Goal{pos = {24.5, 24.5}, weight = 2}

    for _ in 0..<40 {
        step(&one, []Goal{goal_one}, 1)
        step(&two, []Goal{goal_two}, 1)
    }

    for i in 0..<len(two.field.data) {
        assert(abs(two.field.data[i] - 2 * one.field.data[i]) < 1e-4,
            "weight must scale the source linearly")
    }
}

@(test)
walls_block_heat :: proc(t: ^testing.T) {
    walls := make_walls(48, 48)
    defer delete(walls.data)

    for y in 0..<walls.height {
        grid.at(walls, 24, y)^ = maps.OBSTACLE
    }

    s, err := heat(walls, 0.5, 0.01, 3, 1, context.temp_allocator)
    assert(err == .None)

    source := Goal{pos = {12.5, 24.5}, weight = 1}
    for _ in 0..<300 {
        step(&s, []Goal{source}, 1)
    }

    for y in 0..<s.field.height {
        for x in 25..<s.field.width {
            assert(grid.at(s.field, x, y)^ == 0,
                "heat must not diffuse through a solid wall")
        }
    }
}

@(test)
decay_reduces_heat :: proc(t: ^testing.T) {
    walls := make_walls(32, 32)
    defer delete(walls.data)

    s, err := heat(walls, 0, 0.1, 3, 0, context.temp_allocator)
    assert(err == .None)

    for i in 0..<len(s.field.data) {
        s.field.data[i] = 1
    }

    before := total_heat(s.field)
    step(&s, nil, 10)
    after := total_heat(s.field)

    assert(after < before * 0.5, "decay must reduce the field without sources")
}

@(test)
moving_goal_updates_field :: proc(t: ^testing.T) {
    walls := make_walls(48, 48)
    defer delete(walls.data)

    s, err := heat(walls, 0.5, 0.02, 3, 1, context.temp_allocator)
    assert(err == .None)

    old_goal := Goal{pos = {12.5, 24.5}, weight = 1}
    new_goal := Goal{pos = {36.5, 24.5}, weight = 1}

    for _ in 0..<100 {
        step(&s, []Goal{old_goal}, 1)
    }
    old_peak := grid.at(s.field, 12, 24)^

    for _ in 0..<300 {
        step(&s, []Goal{new_goal}, 1)
    }

    old_after := grid.at(s.field, 12, 24)^
    new_peak := grid.at(s.field, 36, 24)^

    assert(new_peak > old_after, "the field must follow the moving goal")
    assert(old_after < old_peak, "heat must decay at the abandoned goal")
}

@(test)
source_ignores_walls :: proc(t: ^testing.T) {
    walls := make_walls(16, 16)
    defer delete(walls.data)

    for i in 0..<len(walls.data) {
        walls.data[i] = maps.OBSTACLE
    }

    s, err := heat(walls, 0.5, 0, 3, 1, context.temp_allocator)
    assert(err == .None)

    step(&s, []Goal{{pos = {8.5, 8.5}, weight = 1}}, 1)

    for value in s.field.data {
        assert(value == 0, "sources must never inject heat into walls")
    }
}

// A wall is an inert tap: the normalized convolution drops the missing neighbour from
// both the sum and the count, so a constant field is a fixed point even beside geometry.
// The cold-wall operator reported 0.875 here, that is the wall voting zero.
@(test)
wall_is_an_inert_neighbour :: proc(t: ^testing.T) {
    walls := make_walls(3, 3)
    defer delete(walls.data)
    grid.at(walls, 1, 0)^ = maps.OBSTACLE

    s, err := heat(walls, 0.5, 0, 0, 0, context.temp_allocator)
    assert(err == .None)

    for y in 0..<walls.height {
        for x in 0..<walls.width {
            grid.at(s.field, x, y)^ = grid.at(walls, x, y)^ == maps.WALKABLE ? 1 : 0
        }
    }

    step(&s, nil, 1)

    // The centre has three walkable neighbours of 1 and one wall arm: the sum loses
    // rate / 4 and the coverage loses exactly the same amount of kernel weight.
    assert(abs(grid.at(s.field, 1, 1)^ - 1) < 1e-6,
        "a wall must be an inert neighbour: the missing arm leaves both sum and count")
    assert(grid.at(s.field, 1, 0)^ == 0, "the mask must keep the wall at zero")
}

// The whole image summed: with insulating walls the only sink is the decay, so a uniform
// field loses exactly the factor keep per step and geometry costs nothing. Under the
// cold-wall operator mass instead drained through every arm that landed on a wall.
@(test)
insulating_walls_leave_only_decay :: proc(t: ^testing.T) {
    walls := make_walls(24, 24)
    defer delete(walls.data)
    for y in 0..<walls.height {
        for x in 0..<walls.width {
            if (x * 7 + y * y) % 5 == 0 {
                grid.at(walls, x, y)^ = maps.OBSTACLE
            }
        }
    }

    rate : f32 = 0.4
    decay : f32 = 0.1
    s, err := heat(walls, rate, decay, 0, 0, context.temp_allocator)
    assert(err == .None)

    walkable := 0
    for i in 0..<len(s.field.data) {
        if s.grid.data[i] == maps.WALKABLE {
            s.field.data[i] = 1
            walkable += 1
        } else {
            s.field.data[i] = 0
        }
    }
    step(&s, nil, 1)

    keep := 1 - decay
    // Summing 456 f32 terms accumulates ~2e-3 of rounding on a total near 410, so the
    // bound is 1e-2: the law being checked is exact, the summation is not.
    assert(abs(total_heat(s.field) - keep * f32(walkable)) < 1e-2,
        "insulating walls must cost nothing: mass shrinks by keep alone")
}

@(test)
isolated_cell_decays_by_keep_only :: proc(t: ^testing.T) {
    walls := make_walls(3, 3)
    defer delete(walls.data)
    for i in 0..<len(walls.data) {
        walls.data[i] = maps.OBSTACLE
    }
    grid.at(walls, 1, 1)^ = maps.WALKABLE

    s, err := heat(walls, 0.5, 0.1, 0, 0, context.temp_allocator)
    assert(err == .None)
    grid.at(s.field, 1, 1)^ = 1

    step(&s, nil, 1)

    // Coverage is the centre tap alone, so the renormalization cancels it and decay is
    // the only factor left: 0.9. The cold-wall operator gave 0.9 * (1 - rate) = 0.45,
    // charging the cell for arms the walkable mask closes.
    assert(abs(grid.at(s.field, 1, 1)^ - 0.9) < 1e-6,
        "an isolated cell must decay by keep, not by the unused arms")
}

// One step of the solver is the kernel of conv.diffusion correlated once with the field
// and multiplied by the walkable mask. On an all-walkable map that mask is the constant
// 1, so the solver must agree with conv.convolution to the last bit: the same stencil
// code path, and the same Border.CONSTANT zero extension cv::filter2D would apply.
@(test)
step_equals_repeated_convolution :: proc(t: ^testing.T) {
    walls := make_walls(16, 16)
    defer delete(walls.data)

    kernel, err_k := diffusion(0.45, 0.3, context.temp_allocator)
    assert(err_k == .None)

    s, err := heat(walls, 0.45, 0.3, 0, 0, context.temp_allocator)
    assert(err == .None)

    for i in 0..<len(s.field.data) {
        s.field.data[i] = 0
    }
    grid.at(s.field, 8, 8)^ = 1

    seeded, err_s := convolution(kernel, s.field, 6, Border.CONSTANT)
    assert(err_s == .None)
    defer delete(seeded.data)

    step(&s, nil, 6)

    for i in 0..<len(s.field.data) {
        assert(abs(s.field.data[i] - seeded.data[i]) < 1e-6,
            "the masked stencil and the raw kernel chain must be the same operator")
    }
}

// The footprint the splat writes is pinned rather than eyeballed: radius is measured to
// cell centres, so GOAL_RADIUS picks the sink region by cell count, and the same image is
// the heat source.
@(test)
goal_image_footprint :: proc(t: ^testing.T) {
    walls := make_walls(16, 16)
    defer delete(walls.data)

    goal := Goal{pos = {8.5, 8.5}, weight = 1}

    cell, err_cell := goal_image(walls, goal, 0.4, context.temp_allocator)
    assert(err_cell == .None)
    assert(total_heat(cell) == 1, "0.4 is the goal cell alone")

    cross, err_cross := goal_image(walls, goal, 1.0, context.temp_allocator)
    assert(err_cross == .None)
    assert(total_heat(cross) == 5, "1.0 is the four-neighbour cross, diagonals being further away")

    block, err_block := goal_image(walls, goal, 1.5, context.temp_allocator)
    assert(err_block == .None)
    assert(total_heat(block) == 9, "1.5 is the whole 3x3 block")

    grid.at(walls, 8, 7)^ = maps.OBSTACLE
    clipped, err_clip := goal_image(walls, goal, 1.5, context.temp_allocator)
    assert(err_clip == .None)
    assert(total_heat(clipped) == 8, "the footprint must be clipped by the walkable mask")
    assert(grid.at(clipped, 8, 7)^ == 0, "a goal may not put heat inside a wall")

    dead, err_dead := goal_image(walls, Goal{pos = {8.5, 8.5}, weight = 0}, 1.0, context.temp_allocator)
    assert(err_dead == .None)
    assert(total_heat(dead) == 0, "a zero weight must give the empty image")
}
