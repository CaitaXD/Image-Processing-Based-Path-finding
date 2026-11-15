package heat

import "base:runtime"
import "core:testing"
import "../maps"
import cv "../odin-opencv"

@(private)
make_walls :: proc(w, h: int) -> cv.Mat(f32) {
    g, err := cv.mat_new(f32, h, w, context.temp_allocator)
    assert(err == .None)
    for i in 0..<int(g.rows * g.cols) {
        g.data[i] = maps.WALKABLE
    }
    return g
}

@(private)
total_heat :: proc(g: cv.Mat(f32)) -> f32 {
    total: f32
    for value in cv.pixels(g) {
        total += value
    }
    return total
}

@(test)
sources_add_linearly :: proc(t: ^testing.T) {
    runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
    context.allocator = context.temp_allocator
    walls := make_walls(48, 48)

    a, err_a := solver(walls, 0.5, 0.01, context.temp_allocator)
    assert(err_a == .None)

    b, err_b := solver(walls, 0.5, 0.01, context.temp_allocator)
    assert(err_b == .None)

    both, err_both := solver(walls, 0.5, 0.01, context.temp_allocator)
    assert(err_both == .None)

    pos_a := [2]int{12, 24}
    pos_b := [2]int{36, 24}

    src_a, err_a1 := splat_heat_source(walls, pos_a, 3, 1, context.temp_allocator)
    assert(err_a1 == .None)
    src_b, err_b1 := splat_heat_source(walls, pos_b, 3, 1, context.temp_allocator)
    assert(err_b1 == .None)

    both_src, err_both_src := cv.mat_new(f32, int(walls.rows), int(walls.cols), context.temp_allocator)
    assert(err_both_src == .None)
    for i in 0..<int(both_src.rows * both_src.cols) {
        both_src.data[i] = src_a.data[i] + src_b.data[i]
    }

    for _ in 0..<30 {
        step(&a, &src_a, 1)
        step(&b, &src_b, 1)
        step(&both, &both_src, 1)
    }

    for i in 0..<int(both.field.rows * both.field.cols) {
        expected := a.field.data[i] + b.field.data[i]
        assert(abs(both.field.data[i] - expected) < 1e-4,
            "multiple goals must superpose linearly")
    }
}

@(test)
weights_scale_field :: proc(t: ^testing.T) {
    runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
    context.allocator = context.temp_allocator
    walls := make_walls(48, 48)

    one, err_one := solver(walls, 0.5, 0.01, context.temp_allocator)
    assert(err_one == .None)

    two, err_two := solver(walls, 0.5, 0.01, context.temp_allocator)
    assert(err_two == .None)

    pos := [2]int{24, 24}

    one_src, err_one1 := splat_heat_source(walls, pos, 3, 1, context.temp_allocator)
    assert(err_one1 == .None)
    two_src, err_two1 := splat_heat_source(walls, pos, 3, 2, context.temp_allocator)
    assert(err_two1 == .None)

    for _ in 0..<40 {
        step(&one, &one_src, 1)
        step(&two, &two_src, 1)
    }

    for i in 0..<int(two.field.rows * two.field.cols) {
        assert(abs(two.field.data[i] - 2 * one.field.data[i]) < 1e-4,
            "weight must scale the source linearly")
    }
}

@(test)
walls_block_heat :: proc(t: ^testing.T) {
    runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
    context.allocator = context.temp_allocator
    walls := make_walls(48, 48)

    for y in 0..<int(walls.rows) {
        cv.at(&walls, 24, y)^ = maps.OBSTACLE
    }

    s, err := solver(walls, 0.5, 0.01, context.temp_allocator)
    assert(err == .None)

    source, err_source := splat_heat_source(walls, [2]int{12, 24}, 3, 1, context.temp_allocator)
    assert(err_source == .None)
    for _ in 0..<300 {
        step(&s, &source, 1)
    }

    for y in 0..<int(s.field.rows) {
        for x in 25..<int(s.field.cols) {
            assert(cv.at(&s.field, x, y)^ == 0,
                "heat must not diffuse through a solid wall")
        }
    }
}

@(test)
decay_reduces_heat :: proc(t: ^testing.T) {
    runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
    context.allocator = context.temp_allocator
    walls := make_walls(32, 32)

    s, err := solver(walls, 0, 0.1, context.temp_allocator)
    assert(err == .None)

    for i in 0..<int(s.field.rows * s.field.cols) {
        s.field.data[i] = 1
    }

    before := total_heat(s.field)
    step(&s, nil, 10)
    after := total_heat(s.field)

    assert(after < before * 0.5, "decay must reduce the field without sources")
}

@(test)
moving_goal_updates_field :: proc(t: ^testing.T) {
    runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
    context.allocator = context.temp_allocator
    walls := make_walls(48, 48)

    s, err := solver(walls, 0.5, 0.02, context.temp_allocator)
    assert(err == .None)

    old_src, err_old := splat_heat_source(walls, [2]int{12, 24}, 3, 1, context.temp_allocator)
    assert(err_old == .None)
    new_src, err_new := splat_heat_source(walls, [2]int{36, 24}, 3, 1, context.temp_allocator)
    assert(err_new == .None)

    for _ in 0..<100 {
        step(&s, &old_src, 1)
    }
    old_peak := cv.at(&s.field, 12, 24)^

    for _ in 0..<300 {
        step(&s, &new_src, 1)
    }

    old_after := cv.at(&s.field, 12, 24)^
    new_peak := cv.at(&s.field, 36, 24)^

    assert(new_peak > old_after, "the field must follow the moving goal")
    assert(old_after < old_peak, "heat must decay at the abandoned goal")
}

@(test)
source_ignores_walls :: proc(t: ^testing.T) {
    runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
    context.allocator = context.temp_allocator
    walls := make_walls(16, 16)

    for i in 0..<int(walls.rows * walls.cols) {
        walls.data[i] = maps.OBSTACLE
    }

    s, err := solver(walls, 0.5, 0, context.temp_allocator)
    assert(err == .None)

    source, err_source := splat_heat_source(walls, [2]int{8, 8}, 3, 1, context.temp_allocator)
    assert(err_source == .None)
    step(&s, &source, 1)

    for value in cv.pixels(s.field) {
        assert(value == 0, "sources must never inject heat into walls")
    }
}

// A wall is an inert tap: the normalized convolution drops the missing neighbour from
// both the sum and the count, so a constant field is a fixed point even beside geometry.
// The cold-wall operator reported 0.875 here, that is the wall voting zero.
@(test)
wall_is_an_inert_neighbour :: proc(t: ^testing.T) {
    runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
    context.allocator = context.temp_allocator
    walls := make_walls(3, 3)
    cv.at(&walls, 1, 0)^ = maps.OBSTACLE

    s, err := solver(walls, 0.5, 0, context.temp_allocator)
    assert(err == .None)

    for y in 0..<int(walls.rows) {
        for x in 0..<int(walls.cols) {
            cv.at(&s.field, x, y)^ = cv.at(&walls, x, y)^ == maps.WALKABLE ? 1 : 0
        }
    }

    step(&s, nil, 1)

    // The centre has three walkable neighbours of 1 and one wall arm: the sum loses
    // rate / 4 and the coverage loses exactly the same amount of kernel weight.
    assert(abs(cv.at(&s.field, 1, 1)^ - 1) < 1e-6,
        "a wall must be an inert neighbour: the missing arm leaves both sum and count")
    assert(cv.at(&s.field, 1, 0)^ == 0, "the mask must keep the wall at zero")
}

// The whole image summed: with insulating walls the only sink is the decay, so a uniform
// field loses exactly the factor keep per step and geometry costs nothing. Under the
// cold-wall operator mass instead drained through every arm that landed on a wall.
@(test)
insulating_walls_leave_only_decay :: proc(t: ^testing.T) {
    runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
    context.allocator = context.temp_allocator
    walls := make_walls(24, 24)
    for y in 0..<int(walls.rows) {
        for x in 0..<int(walls.cols) {
            if (x * 7 + y * y) % 5 == 0 {
                cv.at(&walls, x, y)^ = maps.OBSTACLE
            }
        }
    }

    rate : f32 = 0.4
    decay : f32 = 0.1
    s, err := solver(walls, rate, decay, context.temp_allocator)
    assert(err == .None)

    walkable := 0
    for i in 0..<int(s.field.rows * s.field.cols) {
        if s.navmap.data[i] == maps.WALKABLE {
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
    runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
    context.allocator = context.temp_allocator
    walls := make_walls(3, 3)
    for i in 0..<int(walls.rows * walls.cols) {
        walls.data[i] = maps.OBSTACLE
    }
    cv.at(&walls, 1, 1)^ = maps.WALKABLE

    s, err := solver(walls, 0.5, 0.1, context.temp_allocator)
    assert(err == .None)
    cv.at(&s.field, 1, 1)^ = 1

    step(&s, nil, 1)

    // Coverage is the centre tap alone, so the renormalization cancels it and decay is
    // the only factor left: 0.9. The cold-wall operator gave 0.9 * (1 - rate) = 0.45,
    // charging the cell for arms the walkable mask closes.
    assert(abs(cv.at(&s.field, 1, 1)^ - 0.9) < 1e-6,
        "an isolated cell must decay by keep, not by the unused arms")
}

// The footprint the splat writes is pinned rather than eyeballed: radius is measured to
// cell centres, so GOAL_RADIUS picks the sink region by cell count, and the same image is
// the heat source.
@(test)
goal_image_footprint :: proc(t: ^testing.T) {
    runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
    context.allocator = context.temp_allocator
    walls := make_walls(16, 16)

    pos := [2]int{8, 8}

    cell, err_cell := splat_goal_disc(walls, pos, 0.4, context.temp_allocator)
    assert(err_cell == .None)
    assert(total_heat(cell) == 1, "0.4 is the goal cell alone")

    cross, err_cross := splat_goal_disc(walls, pos, 1.0, context.temp_allocator)
    assert(err_cross == .None)
    assert(total_heat(cross) == 5, "1.0 is the four-neighbour cross, diagonals being further away")

    block, err_block := splat_goal_disc(walls, pos, 1.5, context.temp_allocator)
    assert(err_block == .None)
    assert(total_heat(block) == 9, "1.5 is the whole 3x3 block")

    cv.at(&walls, 8, 7)^ = maps.OBSTACLE
    clipped, err_clip := splat_goal_disc(walls, pos, 1.5, context.temp_allocator)
    assert(err_clip == .None)
    assert(total_heat(clipped) == 8, "the footprint must be clipped by the walkable mask")
    assert(cv.at(&clipped, 8, 7)^ == 0, "a goal may not put heat inside a wall")

    dead, err_dead := splat_goal_disc(walls, pos, 0, context.temp_allocator)
    assert(err_dead == .None)
    assert(total_heat(dead) == 0, "a goal with no footprint must give the empty image")
}
