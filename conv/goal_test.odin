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
