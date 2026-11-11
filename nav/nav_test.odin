package nav

import "core:math"
import "core:testing"
import "../conv"
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

@(test)
gradient_points_uphill :: proc(t: ^testing.T) {
    walls := make_walls(9, 9)
    defer delete(walls.data)

    s := conv.heat(walls, 0.5, 0.0005, 1, 1, context.temp_allocator)

    for y in 0..<walls.height {
        for x in 0..<walls.width {
            grid.at(s.field, x, y)^ = f32(x*x + y*y)
        }
    }

    f := make(walls, context.temp_allocator)
    update(&f, &s)

    dir := sample(&f, {4.5, 4.5})
    assert(dir[0] > 0.5 && dir[1] > 0.5, "gradient must point toward increasing heat")
}

@(test)
walls_do_not_create_normal_gradient :: proc(t: ^testing.T) {
    walls := make_walls(9, 9)
    defer delete(walls.data)

    grid.at(walls, 5, 4)^ = maps.OBSTACLE

    s := conv.heat(walls, 0.5, 0.0005, 1, 1, context.temp_allocator)

    for y in 0..<walls.height {
        for x in 0..<walls.width {
            grid.at(s.field, x, y)^ = f32(y)
        }
    }

    f := make(walls, context.temp_allocator)
    update(&f, &s)

    dir := sample(&f, {4.5, 4.5})
    assert(dir[0] == 0, "gradient must not point into a wall")
    assert(dir[1] > 0.5, "gradient must still follow the wall")

    wall_dir := sample(&f, {5.5, 4.5})
    assert(wall_dir[0] == 0 && wall_dir[1] == 0, "wall cells have no direction")
}

@(test)
sample_interpolates_directions :: proc(t: ^testing.T) {
    walls := make_walls(5, 5)
    defer delete(walls.data)

    f := make(walls, context.temp_allocator)

    for i in 0..<len(f.field.data) {
        f.field.data[i] = {1, 0}
    }

    dir := sample(&f, {2.5, 2.5})
    assert(abs(dir[0] - 1) < 1e-6 && abs(dir[1]) < 1e-6, "sample must return the local direction")
}

@(test)
broad_source_has_gradient_across_map :: proc(t: ^testing.T) {
    walls := make_walls(200, 200)
    defer delete(walls.data)

    goal := [2]f32{100.5, 100.5}
    sigma := f32(200) / 2

    s := conv.heat(walls, 0.5, 0.0005, sigma, 1, context.temp_allocator)
    src := conv.make_source(walls, conv.Goal{pos = goal, weight = 1}, sigma, 1, context.temp_allocator)
    conv.seed_source(&s, src)

    f := make(walls, context.temp_allocator)
    update(&f, &s)

    corners := [][2]f32{{0.5, 0.5}, {199.5, 0.5}, {0.5, 199.5}, {199.5, 199.5}}
    for pos in corners {
        dir := sample(&f, pos)
        to_goal := goal - pos
        length := math.sqrt(to_goal[0]*to_goal[0] + to_goal[1]*to_goal[1])
        dot := dir[0]*to_goal[0]/length + dir[1]*to_goal[1]/length
        assert(dot > 0.9, "broad source must point toward the goal across the map")
    }
}
