package nav

import "base:runtime"
import "core:math"
import "core:testing"
import "../heat"
import "../maps"
import cv "../odin-opencv"

@(private)
make_walls :: proc(w, h: int) -> cv.Mat(f32) {
    g, err := cv.mat_new(f32, h, w, context.temp_allocator)
    assert(err == .None)
    for i in 0..<w * h {
        g.data[i] = maps.WALKABLE
    }
    return g
}

@(test)
gradient_points_uphill :: proc(t: ^testing.T) {
    runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
    context.allocator = context.temp_allocator
    walls := make_walls(9, 9)

    s := heat.solver(walls, 0.5, 0.0005, context.temp_allocator)

    for y in 0..<int(walls.rows) {
        for x in 0..<int(walls.cols) {
            cv.at(&s.field, x, y)^ = f32(x*x + y*y)
        }
    }

    f := make(walls, context.temp_allocator)
    update(&f, &s)

    dir := cv.at(&f.field, 4, 4)^
    assert(dir[0] > 0.5 && dir[1] > 0.5, "gradient must point toward increasing heat")
}

@(test)
walls_do_not_create_normal_gradient :: proc(t: ^testing.T) {
    runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
    context.allocator = context.temp_allocator
    walls := make_walls(9, 9)

    cv.at(&walls, 5, 4)^ = maps.OBSTACLE

    s := heat.solver(walls, 0.5, 0.0005, context.temp_allocator)

    for y in 0..<int(walls.rows) {
        for x in 0..<int(walls.cols) {
            cv.at(&s.field, x, y)^ = f32(y)
        }
    }

    f := make(walls, context.temp_allocator)
    update(&f, &s)

    dir := cv.at(&f.field, 4, 4)^
    assert(dir[0] == 0, "gradient must not point into a wall")
    assert(dir[1] > 0.5, "gradient must still follow the wall")

    wall_dir := cv.at(&f.field, 5, 4)^
    assert(wall_dir[0] == 0 && wall_dir[1] == 0, "wall cells have no direction")
}

@(test)
broad_source_has_gradient_across_map :: proc(t: ^testing.T) {
    runtime.DEFAULT_TEMP_ALLOCATOR_TEMP_GUARD()
    context.allocator = context.temp_allocator
    walls := make_walls(200, 200)

    goal := [2]f32{100.5, 100.5}
    goal_cell := [2]int{100, 100}
    sigma := f32(200) / 2

    s := heat.solver(walls, 0.5, 0.0005, context.temp_allocator)
    src, err_src := heat.splat_heat_source(walls, goal_cell, sigma, 1, context.temp_allocator)
    assert(err_src == .None)
    heat.seed_source(&s, src)

    f := make(walls, context.temp_allocator)
    update(&f, &s)

    corners := [][2]f32{{0.5, 0.5}, {199.5, 0.5}, {0.5, 199.5}, {199.5, 199.5}}
    for pos in corners {
        dir := cv.at(&f.field, int(pos[0]), int(pos[1]))^
        to_goal := goal - pos
        length := math.sqrt(to_goal[0]*to_goal[0] + to_goal[1]*to_goal[1])
        dot := dir[0]*to_goal[0]/length + dir[1]*to_goal[1]/length
        assert(dot > 0.9, "broad source must point toward the goal across the map")
    }
}
