package nav

import "base:runtime"
import "core:math"
import "../conv"
import "../grid"
import "../maps"

FlowField :: struct {
    walls: grid.Grid2D(u8),
    dirs:  grid.Grid2D([2]f32),
}

make :: proc(walls: grid.Grid2D(u8), allocator := context.allocator) -> (f: FlowField, err: runtime.Allocator_Error) {
    f.walls = walls
    f.dirs, err = grid.make([2]f32, walls.width, walls.height, allocator)
    return
}

update :: proc(f: ^FlowField, s: ^conv.HeatSolver) {
    assert(s.field.width == f.walls.width && s.field.height == f.walls.height)

    for y in 0..<f.walls.height {
        for x in 0..<f.walls.width {
            if grid.at(f.walls, x, y)^ != maps.WALKABLE {
                grid.at(f.dirs, x, y)^ = {}
                continue
            }

            value := grid.at(s.field, x, y)^
            left, right, up, down := value, value, value, value

            if x > 0 && grid.at(f.walls, x-1, y)^ == maps.WALKABLE {
                left = grid.at(s.field, x-1, y)^
            }
            if x+1 < f.walls.width && grid.at(f.walls, x+1, y)^ == maps.WALKABLE {
                right = grid.at(s.field, x+1, y)^
            }
            if y > 0 && grid.at(f.walls, x, y-1)^ == maps.WALKABLE {
                up = grid.at(s.field, x, y-1)^
            }
            if y+1 < f.walls.height && grid.at(f.walls, x, y+1)^ == maps.WALKABLE {
                down = grid.at(s.field, x, y+1)^
            }

            gx := right - left
            gy := down - up
            length := math.sqrt(gx*gx + gy*gy)

            if length > 1e-12 {
                grid.at(f.dirs, x, y)^ = {gx / length, gy / length}
            } else {
                grid.at(f.dirs, x, y)^ = {}
            }
        }
    }
}

sample :: proc(f: ^FlowField, pos: [2]f32) -> [2]f32 {
    cx := int(math.floor(pos[0]))
    cy := int(math.floor(pos[1]))
    if !walkable_cell(f.walls, cx, cy) {
        return {}
    }

    gx := pos[0] - 0.5
    gy := pos[1] - 0.5
    x0 := int(math.floor(gx))
    y0 := int(math.floor(gy))
    x1 := x0 + 1
    y1 := y0 + 1

    if walkable_cell(f.walls, x0, y0) &&
       walkable_cell(f.walls, x1, y0) &&
       walkable_cell(f.walls, x0, y1) &&
       walkable_cell(f.walls, x1, y1) {
        fx := gx - f32(x0)
        fy := gy - f32(y0)

        d00 := grid.at(f.dirs, x0, y0)^
        d10 := grid.at(f.dirs, x1, y0)^
        d01 := grid.at(f.dirs, x0, y1)^
        d11 := grid.at(f.dirs, x1, y1)^

        dx := (d00[0]*(1-fx) + d10[0]*fx)*(1-fy) + (d01[0]*(1-fx) + d11[0]*fx)*fy
        dy := (d00[1]*(1-fx) + d10[1]*fx)*(1-fy) + (d01[1]*(1-fx) + d11[1]*fx)*fy

        length := math.sqrt(dx*dx + dy*dy)
        if length > 1e-12 {
            return {dx / length, dy / length}
        }
    }

    return grid.at(f.dirs, cx, cy)^
}

@(private)
walkable_cell :: proc(walls: grid.Grid2D(u8), x, y: int) -> bool {
    if x < 0 || y < 0 || x >= walls.width || y >= walls.height {
        return false
    }
    return grid.at(walls, x, y)^ == maps.WALKABLE
}
