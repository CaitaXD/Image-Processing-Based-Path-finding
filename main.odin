package main

import "core:math"
import "conv"
import "maps"
import "grid"
import "units"
import "nav"
import rl "vendor:raylib"

W :: 400
H :: 400

WINDOW_SCALE :: 2

SEED                 :: 42
UNIT_COUNT           :: 100
HEAT_SUBSTEPS        :: 2
HEAT_RATE            :: 0.5
HEAT_DECAY           :: 0.0005
HEAT_SOURCE_RADIUS   :: f32(max(W, H)) / 2
HEAT_SOURCE_STRENGTH :: 1.0
UNIT_SPEED           :: 12.0
MAX_MOVE_STEP        :: 0.25
GOAL_RADIUS          :: 1.0
ZOOM_MAX             :: 64.0
ZOOM_STEP            :: 1.2
FPS                  :: 10
SIM_DT               :: f32(1.0 / FPS)

Visualization :: enum {
    NONE,
    HEAT,
}

VISUALIZATIONS :: [?]Visualization{.NONE, .HEAT}
VISUALIZATION_COUNT :: len(VISUALIZATIONS)

main :: proc() {
    rl.InitWindow(W * WINDOW_SCALE, H * WINDOW_SCALE, "Caverns")
    defer rl.CloseWindow()
    rl.SetTargetFPS(FPS)

    dims := [2]int{W, H}
    sim := simulation_make(dims, 0)
    defer {
        rl.UnloadTexture(sim.map_tex)
        rl.UnloadTexture(sim.heat_tex)
        rl.UnloadImage(sim.heat_img)
    }

    running := true
    active_visualization := Visualization.NONE
    recorder: Recorder
    defer record_stop(&recorder)

    camera := rl.Camera2D {
        target = {f32(dims[0]) / 2, f32(dims[1]) / 2},
        zoom   = 1,
    }
    record_camera: rl.Camera2D

    for !rl.WindowShouldClose() {
        if rl.IsKeyPressed(.R) && !recorder.active {
            simulation_regenerate(&sim)
        }

        if rl.IsKeyPressed(.P) {
            running = !running
        }

        if rl.IsKeyPressed(.V) {
            if recorder.active {
                record_stop(&recorder)
            } else {
                simulation_reset(&sim)
                running = true
                record_camera = rl.Camera2D {
                    target = {f32(dims[0]) / 2, f32(dims[1]) / 2},
                    offset = {f32(RECORD_WIDTH) / 2, f32(RECORD_HEIGHT) / 2},
                    zoom   = max(min(f32(RECORD_WIDTH) / f32(dims[0]), f32(RECORD_HEIGHT) / f32(dims[1])), 0.001),
                }
                _ = record_start(&recorder)
            }
        }

        if rl.IsKeyPressed(.ZERO) {
            active_visualization = .NONE
        }
        if rl.IsKeyPressed(.ONE) {
            active_visualization = .HEAT
        }

        if running || recorder.active || rl.IsKeyPressed(.SPACE) {
            if !(recorder.active && recorder.frames == 0) {
                simulation_step(&sim, SIM_DT)
            }
        }
        if recorder.active || active_visualization == .HEAT {
            simulation_update_heat_texture(&sim)
        }

        screen := [2]f32{f32(rl.GetScreenWidth()), f32(rl.GetScreenHeight())}
        min_zoom := max(min(screen[0] / f32(dims[0]), screen[1] / f32(dims[1])), 0.001)

        camera.offset = screen / 2
        camera.zoom = clamp(camera.zoom, min_zoom, ZOOM_MAX)

        if wheel := rl.GetMouseWheelMove(); wheel != 0 {
            zoom_at(&camera, rl.GetMousePosition(), math.pow(ZOOM_STEP, wheel), min_zoom, ZOOM_MAX)
        }

        if rl.IsMouseButtonDown(.MIDDLE) || rl.IsMouseButtonDown(.RIGHT) {
            camera.target -= rl.GetMouseDelta() / camera.zoom
        }

        camera.target[0] = clamp(camera.target[0], 0, f32(dims[0]))
        camera.target[1] = clamp(camera.target[1], 0, f32(dims[1]))

        rl.BeginDrawing()
        rl.ClearBackground({0x18, 0x18, 0x18, 255})

        if recorder.active {
            record_capture(&recorder, &sim, record_camera)
        }

        simulation_draw(&sim, active_visualization, camera)

        rl.DrawText(rl.TextFormat("zoom: %.2fx", camera.zoom), 10, 10, 20, rl.WHITE)
        rl.DrawText("Wheel: zoom   RMB/MMB: pan   Space: step   P: run/pause   R: regenerate   V: record all views from start   0: none   1: heat/nav", 10, 34, 16, rl.LIGHTGRAY)
        if recorder.active {
            label := rl.TextFormat("REC all views %d/%d", recorder.frames, RECORD_SECONDS * FPS)
            rl.DrawText(label, 10, 58, 16, rl.RED)
        }

        rl.EndDrawing()
    }
}

zoom_at :: proc(camera: ^rl.Camera2D, screen: rl.Vector2, factor, min_zoom, max_zoom: f32) {
    before := rl.GetScreenToWorld2D(screen, camera^)
    camera.zoom = clamp(camera.zoom * factor, min_zoom, max_zoom)
    after := rl.GetScreenToWorld2D(screen, camera^)
    camera.target += before - after
}

update_units :: proc(all_units: []units.Unit, walls: grid.Grid2D(u8), flow: ^nav.FlowField, goal: units.Goal, has_goal: bool, dt: f32) {
    goal_pos := [2]f32{f32(goal.pos[0]) + 0.5, f32(goal.pos[1]) + 0.5}

    for i in 0..<len(all_units) {
        u := &all_units[i]

        if has_goal {
            dx := u.pos[0] - goal_pos[0]
            dy := u.pos[1] - goal_pos[1]
            if dx*dx + dy*dy <= GOAL_RADIUS*GOAL_RADIUS {
                continue
            }
        }

        dir := nav.sample(flow, u.pos)
        if dir[0] != 0 || dir[1] != 0 {
            u.pos = move_with_collision(walls, u.pos, dir * (UNIT_SPEED * dt))
        }
    }
}

move_with_collision :: proc(walls: grid.Grid2D(u8), pos, delta: [2]f32) -> [2]f32 {
    length := math.sqrt(delta[0]*delta[0] + delta[1]*delta[1])
    if length <= 0 {
        return pos
    }

    steps := int(math.ceil(length / MAX_MOVE_STEP))
    if steps < 1 {
        steps = 1
    }
    step := delta / f32(steps)

    current := pos
    for _ in 0..<steps {
        next := try_move(walls, current, step)
        if next == current {
            break
        }
        current = next
    }
    return current
}

try_move :: proc(walls: grid.Grid2D(u8), pos, delta: [2]f32) -> [2]f32 {
    full := pos + delta
    if walkable_at(walls, full) && !cuts_corner(walls, pos, full) {
        return full
    }

    x_only := [2]f32{pos[0] + delta[0], pos[1]}
    if walkable_at(walls, x_only) {
        return x_only
    }

    y_only := [2]f32{pos[0], pos[1] + delta[1]}
    if walkable_at(walls, y_only) {
        return y_only
    }

    return pos
}

cuts_corner :: proc(walls: grid.Grid2D(u8), from, to: [2]f32) -> bool {
    fx := int(math.floor(from[0]))
    fy := int(math.floor(from[1]))
    tx := int(math.floor(to[0]))
    ty := int(math.floor(to[1]))

    if fx == tx || fy == ty {
        return false
    }

    return !walkable_cell(walls, tx, fy) || !walkable_cell(walls, fx, ty)
}

walkable_at :: proc(walls: grid.Grid2D(u8), pos: [2]f32) -> bool {
    return walkable_cell(walls, int(math.floor(pos[0])), int(math.floor(pos[1])))
}

walkable_cell :: proc(walls: grid.Grid2D(u8), x, y: int) -> bool {
    if x < 0 || y < 0 || x >= walls.width || y >= walls.height {
        return false
    }
    return grid.at(walls, x, y)^ == maps.WALKABLE
}

update_heat_texture :: proc(tex: rl.Texture2D, img: ^rl.Image, s: ^conv.HeatSolver) {
    max_u: f32
    for value in s.field.data {
        if value > max_u {
            max_u = value
        }
    }
    inv_max := 1 / max(max_u, f32(1e-6))

    pixels := cast([^]rl.Color)img.data
    for y in 0..<s.field.height {
        for x in 0..<s.field.width {
            index := x + y * s.field.width
            if grid.at(s.grid, x, y)^ != maps.WALKABLE {
                pixels[index] = {0, 0, 0, 0}
            } else {
                pixels[index] = heat_color(grid.at(s.field, x, y)^ * inv_max)
            }
        }
    }

    rl.UpdateTexture(tex, img.data)
}

heat_color :: proc(t: f32) -> rl.Color {
    t := clamp(t, 0, 1)
    r, g, b: f32

    if t < 0.25 {
        k := t / 0.25
        r, g, b = 0, k, 1
    } else if t < 0.5 {
        k := (t - 0.25) / 0.25
        r, g, b = 0, 1, 1 - k
    } else if t < 0.75 {
        k := (t - 0.5) / 0.25
        r, g, b = k, 1, 0
    } else {
        k := (t - 0.75) / 0.25
        r, g, b = 1, 1 - k, 0
    }

    shade := 0.15 + 0.85 * t
    return {
        u8(r * 255 * shade),
        u8(g * 255 * shade),
        u8(b * 255 * shade),
        255,
    }
}

map_texture :: proc(a: grid.Grid2D(u8)) -> rl.Texture2D {
    img := rl.GenImageColor(cast(i32)a.width, cast(i32)a.height, rl.BLACK)
    pixels := cast([^]rl.Color)img.data
    for y in 0..<a.height {
        for x in 0..<a.width {
            pixels[x + y * a.width] = grid.at(a, x, y)^ == maps.WALKABLE ? rl.GRAY : rl.BLACK
        }
    }
    tex := rl.LoadTextureFromImage(img)
    rl.SetTextureFilter(tex, .POINT)
    rl.UnloadImage(img)
    return tex
}
