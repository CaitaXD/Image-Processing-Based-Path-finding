package main

import "core:math"
import "conv"
import "maps"
import "grid"
import "units"
import rl "vendor:raylib"

SEED :: 42
UNIT_COUNT :: 10000

HEAT_SUBSTEPS :: 2
HEAT_RATE :: 0.5
HEAT_DECAY :: 0.005
HEAT_SOURCE_RADIUS :: 4.0
HEAT_SOURCE_STRENGTH :: 1.0

ZOOM_MAX :: 64.0
ZOOM_STEP :: 1.2

W :: 1024
H :: 1024

main :: proc() {
    rl.InitWindow(W, H, "Caverns")
    defer rl.CloseWindow()
    rl.SetTargetFPS(60)

    dims := [2]int{auto_cast(rl.GetScreenWidth()), auto_cast(rl.GetScreenHeight())}

    generation  := 0
    cavern_grid := maps.cellular_automata(dims, SEED)
    all_units   := units.sample(cavern_grid, UNIT_COUNT, cast(u64)(SEED))
    goal, has_goal := units.sample_goal(cavern_grid, cast(u64)(SEED) + 1)

    heat_solver := conv.heat(
        cavern_grid,
        HEAT_RATE,
        HEAT_DECAY,
        HEAT_SOURCE_RADIUS,
        HEAT_SOURCE_STRENGTH,
    )
    heat_goals: [1]conv.Goal
    if has_goal {
        heat_goals[0] = conv.Goal {
            pos    = {f32(goal.pos[0]) + 0.5, f32(goal.pos[1]) + 0.5},
            weight = 1,
        }
    }

    heat_img := rl.GenImageColor(cast(i32)dims[0], cast(i32)dims[1], rl.BLACK)
    heat_tex := rl.LoadTextureFromImage(heat_img)
    rl.SetTextureFilter(heat_tex, .POINT)
    defer {
        rl.UnloadTexture(heat_tex)
        rl.UnloadImage(heat_img)
    }

    running := false

    tex := map_texture(cavern_grid)
    defer rl.UnloadTexture(tex)

    camera := rl.Camera2D {
        target = {f32(dims[0]) / 2, f32(dims[1]) / 2},
        zoom   = 1,
    }

    for !rl.WindowShouldClose() {
        if rl.IsKeyPressed(.R) {
            generation += 1
            seed := SEED + generation
            cavern_grid = maps.cellular_automata(dims, seed)
            all_units = units.sample(cavern_grid, UNIT_COUNT, cast(u64)(seed))
            goal, has_goal = units.sample_goal(cavern_grid, cast(u64)(seed) + 1)

            heat_solver = conv.heat(
                cavern_grid,
                HEAT_RATE,
                HEAT_DECAY,
                HEAT_SOURCE_RADIUS,
                HEAT_SOURCE_STRENGTH,
            )
            heat_goals[0] = {}
            if has_goal {
                heat_goals[0] = conv.Goal {
                    pos    = {f32(goal.pos[0]) + 0.5, f32(goal.pos[1]) + 0.5},
                    weight = 1,
                }
            }

            rl.UnloadTexture(tex)
            tex = map_texture(cavern_grid)
        }

        if rl.IsKeyPressed(.P) {
            running = !running
        }

        if running || rl.IsKeyPressed(.SPACE) {
            conv.step(&heat_solver, heat_goals[:], HEAT_SUBSTEPS)
        }
        update_heat_texture(heat_tex, &heat_img, &heat_solver)

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

        rl.BeginMode2D(camera)
        rl.DrawTextureEx(tex, {0, 0}, 0, 1, rl.WHITE)
        rl.DrawTextureEx(heat_tex, {0, 0}, 0, 1, rl.WHITE)

        unit_size := max(1, 2 / camera.zoom)
        for unit in all_units {
            pos := [2]f32{
                f32(unit.pos[0]) + 0.5 - unit_size / 2,
                f32(unit.pos[1]) + 0.5 - unit_size / 2,
            }
            rl.DrawRectangleV(pos, {unit_size, unit_size}, rl.RED)
        }

        if has_goal {
            goal_size := max(1, 4 / camera.zoom)
            goal_pos := [2]f32{
                f32(goal.pos[0]) + 0.5 - goal_size / 2,
                f32(goal.pos[1]) + 0.5 - goal_size / 2,
            }
            rl.DrawRectangleV(goal_pos, {goal_size, goal_size}, rl.GREEN)
        }
        rl.EndMode2D()

        rl.DrawText(rl.TextFormat("zoom: %.2fx", camera.zoom), 10, 10, 20, rl.WHITE)
        rl.DrawText("Wheel: zoom   RMB/MMB: pan   Space: step   P: run/pause   R: regenerate", 10, 34, 16, rl.LIGHTGRAY)

        rl.EndDrawing()
    }
}

zoom_at :: proc(camera: ^rl.Camera2D, screen: rl.Vector2, factor, min_zoom, max_zoom: f32) {
    before := rl.GetScreenToWorld2D(screen, camera^)
    camera.zoom = clamp(camera.zoom * factor, min_zoom, max_zoom)
    after := rl.GetScreenToWorld2D(screen, camera^)
    camera.target += before - after
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
