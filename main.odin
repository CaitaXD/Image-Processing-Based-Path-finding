package main

import "core:math"
import "maps"
import "grid"
import "units"
import rl "vendor:raylib"

SEED :: 42
UNIT_COUNT :: 10000

ZOOM_MAX :: 64.0
ZOOM_STEP :: 1.2

W :: 1024
H :: 1024

main :: proc() {
    rl.InitWindow(W, H, "Caverns")
    defer rl.CloseWindow()
    rl.SetTargetFPS(60)

    dims := [2]int{auto_cast(rl.GetScreenWidth()), auto_cast(rl.GetScreenHeight())}

    generation := 0
    a := maps.cellular_automata(dims, SEED)
    u := units.sample(a, UNIT_COUNT, cast(u64)(SEED))
    goal, has_goal := units.sample_goal(a, cast(u64)(SEED) + 1)
    tex := map_texture(a)
    defer {rl.UnloadTexture(tex)}

    camera := rl.Camera2D {
        target = {f32(dims[0]) / 2, f32(dims[1]) / 2},
        zoom   = 1,
    }

    for !rl.WindowShouldClose() {
        if rl.IsKeyPressed(.R) {
            generation += 1
            seed := SEED + generation
            a = maps.cellular_automata(dims, seed)
            u = units.sample(a, UNIT_COUNT, cast(u64)(seed))
            goal, has_goal = units.sample_goal(a, cast(u64)(seed) + 1)
            rl.UnloadTexture(tex)
            tex = map_texture(a)
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

        rl.BeginMode2D(camera)
        rl.DrawTextureEx(tex, {0, 0}, 0, 1, rl.WHITE)

        unit_size := max(1, 2 / camera.zoom)
        for unit in u {
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
        rl.DrawText("Wheel: zoom   RMB/MMB: pan   R: regenerate", 10, 34, 16, rl.LIGHTGRAY)

        rl.EndDrawing()
    }
}

zoom_at :: proc(camera: ^rl.Camera2D, screen: rl.Vector2, factor, min_zoom, max_zoom: f32) {
    before := rl.GetScreenToWorld2D(screen, camera^)
    camera.zoom = clamp(camera.zoom * factor, min_zoom, max_zoom)
    after := rl.GetScreenToWorld2D(screen, camera^)
    camera.target += before - after
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
