package main

import sim "simulation"
import rl "vendor:raylib"
import "core:math"

WINDOW_SCALE :: 3
ZOOM_MAX     :: 64.0
ZOOM_STEP    :: 1.2

main :: proc() {
    rl.InitWindow(1, 1, "Caverns")
    rl.SetWindowState({.FULLSCREEN_MODE})
    defer rl.CloseWindow()
    rl.SetTargetFPS(sim.FPS)

    dims := [2]int{sim.W, sim.H}
    s := sim.make(dims, 0)
    defer {
        rl.UnloadTexture(s.map_tex)
        rl.UnloadTexture(s.heat_tex)
        rl.UnloadImage(s.heat_img)
    }

    running := true
    active_visualization := sim.Visualization.NONE
    recorder: Recorder
    defer record_stop(&recorder)

    camera := rl.Camera2D {
        target = {f32(dims[0]) / 2, f32(dims[1]) / 2},
        zoom   = 1,
    }
    record_camera: rl.Camera2D

    for !rl.WindowShouldClose() {
        if rl.IsKeyPressed(.R) && !recorder.active {
            sim.regenerate(&s)
        }

        if rl.IsKeyPressed(.P) {
            running = !running
        }

        if rl.IsKeyPressed(.V) {
            if recorder.active {
                record_stop(&recorder)
            } 
            else {
                sim.reset(&s)
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
        if rl.IsKeyPressed(.TWO) {
            active_visualization = .FLOW
        }

        if running || recorder.active || rl.IsKeyPressed(.SPACE) {
            if !(recorder.active && recorder.frames == 0) {
                sim.step(&s, sim.SIM_DT)
            }
        }
        if recorder.active || active_visualization == .HEAT {
            sim.update_heat(&s)
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
            record_capture(&recorder, &s, record_camera)
        }

        sim.draw(&s, active_visualization, camera)

        rl.DrawText(rl.TextFormat("zoom: %.2fx", camera.zoom), 10, 10, 20, rl.WHITE)
        rl.DrawText("Wheel: zoom   RMB/MMB: pan   Space: step   P: run/pause   R: regenerate   V: record all views from start   0: none   1: heat   2: flow", 10, 34, 16, rl.LIGHTGRAY)
        if recorder.active {
            label := rl.TextFormat("REC all views %d/%d", recorder.frames, RECORD_SECONDS * sim.FPS)
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
