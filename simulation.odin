package main

import "conv"
import "grid"
import "maps"
import "nav"
import "units"
import rl "vendor:raylib"

Simulation :: struct {
    dims:       [2]int,
    generation: int,
    walls:      grid.Grid2D(u8),
    all_units:  []units.Unit,
    goal:       units.Goal,
    has_goal:   bool,
    heat:       conv.HeatSolver,
    source:     grid.Grid2D(f32),
    flow:       nav.FlowField,
    map_tex:    rl.Texture2D,
    heat_img:   rl.Image,
    heat_tex:   rl.Texture2D,
}

simulation_make :: proc(dims: [2]int, generation: int) -> Simulation {
    s := Simulation {
        dims       = dims,
        generation = generation,
        heat_img   = rl.GenImageColor(cast(i32)dims[0], cast(i32)dims[1], rl.BLACK),
    }
    s.heat_tex = rl.LoadTextureFromImage(s.heat_img)
    rl.SetTextureFilter(s.heat_tex, .BILINEAR)

    simulation_reset(&s)
    return s
}

simulation_reset :: proc(s: ^Simulation) {
    seed := SEED + s.generation

    s.walls = maps.cellular_automata(s.dims, seed)
    s.all_units = units.sample(s.walls, UNIT_COUNT, cast(u64)(seed))
    s.goal, s.has_goal = units.sample_goal(s.walls, cast(u64)(seed) + 1)

    s.heat = conv.heat(
        s.walls,
        HEAT_RATE,
        HEAT_DECAY,
        HEAT_SOURCE_RADIUS,
        HEAT_SOURCE_STRENGTH,
    )

    heat_goal := conv.Goal{}
    if s.has_goal {
        heat_goal = conv.Goal {
            pos    = {f32(s.goal.pos[0]) + 0.5, f32(s.goal.pos[1]) + 0.5},
            weight = 1,
        }
    }

    s.source = conv.make_source(
        s.walls,
        heat_goal,
        HEAT_SOURCE_RADIUS,
        HEAT_SOURCE_STRENGTH,
    )
    conv.seed_source(&s.heat, s.source)

    s.flow = nav.make(s.walls)
    nav.update(&s.flow, &s.heat)

    if s.map_tex.id != 0 {
        rl.UnloadTexture(s.map_tex)
    }
    s.map_tex = map_texture(s.walls)
}

simulation_regenerate :: proc(s: ^Simulation) {
    s.generation += 1
    simulation_reset(s)
}

simulation_step :: proc(s: ^Simulation, dt: f32) {
    conv.step_with_source(&s.heat, s.source, HEAT_SUBSTEPS)
    nav.update(&s.flow, &s.heat)
    update_units(s.all_units, s.walls, &s.flow, s.goal, s.has_goal, dt)
}

simulation_update_heat_texture :: proc(s: ^Simulation) {
    update_heat_texture(s.heat_tex, &s.heat_img, &s.heat)
}

simulation_draw :: proc(s: ^Simulation, visualization: Visualization, camera: rl.Camera2D) {
    rl.BeginMode2D(camera)
    rl.DrawTextureEx(s.map_tex, {0, 0}, 0, 1, rl.WHITE)
    if visualization == .HEAT {
        rl.DrawTextureEx(s.heat_tex, {0, 0}, 0, 1, rl.WHITE)
    }

    unit_size := max(1, 2 / camera.zoom)
    for unit in s.all_units {
        pos := [2]f32{
            unit.pos[0] - unit_size / 2,
            unit.pos[1] - unit_size / 2,
        }
        rl.DrawRectangleV(pos, {unit_size, unit_size}, rl.RED)
    }

    if s.has_goal {
        goal_size := max(1, 4 / camera.zoom)
        goal_pos := [2]f32{
            f32(s.goal.pos[0]) + 0.5 - goal_size / 2,
            f32(s.goal.pos[1]) + 0.5 - goal_size / 2,
        }
        rl.DrawRectangleV(goal_pos, {goal_size, goal_size}, rl.GREEN)
    }
    rl.EndMode2D()
}
