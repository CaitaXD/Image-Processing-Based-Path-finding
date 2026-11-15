package simulation

import "core:math"
import "../heat"
import "../maps"
import "../nav"
import cv "../odin-opencv"
import rl "vendor:raylib"

W :: 400
H :: 400

FPS                  :: 24
SIM_DT               :: f32(1.0 / FPS)

UNIT_MARKER_PX :: 6
GOAL_MARKER_PX :: 10

SEED                 :: u64(42)
UNIT_COUNT           :: 100
HEAT_SUBSTEPS        :: 2
HEAT_PREBAKE         :: 5000
HEAT_RATE            :: 0.5
HEAT_DECAY           :: 0.000002
HEAT_SOURCE_RADIUS   :: f32(max(W, H))/2
HEAT_SOURCE_STRENGTH :: 1.0
GOAL_RADIUS          :: 1.0
SPAWN_FRACTION       :: 0.2
GOAL_SAMPLES         :: 128
UNIT_SPEED           :: 12.0
MAX_MOVE_STEP        :: 0.25

FLOW_STRIDE       :: 8
FLOW_ARROW_LENGTH :: 3.0
FLOW_ARROW_PX     :: 2.0
FLOW_ARROW_HEAD   :: 0.35

Visualization :: enum {
    NONE,
    HEAT,
    FLOW,
}

VISUALIZATIONS  :: [?]Visualization{.NONE, .HEAT, .FLOW}

Simulation :: struct {
    generation:  u64,
    dims:        [2]int,
    heat:        heat.HeatSolver,
    flow:        nav.FlowField,
    navmap:      cv.Mat(f32),
    units:       cv.Mat(f32),
    units_next:  cv.Mat(f32),
    goal:        cv.Mat(f32),
    map_tex:     rl.Texture2D,
    heat_img:    rl.Image,
    heat_tex:    rl.Texture2D,
    pipe:        ^Pipeline,
}

make :: proc(dims: [2]int, generation: u64) -> Simulation {
    s := Simulation {
        dims        = dims,
        generation  = generation,
        heat_img    = rl.GenImageColor(cast(i32)dims[0], cast(i32)dims[1], rl.BLACK),
    }
    s.heat_tex = rl.LoadTextureFromImage(s.heat_img)
    rl.SetTextureFilter(s.heat_tex, .BILINEAR)

    reset(&s)
    return s
}

reset :: proc(s: ^Simulation) {
    seed := SEED + s.generation

    s.navmap = maps.cellular_automata(s.dims, seed)

    // rate/decay build the per-cell operator. The source image the heat reads at frame 0 is
    // built on reset below, since the solver only ever takes images (see heat/solver.odin).
    s.heat = heat.solver(s.navmap, HEAT_RATE, HEAT_DECAY)

    comb := spawn_comb(s.navmap, seed, UNIT_COUNT, SPAWN_FRACTION)

    goal_pos := goal_cell(s.navmap, comb, seed, GOAL_SAMPLES)
    s.goal = heat.splat_goal_disc(s.navmap, goal_pos, GOAL_RADIUS)
    s.units = comb
    units_next, err_units_next := cv.mat_new(f32, int(comb.rows), int(comb.cols))
    assert(err_units_next == .None)
    s.units_next = units_next

    // The heat source is an image: the goal splatted at HEAT_SOURCE_RADIUS (half the map,
    // so the wash covers the field and the flow field has a gradient everywhere at frame 0)
    // through heat.splat_heat_source, then seeded with heat.seed_source. The image the step
    // re-injects and the units sink into is the GOAL_RADIUS-wide one above, sampled from the
    // same goal cell, so nothing downstream carries a goal position.
    wash := heat.splat_heat_source(s.navmap, goal_pos, HEAT_SOURCE_RADIUS, HEAT_SOURCE_STRENGTH)
    heat.seed_source(&s.heat, wash)

    s.flow = nav.make(s.navmap)
    pipeline_setup(s)
    pipeline_prebake(s)
    nav.update(&s.flow, &s.heat)

    if s.map_tex.id != 0 {
        rl.UnloadTexture(s.map_tex)
    }
    s.map_tex = map_texture(s.navmap)
}

regenerate :: proc(s: ^Simulation) {
    s.generation += 1
    reset(s)
}

// Splitmix64 over (SEED + generation): the layout is a pure function of the seed, so a
// recorded run replays exactly and R only moves the generation counter.
next_u64 :: proc(state: ^u64) -> u64 {
    state^ += 0x9E3779B97F4A7C15
    a := state^
    b := (a ~ (a >> 30)) * 0xBF58476D1CE4E5B9
    c := (b ~ (b >> 27)) * 0x94D049BB133111EB
    return (c) ~ (c >> 31)
}

pick :: proc(state: ^u64, n: int) -> int {
    assert(n > 0)
    return int(next_u64(state) % cast(u64)(n))
}

// Units stay an image, like everything that moves on the map: a Dirac comb, 1 at one
// walkable cell per unit inside the spawn band on the left of the map, 0 elsewhere.
// step advects that very image through the flow field (advect_units), so its support
// is where the units are this frame and its total is the unit count every frame -
// conservation reads as "no unit was ever lost".
spawn_comb :: proc(navmap: cv.Mat(f32), seed: u64, count: int, spawn_fraction: f32) -> cv.Mat(f32) {
    comb, err := cv.mat_new(f32, int(navmap.rows), int(navmap.cols))
    assert(err == .None)

    band := max(1, int(f32(navmap.cols) * spawn_fraction))
    state := seed
    placed := 0
    for attempts := 0; placed < count && attempts < count * 200; attempts += 1 {
        x := pick(&state, band)
        y := pick(&state, int(navmap.rows))
        if navmap.data[x + y * int(navmap.cols)] != maps.WALKABLE {
            continue
        }
        index := x + y * int(navmap.cols)
        if comb.data[index] != 0 {
            continue
        }
        comb.data[index] = 1
        placed += 1
    }
    assert(placed > 0, "spawn_comb: the spawn band must hold at least one walkable cell")

    return comb
}

// Geodesic 4-connected distance from the comb, -1 at walls and at cells the units cannot
// reach. That connectivity is the one the heat stencil, nav.update, walkable_cell and the
// advection corner guard share, so a goal outside the reachable set is one the units
// could never advect to: the goal splat would stay empty for the whole run.
goal_cell :: proc(navmap: cv.Mat(f32), units: cv.Mat(f32), seed: u64, samples: int) -> [2]int {
    w := int(navmap.cols)
    h := int(navmap.rows)

    dist, err_dist := cv.mat_new(i32, h, w, context.temp_allocator)
    assert(err_dist == .None)
    for i in 0..<w * h {
        dist.data[i] = -1
    }

    // Each cell is pushed once, so the flat queue never needs more room than the image.
    queue, err_queue := cv.mat_new(i32, h, w, context.temp_allocator)
    assert(err_queue == .None)

    tail := 0
    for i in 0..<w * h {
        if units.data[i] <= 0 {
            continue
        }
        dist.data[i] = 0
        queue.data[tail] = i32(i)
        tail += 1
    }

    head := 0
    for head < tail {
        i := int(queue.data[head])
        head += 1
        x := i % w
        y := i / w
        d := int(dist.data[i]) + 1

        if x > 0 && dist.data[i - 1] == -1 && navmap.data[i - 1] == maps.WALKABLE {
            dist.data[i - 1] = i32(d)
            queue.data[tail] = i32(i) - 1
            tail += 1
        }
        if x + 1 < w && dist.data[i + 1] == -1 && navmap.data[i + 1] == maps.WALKABLE {
            dist.data[i + 1] = i32(d)
            queue.data[tail] = i32(i) + 1
            tail += 1
        }
        if y > 0 && dist.data[i - w] == -1 && navmap.data[i - w] == maps.WALKABLE {
            dist.data[i - w] = i32(d)
            queue.data[tail] = i32(i - w)
            tail += 1
        }
        if y + 1 < h && dist.data[i + w] == -1 && navmap.data[i + w] == maps.WALKABLE {
            dist.data[i + w] = i32(d)
            queue.data[tail] = i32(i + w)
            tail += 1
        }
    }

    // Farthest-point sample: the random candidates only decide which equally far cell
    // wins, so the goal is always as far from the spawns as a sampled cell can get.
    state := seed
    best_x := 0
    best_y := 0
    best_d: int = 0
    found := false
    for _ in 0..<samples * 16 {
        x := pick(&state, w)
        y := pick(&state, h)
        index := x + y * w
        if dist.data[index] <= 0 {
            continue
        }
        if !found || int(dist.data[index]) > best_d {
            found = true
            best_d = int(dist.data[index])
            best_x = x
            best_y = y
        }
    }
    assert(found, "goal_cell: a reachable cell clear of the spawns must exist")

    return {best_x, best_y}
}

step :: proc(s: ^Simulation, dt: f32) {
    // One frame of Experiment 1: OpenCV masks, heat diffusion, Sobel vector
    // field, wall masking and cv2.remap of the unit image (see pipeline.odin).
    pipeline_step(s)
}

update_heat :: proc(s: ^Simulation) {
    heat.update_texture(s.heat_tex, &s.heat_img, &s.heat)
}

// Grid advection of the units, one hop per frame on the very grid the flow field lives
// on: each unit moves to the neighbour cell the flow vector at its own cell points at
// best, walls only close a cell so a unit never waits for an open downstream - it takes
// the best aligned open neighbour or holds if none aligns, a diagonal never squeezes
// past wall corners (the same guard the old mover's cuts_corner kept), and a unit
// standing on the goal splat has arrived and stays put. Mass is only ever moved, never
// created nor damped, so the comb's total stays the unit count every frame and its
// support is where the units are this frame.
draw :: proc(s: ^Simulation, visualization: Visualization, camera: rl.Camera2D) {
    rl.BeginMode2D(camera)
    rl.DrawTextureEx(s.map_tex, {0, 0}, 0, 1, rl.WHITE)
    switch visualization {
    case .HEAT:
        rl.DrawTextureEx(s.heat_tex, {0, 0}, 0, 1, rl.WHITE)
    case .FLOW:
        draw_flow_field(s, camera)
    case .NONE:
    }

    unit_size := max(1, UNIT_MARKER_PX / camera.zoom)
    goal_size := max(1, GOAL_MARKER_PX / camera.zoom)
    draw_markers(s.units, unit_size, rl.RED)
    draw_markers(s.goal, goal_size, rl.BLUE)
    rl.EndMode2D()
}

// Markers are the splat read backwards: an image's support rasterized back into screen
// squares, so the display needs no sampling coordinates either and every view shows the
// same units and goal the operators act on.
draw_markers :: proc(img: cv.Mat(f32), size: f32, color: rl.Color) {
    for y in 0..<int(img.rows) {
        for x in 0..<int(img.cols) {
            if img.data[x + y * int(img.cols)] <= 0 {
                continue
            }
            corner := [2]f32{f32(x) + 0.5, f32(y) + 0.5} - {size / 2, size / 2}
            rl.DrawRectangleV(corner, {size, size}, color)
        }
    }
}

map_texture :: proc(a: cv.Mat(f32)) -> rl.Texture2D {
    img := rl.GenImageColor(cast(i32)a.cols, cast(i32)a.rows, rl.BLACK)
    pixels := cast([^]rl.Color)img.data
    for y in 0..<int(a.rows) {
        for x in 0..<int(a.cols) {
            pixels[x + y * int(a.cols)] = a.data[x + y * int(a.cols)] == maps.WALKABLE ? rl.GRAY : rl.BLACK
        }
    }
    tex := rl.LoadTextureFromImage(img)
    rl.SetTextureFilter(tex, .POINT)
    rl.UnloadImage(img)
    return tex
}

draw_flow_field :: proc(s: ^Simulation, camera: rl.Camera2D) {
    thickness := max(0.5, FLOW_ARROW_PX / camera.zoom)

    for y := FLOW_STRIDE / 2; y < int(s.flow.field.rows); y += FLOW_STRIDE {
        for x := FLOW_STRIDE / 2; x < int(s.flow.field.cols); x += FLOW_STRIDE {
            dir := cv.at(&s.flow.field, x, y)^
            if dir[0] == 0 && dir[1] == 0 {
                continue
            }

            start := [2]f32{f32(x) + 0.5, f32(y) + 0.5}
            tip := start + dir * FLOW_ARROW_LENGTH

            angle := math.atan2(dir[1], dir[0])
            hue := (angle + f32(math.PI)) / (2 * f32(math.PI)) * 360
            color := rl.ColorFromHSV(hue, 0.85, 1.0)

            rl.DrawLineEx(start, tip, thickness, color)

            head_len := f32(FLOW_ARROW_LENGTH * FLOW_ARROW_HEAD)
            turn := f32(math.PI) * 5 / 6
            head_a := tip + [2]f32{
                math.cos(angle + turn) * head_len,
                math.sin(angle + turn) * head_len,
            }
            head_b := tip + [2]f32{
                math.cos(angle - turn) * head_len,
                math.sin(angle - turn) * head_len,
            }
            rl.DrawLineEx(tip, head_a, thickness, color)
            rl.DrawLineEx(tip, head_b, thickness, color)
        }
    }
}
