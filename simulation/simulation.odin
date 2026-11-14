package simulation

import "core:math"
import "../conv"
import "../grid"
import "../maps"
import "../nav"
import "../heat"
import rl "vendor:raylib"

W :: 400
H :: 400

SEED                 :: u64(42)
UNIT_COUNT           :: 100
HEAT_SUBSTEPS        :: 2
// HEAT_PREBAKE is how many substeps the field is relaxed before the first
// frame. A fresh wash is a free-space Green's function, so it ignores the
// walls and its Sobel points in a straight line; relaxing it in place turns it
// into the wall-aware steady state of the same operator, and that field has a
// single maximum at the goal. The prebake is paid once, on reset.
HEAT_PREBAKE         :: 5000
HEAT_RATE            :: 0.5
// The decay is what makes the steady state well conditioned. It is also the screening length:
// lambda = sqrt(rate / (4 * decay)) cells, so 2e-4 gives an e-folding of about 25 cells and
// 2e-6 gives about 250 - enough for the potential to stay readable across the whole map, whose
// geodesic diameter is 641 cells on seed 42. Reach comes from the seeding, not from iterating:
// reset splats the goal at half the map, so the gaussian lands on every walkable cell at frame
// 0 and the flow field has a gradient everywhere from the start (the seeded field's minimum over
// the 112,627 walkable cells was 5.6e-01). The only attempt at front-loading agreement instead,
// pyramid.odin, measured 1.4e9 cell touches against 9e8 for a flat native sweep to a 5e-5
// residual, so it is kept on file as a negative result and the application never calls it.
HEAT_DECAY           :: 0.000002
HEAT_SOURCE_RADIUS   :: f32(max(W, H))/2
HEAT_SOURCE_STRENGTH :: 1.0
GOAL_RADIUS          :: 1.0
SPAWN_FRACTION       :: 0.2
GOAL_SAMPLES         :: 128
UNIT_SPEED           :: 12.0
MAX_MOVE_STEP        :: 0.25

FPS                  :: 24
SIM_DT               :: f32(1.0 / FPS)

UNIT_MARKER_PX :: 6
GOAL_MARKER_PX :: 10

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
    heat:        conv.HeatSolver,
    flow:        nav.FlowField,
    navmap:      grid.Grid2D(u8),
    units:       grid.Grid2D(f32),
    units_next:  grid.Grid2D(f32),
    goal:        grid.Grid2D(f32),
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

    // rate/decay build the per-cell operator; the two source constants are the goal splat the
    // heat reads at frame 0 - conv.seed_goal writes the goal through conv.add_source at exactly
    // these, into the solver's own scratch buffer, and seeds from it (see conv/goal.odin).
    s.heat = conv.heat(
        s.navmap,
        HEAT_RATE,
        HEAT_DECAY,
        HEAT_SOURCE_RADIUS,
        HEAT_SOURCE_STRENGTH,
    )

    comb := spawn_comb(s.navmap, seed, UNIT_COUNT, SPAWN_FRACTION)

    goal := conv.Goal{
        pos    = goal_cell(s.navmap, comb, seed, GOAL_SAMPLES),
        weight = 1,
    }
    s.goal = conv.goal_image(s.navmap, goal, GOAL_RADIUS)
    s.units = comb
    s.units_next = s.units
    next, err_next := grid.make(f32, s.navmap.width, s.navmap.height)
    assert(err_next == .None)

    // The heat source is the goal, splatted: conv keeps the splat's width as source_radius
    // (HEAT_SOURCE_RADIUS is half the map, so that wash covers the field and the flow field
    // exists everywhere at frame 0), while the splat the step re-injects and the units
    // sink into is GOAL_RADIUS wide. The wash is written through the solver's own scratch
    // buffer, so seeding allocates nothing of its own.
    conv.seed_goal(&s.heat, goal)

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
spawn_comb :: proc(navmap: grid.Grid2D(u8), seed: u64, count: int, spawn_fraction: f32) -> grid.Grid2D(f32) {
    comb, err := grid.make(f32, navmap.width, navmap.height)
    assert(err == .None)

    band := max(1, int(f32(navmap.width) * spawn_fraction))
    state := seed
    placed := 0
    for attempts := 0; placed < count && attempts < count * 200; attempts += 1 {
        x := pick(&state, band)
        y := pick(&state, navmap.height)
        if grid.at(navmap, x, y)^ != maps.WALKABLE {
            continue
        }
        index := x + y * navmap.width
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
goal_cell :: proc(navmap: grid.Grid2D(u8), units: grid.Grid2D(f32), seed: u64, samples: int) -> [2]f32 {
    dist, err_dist := grid.make(int, navmap.width, navmap.height, context.temp_allocator)
    assert(err_dist == .None)
    for i in 0..<len(dist.data) {
        dist.data[i] = -1
    }

    // Each cell is pushed once, so the flat queue never needs more room than the image.
    queue, err_queue := grid.make(int, navmap.width, navmap.height, context.temp_allocator)
    assert(err_queue == .None)

    tail := 0
    for i in 0..<len(units.data) {
        if units.data[i] <= 0 {
            continue
        }
        dist.data[i] = 0
        queue.data[tail] = i
        tail += 1
    }

    head := 0
    for head < tail {
        i := queue.data[head]
        head += 1
        x := i % navmap.width
        y := i / navmap.width
        d := dist.data[i] + 1

        if x > 0 && dist.data[i - 1] == -1 && navmap.data[i - 1] == maps.WALKABLE {
            dist.data[i - 1] = d
            queue.data[tail] = i - 1
            tail += 1
        }
        if x + 1 < navmap.width && dist.data[i + 1] == -1 && navmap.data[i + 1] == maps.WALKABLE {
            dist.data[i + 1] = d
            queue.data[tail] = i + 1
            tail += 1
        }
        if y > 0 && dist.data[i - navmap.width] == -1 && navmap.data[i - navmap.width] == maps.WALKABLE {
            dist.data[i - navmap.width] = d
            queue.data[tail] = i - navmap.width
            tail += 1
        }
        if y + 1 < navmap.height && dist.data[i + navmap.width] == -1 && navmap.data[i + navmap.width] == maps.WALKABLE {
            dist.data[i + navmap.width] = d
            queue.data[tail] = i + navmap.width
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
        x := pick(&state, navmap.width)
        y := pick(&state, navmap.height)
        index := x + y * navmap.width
        if dist.data[index] <= 0 {
            continue
        }
        if !found || dist.data[index] > best_d {
            found = true
            best_d = dist.data[index]
            best_x = x
            best_y = y
        }
    }
    assert(found, "goal_cell: a reachable cell clear of the spawns must exist")

    return {f32(best_x) + 0.5, f32(best_y) + 0.5}
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
draw_markers :: proc(img: grid.Grid2D(f32), size: f32, color: rl.Color) {
    for y in 0..<img.height {
        for x in 0..<img.width {
            if img.data[x + y * img.width] <= 0 {
                continue
            }
            corner := [2]f32{f32(x) + 0.5, f32(y) + 0.5} - {size / 2, size / 2}
            rl.DrawRectangleV(corner, {size, size}, color)
        }
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

draw_flow_field :: proc(s: ^Simulation, camera: rl.Camera2D) {
    thickness := max(0.5, FLOW_ARROW_PX / camera.zoom)

    for y := FLOW_STRIDE / 2; y < s.flow.field.height; y += FLOW_STRIDE {
        for x := FLOW_STRIDE / 2; x < s.flow.field.width; x += FLOW_STRIDE {
            dir := grid.at(s.flow.field, x, y)^
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
