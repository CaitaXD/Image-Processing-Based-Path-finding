package simulation

import "core:math"
import "core:testing"
import "core:fmt"
import "../conv"
import "../grid"
import "../maps"
import "../nav"
import cv "../odin-opencv"

// Headless construction of the pipeline's images: no raylib textures, but the
// same navmap/heat/goal/units the application builds in make()/reset().
@(private)
pipeline_headless :: proc(seed: u64) -> Simulation {
	s: Simulation
	pipeline_headless_fill(&s, seed)
	return s
}

@(private)
pipeline_headless_fill :: proc(s: ^Simulation, seed: u64) {
	s.dims = {W, H}
	s.generation = 0
	s.navmap = maps.cellular_automata(s.dims, seed)
	s.heat = conv.heat(
		s.navmap,
		HEAT_RATE,
		HEAT_DECAY,
		HEAT_SOURCE_RADIUS,
		HEAT_SOURCE_STRENGTH,
	)

	comb := spawn_comb(s.navmap, seed, UNIT_COUNT, SPAWN_FRACTION)
	goal := conv.Goal {
		pos    = goal_cell(s.navmap, comb, seed, GOAL_SAMPLES),
		weight = 1,
	}
	s.goal = conv.goal_image(s.navmap, goal, GOAL_RADIUS)
	s.units = comb
	next, err_next := grid.make(f32, s.navmap.width, s.navmap.height)
	assert(err_next == .None)
	s.units_next = next

	conv.seed_goal(&s.heat, goal)

	s.flow = nav.make(s.navmap)
	pipeline_setup(s)
	pipeline_prebake(s)
	nav.update(&s.flow, &s.heat)
}

@(private)
unit_mass :: proc(units: grid.Grid2D(f32)) -> f32 {
	total := f32(0)
	for value in units.data {
		total += value
	}
	return total
}

// unit_centroid returns where the comb is and how much mass it carries.
@(private)
unit_centroid :: proc(units: grid.Grid2D(f32)) -> ([2]f32, f32) {
	total := f32(0)
	sum := [2]f32{0, 0}
	for y in 0..<units.height {
		for x in 0..<units.width {
			mass := units.data[x + y * units.width]
			if mass <= 0 {
				continue
			}
			total += mass
			sum[0] += f32(x) * mass
			sum[1] += f32(y) * mass
		}
	}
	if total <= 0 {
		return {}, 0
	}
	return {sum[0] / total, sum[1] / total}, total
}

@(private)
unit_support :: proc(units: grid.Grid2D(f32)) -> int {
	n := 0
	for value in units.data {
		if value > 0 do n += 1
	}
	return n
}

// goal_mass is the amount of unit mass sitting on the goal splat.
@(private)
unit_goal_mass :: proc(units, goal: grid.Grid2D(f32)) -> f32 {
	total := f32(0)
	for i in 0..<len(units.data) {
		if goal.data[i] > 0 do total += units.data[i]
	}
	return total
}

// goal_centroid is the centre of the goal splat (the destination the comb
// should travel toward).
@(private)
goal_centroid :: proc(goal: grid.Grid2D(f32)) -> ([2]f32, int) {
	sum := [2]f32{0, 0}
	n := 0
	for y in 0..<goal.height {
		for x in 0..<goal.width {
			if goal.data[x + y * goal.width] > 0 {
				sum[0] += f32(x)
				sum[1] += f32(y)
				n += 1
			}
		}
	}
	if n == 0 {
		return {}, 0
	}
	return {sum[0] / f32(n), sum[1] / f32(n)}, n
}

@(private)
distance :: proc(a, b: [2]f32) -> f32 {
	dx := a[0] - b[0]
	dy := a[1] - b[1]
	return math.sqrt(dx * dx + dy * dy)
}

// The OpenCV heat step (filter2D + divide by coverage + wall mask + source) must
// reproduce the native conv step exactly; this is the check that the wrapper and
// the kernel/coverage images are wired correctly.
@(test)
pipeline_heat_matches_the_native_step :: proc(t: ^testing.T) {
	s: Simulation
	s.dims = {W, H}
	s.navmap = maps.cellular_automata(s.dims, SEED)
	s.heat = conv.heat(s.navmap, HEAT_RATE, HEAT_DECAY, HEAT_SOURCE_RADIUS, HEAT_SOURCE_STRENGTH)
	comb := spawn_comb(s.navmap, SEED, UNIT_COUNT, SPAWN_FRACTION)
	goal := conv.Goal {
		pos    = goal_cell(s.navmap, comb, SEED, GOAL_SAMPLES),
		weight = 1,
	}
	s.goal = conv.goal_image(s.navmap, goal, GOAL_RADIUS)
	s.units = comb
	next, err_next := grid.make(f32, W, H)
	assert(err_next == .None)
	s.units_next = next
	conv.seed_goal(&s.heat, goal)
	s.flow = nav.make(s.navmap)
	pipeline_setup(&s)

	pipeline_masks(&s)
	pipeline_heat(&s)

	native, err_native := grid.make(f32, W, H)
	assert(err_native == .None)
	copy(native.data, s.heat.field.data)

	conv.seed_goal(&s.heat, goal)
	conv.step_with_source(&s.heat, s.goal, HEAT_SUBSTEPS)

	diff := f32(0)
	for i in 0..<len(native.data) {
		d := math.abs(native.data[i] - s.heat.field.data[i])
		if d > diff do diff = d
	}
	fmt.printf("heat step: max difference pipeline vs native {:.3e}\n", diff)
	testing.expect(t, diff < 1e-4, "the OpenCV heat step must match the native step")
}

@(test)
pipeline_keeps_the_images_alive :: proc(t: ^testing.T) {
	s: Simulation
	pipeline_headless_fill(&s, SEED)

	// One frame with OpenCV: the heat image must stay finite, the flow field
	// must be non-zero wherever the map is walkable, and the masks must agree
	// with the navmap.
	pipeline_step(&s)

	finite := true
	walkable := 0
	flow_nonzero := 0
	for i in 0..<len(s.heat.field.data) {
		value := s.heat.field.data[i]
		if value != value || value > 3.0e38 || value < -3.0e38 {
			finite = false
		}
		if s.navmap.data[i] != maps.WALKABLE {
			continue
		}
		walkable += 1
		dir := s.flow.field.data[i]
		if dir[0] != 0 || dir[1] != 0 {
			flow_nonzero += 1
		}
	}

	testing.expect(t, finite, "the heat image must stay finite")
	testing.expect(t, walkable > 0, "the map must have walkable cells")
	testing.expect(
		t,
		flow_nonzero > walkable / 2,
		"the Sobel field must cover most of the walkable map",
	)
	testing.expect(
		t,
		s.pipe.wall_mask.data[0] == 1 || s.pipe.wall_mask.data[0] == 0,
		"the wall mask must be binary",
	)
}

// The whole experiment: one image is thresholded into masks, diffused into a
// heat field, differentiated into a Sobel vector field, wall-masked and used to
// remap the unit image. The comb must advance toward the goal as a set of point
// masses (one cell each) without losing mass.
@(test)
pipeline_units_travel_toward_the_goal :: proc(t: ^testing.T) {
	s: Simulation
	pipeline_headless_fill(&s, SEED)

	start_centroid, start_mass := unit_centroid(s.units)
	destination, goal_cells := goal_centroid(s.goal)
	testing.expect(t, goal_cells > 0, "the goal image must have cells")

	start_distance := distance(start_centroid, destination)
	best_distance := start_distance
	arrived_frame := -1
	frames := 400
	for frame in 0..<frames {
		pipeline_step(&s)
		centroid, _ := unit_centroid(s.units)
		d := distance(centroid, destination)
		if d < best_distance do best_distance = d
		if arrived_frame < 0 && unit_goal_mass(s.units, s.goal) > 0 {
			arrived_frame = frame
		}
	}

	end_centroid, end_mass := unit_centroid(s.units)
	arrived_mass := unit_goal_mass(s.units, s.goal)
	support := unit_support(s.units)
	fmt.printf(
		"pipeline: start {:.1f}, best {:.1f}, end {:.1f} cells; mass {:.2f} -> {:.2f}; support {}; arrival frame {}; mass on goal {:.2f}\n",
		start_distance,
		best_distance,
		distance(end_centroid, destination),
		start_mass,
		end_mass,
		support,
		arrived_frame,
		arrived_mass,
	)

	// The scatter remap is injective: every unit stays a single cell of mass 1
	// and the total mass is conserved exactly.
	testing.expect(t, support == UNIT_COUNT, "every unit must remain a point mass")
	testing.expect(t, math.abs(end_mass - start_mass) < 0.01, "unit mass must be conserved")
	testing.expect(
		t,
		best_distance < start_distance * 0.5,
		"the unit image must travel most of the way to the goal",
	)
	testing.expect(t, arrived_mass > 0, "some unit mass must reach the goal splat")
}

// Follows the vector field from every walkable cell and counts how many walks
// reach the goal splat. This measures the field itself, independently of where
// the comb happens to start.
@(test)
pipeline_flow_basin :: proc(t: ^testing.T) {
	s: Simulation
	pipeline_headless_fill(&s, SEED)
	pipeline_step(&s)

	w := s.navmap.width
	h := s.navmap.height
	arrived := 0
	started := 0
	for y in 1..<h - 1 {
		for x in 1..<w - 1 {
			start := x + y * w
			if s.navmap.data[start] != maps.WALKABLE {
				continue
			}
			if s.goal.data[start] > 0 {
				arrived += 1
				continue
			}
			started += 1
			fx := f32(x)
			fy := f32(y)
			ok := false
			for _ in 0..<800 {
				cx := int(math.round(fx))
				cy := int(math.round(fy))
				if cx < 0 || cy < 0 || cx >= w || cy >= h {
					break
				}
				cell := cx + cy * w
				if s.goal.data[cell] > 0 {
					ok = true
					break
				}
				dir := s.flow.field.data[cell]
				if dir[0] == 0 && dir[1] == 0 do break
				fx += dir[0]
				fy += dir[1]
			}
			if ok do arrived += 1
		}
	}
	fmt.printf("flow basin: {} of {} walkable cells reach the goal\n", arrived, started)
	testing.expect(
		t,
		arrived * 100 > started,
		"the field must lead at least 1% of the walkable map to the goal",
	)
}

@(test)
pipeline_resample_moves_one_cell :: proc(t: ^testing.T) {
	size := 7
	units, err_units := grid.make(f32, size, size)
	scratch, err_scratch := grid.make(f32, size, size)
	navmap, err_map := grid.make(u8, size, size)
	flow_grid, err_flow := grid.make([2]f32, size, size)
	map_x := cv.new_mat(f32, size, size)
	map_y := cv.new_mat(f32, size, size)
	assert(
		err_units == .None &&
		err_scratch == .None &&
		err_map == .None &&
		err_flow == .None,
	)

	for i in 0..<len(navmap.data) {
		navmap.data[i] = maps.WALKABLE
		flow_grid.data[i] = {1, 0}
	}
	flow := nav.FlowField {
		navmap = navmap,
		field  = flow_grid,
	}

	units.data[3 + 3 * size] = 1
	resample_units(&units, &scratch, navmap, &flow, &map_x, &map_y)

	testing.expect(t, units.data[4 + 3 * size] == 1, "the unit must move one cell along the flow")
	testing.expect(t, units.data[3 + 3 * size] == 0, "the unit must leave its old cell empty")
	testing.expect(t, unit_mass(units) == 1, "the moved unit must keep its mass")
}
