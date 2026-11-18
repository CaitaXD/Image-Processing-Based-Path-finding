package simulation

import cv "../odin-opencv"
import "core:math"
import "../heat"
import "../maps"
import "../nav"

Pipeline :: struct {
	board:      cv.Mat(u8),
	walkable:   cv.Mat(u8),
	beyond:     cv.Mat(u8),
	goal_inv:   cv.Mat(u8),
	units_mask: cv.Mat(u8),
	goal_mask:  cv.Mat(u8),

	// wall_mask is 1 in a wall and walk is its inverse, so a shifted read of
	// walk answers "is the destination of this move inside the grid and open".
	wall_mask: cv.Mat(f32),
	walk:      cv.Mat(f32),

	tmp: cv.Mat(f32), // general scratch; replaces the dedicated `weighted`

	// Coordinate maps for the remap step (claim protocol).
	map_x: cv.Mat(f32),
	map_y: cv.Mat(f32),

	// The nine moves a cell can make, in the order of move_dirs: the eight
	// neighbours and staying put. Each image holds 1 where the cell takes that
	// whole step.
	weights: [9]cv.Mat(f32),

	// open_axis[i] is 1 where the destination of the i-th axis move (east,
	// west, south, north) is inside the grid and not a wall. The diagonal
	// open masks are computed on the fly into tmp.
	open_axis: [4]cv.Mat(f32),

	// The four signs of the flow and whether each axis moves at all.
	ax_pos, ax_neg, ay_pos, ay_neg: cv.Mat(f32),
	ax, ay, not_ax, not_ay:         cv.Mat(f32),

	// Holds `moved` (1 where a full move is taken) through step 4, then
	// `free` (its complement) from step 5 onward; `moved` is not read again.
	moved: cv.Mat(f32),

	// Scratch for the slide steps.
	free_no_x:  cv.Mat(f32),
	vert:   cv.Mat(f32),
	horiz:  cv.Mat(f32),
}

// move_dirs is the (dx, dy) of each image in Pipeline.weights: east, west,
// south, north, the four diagonals, then no move at all.
move_dirs :: [9][2]i32{
	{1, 0}, {-1, 0}, {0, 1}, {0, -1},
	{1, 1}, {-1, 1}, {1, -1}, {-1, -1},
	{0, 0},
}

// pipeline_alloc allocates the images a w by h pipeline works on. units and
// scratch are left out because they belong to the simulation, and
// pipeline_setup points this pipeline at them.
pipeline_alloc :: proc(p: ^Pipeline, w, h: int) {
	p.board = cv.mat_new(u8, h, w)
	p.walkable = cv.mat_new(u8, h, w)
	p.beyond = cv.mat_new(u8, h, w)
	p.goal_inv = cv.mat_new(u8, h, w)
	p.units_mask = cv.mat_new(u8, h, w)
	p.goal_mask = cv.mat_new(u8, h, w)
	p.wall_mask = cv.mat_new(f32, h, w)
	p.walk = cv.mat_new(f32, h, w)

	for i in 0..<9 {
		p.weights[i] = cv.mat_new(f32, h, w)
	}
	for i in 0..<4 {
		p.open_axis[i] = cv.mat_new(f32, h, w)
	}

	p.ax_pos = cv.mat_new(f32, h, w)
	p.ax_neg = cv.mat_new(f32, h, w)
	p.ay_pos = cv.mat_new(f32, h, w)
	p.ay_neg = cv.mat_new(f32, h, w)
	p.ax = cv.mat_new(f32, h, w)
	p.ay = cv.mat_new(f32, h, w)
	p.not_ax = cv.mat_new(f32, h, w)
	p.not_ay = cv.mat_new(f32, h, w)
	p.moved = cv.mat_new(f32, h, w)
	p.free_no_x = cv.mat_new(f32, h, w)
	p.vert = cv.mat_new(f32, h, w)
	p.horiz = cv.mat_new(f32, h, w)
	p.tmp = cv.mat_new(f32, h, w)
	p.map_x = cv.mat_new(f32, h, w)
	p.map_y = cv.mat_new(f32, h, w)
}

pipeline_setup :: proc(s: ^Simulation) {
	if s.pipe == nil {
		s.pipe = new(Pipeline)
	}
	p := s.pipe
	pipeline_alloc(p, s.dims[0], s.dims[1])
}

// pipeline_walk derives the wall image and its inverse from a navmap. The unit
// step reads the inverse shifted, so it knows which destinations are open.
pipeline_walk :: proc(p: ^Pipeline, navmap: ^cv.Mat(f32)) {
	cv.threshold(navmap, &p.wall_mask, 0.5, 1, cv.THRESH_BINARY_INV)
	cv.threshold(&p.wall_mask, &p.walk, 0.5, 1, cv.THRESH_BINARY_INV)
}

pipeline_masks :: proc(s: ^Simulation) {
	p := s.pipe
	for i in 0..<int(p.board.rows * p.board.cols) {
		if s.navmap.data[i] != maps.WALKABLE {
			p.board.data[i] = 0
			continue
		}
		if s.goal.data[i] > 0 {
			p.board.data[i] = 3
			continue
		}
		if s.units.data[i] > 0 {
			p.board.data[i] = 2
			continue
		}
		p.board.data[i] = 1
	}

	cv.threshold(&p.board, &p.walkable, 0.0, 1, cv.THRESH_BINARY)
	cv.threshold(&p.board, &p.beyond, 1.0, 1, cv.THRESH_BINARY)
	cv.threshold(&p.board, &p.goal_mask, 2.0, 1, cv.THRESH_BINARY)
	cv.threshold(&p.board, &p.goal_inv, 2.0, 1, cv.THRESH_BINARY_INV)
	pipeline_walk(p, &s.navmap)
	cv.and(&p.beyond, &p.goal_inv, &p.units_mask)
}

pipeline_heat :: proc(s: ^Simulation) {
	heat.step(&s.heat, &s.goal, HEAT_SUBSTEPS)
}

pipeline_prebake :: proc(s: ^Simulation) {
	pipeline_masks(s)
	heat.step(&s.heat, &s.goal, HEAT_PREBAKE)
}

pipeline_flow :: proc(s: ^Simulation) {
	nav.update(&s.flow, &s.heat)
}

// resample_units moves the unit image one whole cell along the flow, which is
// the advection: the flow is a unit vector, so thresholding each axis at a half
// turns it into the direction of the step, and the nine moves are the eight
// neighbours plus staying put.
//
// A move whose destination is a wall is never taken. The unit slides instead:
// the x axis of the wanted move is tried, then its y axis, then the open side
// of a dead end, and only a cell with no open neighbour at all stays put.
// Without the slide a unit that faces a wall freezes there for good, and the
// flow of a heat field that is zero inside the walls points into a wall in a
// good many cells.
resample_units :: proc(
	units: ^cv.Mat(f32),
	scratch: ^cv.Mat(f32),
	navmap: cv.Mat(f32),
	flow: ^nav.FlowField,
	map_x, map_y: ^cv.Mat(f32),
) {
	w := int(navmap.cols)
	h := int(navmap.rows)

	// Find a trash cell (a walkable cell for unclaimed destinations).
	trash := 0
	for i in 0..<w * h {
		if navmap.data[i] == maps.WALKABLE {
			trash = i
			break
		}
	}

	// Initialize maps to -1 (unclaimed).
	for i in 0..<int(map_x.rows * map_x.cols) {
		map_x.data[i] = -1
		map_y.data[i] = 0
	}

	// Every unit first claims its own cell (stay). A move overwrites the
	// destination cell's entry, so a unit only moves if its destination is
	// unclaimed. This makes the step injective: no two units share a cell.
	for i in 0..<w * h {
		if units.data[i] > 0 && navmap.data[i] == maps.WALKABLE {
			map_x.data[i] = f32(i % w)
			map_y.data[i] = f32(i / w)
		}
	}

	// For each unit, try moves in priority order.
	for y in 0..<h {
		for x in 0..<w {
			i := x + y * w
			if units.data[i] <= 0 || navmap.data[i] != maps.WALKABLE {
				continue
			}

			dir_x := flow.gx.data[i]
			dir_y := flow.gy.data[i]
			dir := [2]f32{dir_x, dir_y}

			// Quantize the flow to a whole-cell step.
			dx := 0
			dy := 0
			if dir[0] > 0.5 { dx = 1 } else if dir[0] < -0.5 { dx = -1 }
			if dir[1] > 0.5 { dy = 1 } else if dir[1] < -0.5 { dy = -1 }
			if dx == 0 && dy == 0 {
				if math.abs(dir[0]) >= math.abs(dir[1]) {
					if dir[0] > 0 { dx = 1 } else if dir[0] < 0 { dx = -1 }
				} else {
					if dir[1] > 0 { dy = 1 } else if dir[1] < 0 { dy = -1 }
				}
			}

			// Try candidates in priority order: full, x-only, y-only.
			dest := -1
			candidates := [3][2]int{{dx, dy}, {dx, 0}, {0, dy}}
			for c in candidates {
				jx := x + c[0]
				jy := y + c[1]
				if jx < 0 || jy < 0 || jx >= w || jy >= h {
					continue
				}
				j := jx + jy * w
				if navmap.data[j] != maps.WALKABLE {
					continue
				}
				if map_x.data[j] >= 0 {
					continue
				}
				dest = j
				break
			}

			// Perpendicular slide: the open axis neighbour best aligned with the flow.
			if dest < 0 {
				best := f32(-2)
				sides := [4][2]int{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}
				for c in sides {
					jx := x + c[0]
					jy := y + c[1]
					if jx < 0 || jy < 0 || jx >= w || jy >= h {
						continue
					}
					j := jx + jy * w
					if navmap.data[j] != maps.WALKABLE {
						continue
					}
					if map_x.data[j] >= 0 {
						continue
					}
					along := f32(c[0]) * dir[0] + f32(c[1]) * dir[1]
					if along > best {
						best = along
						dest = j
					}
				}
			}

			if dest < 0 {
				continue
			}

			// The unit leaves its own cell, so clear the stay entry there.
			if dest != i {
				map_x.data[i] = -1
				map_y.data[i] = 0
			}
			map_x.data[dest] = f32(x)
			map_y.data[dest] = f32(y)
		}
	}

	// Fill unclaimed destinations with trash.
	for i in 0..<int(map_x.rows * map_x.cols) {
		if map_x.data[i] < 0 {
			map_x.data[i] = f32(trash % w)
			map_y.data[i] = f32(trash / w)
		}
	}

	cv.resample(units, scratch, map_x, map_y, cv.INTER_NEAREST)
	copy(units.data[0:int(units.rows * units.cols)], scratch.data[0:int(scratch.rows * scratch.cols)])
}


pipeline_step :: proc(s: ^Simulation, units: ^cv.Mat(f32) = nil) {
	target := units
	if target == nil {
		target = &s.units
	}

	pipeline_masks(s)
	pipeline_heat(s)
	pipeline_flow(s)
	resample_units(target, &s.units_next, s.navmap, &s.flow, &s.pipe.map_x, &s.pipe.map_y)
}