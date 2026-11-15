package simulation

import cv "../odin-opencv"
import "core:math"
import "../heat"
import "../maps"
import "../nav"

// Experiment 1 of the assignment, with OpenCV doing the image operations:
//
//  1. Mask (input)     cv2.threshold on ONE label image that carries walls,
//                      walkable cells, units and the goal, so the same image
//                      feeds the units and the objective. The labels are
//                      walls 0, walkable 1, units 2, goal 3; thresholds at 0,
//                      1 and 2 split it into wall / unit / goal masks.
//  2. Heat diffusion   cv2.filter2D with the diffusion kernel. Walls are
//                      cleared through the mask: cv2.bitwise_and on CV_32F has
//                      freeze semantics (masked-out cells keep their value), so
//                      cv.Mat.set_to(0, wall_mask) is the operation that
//                      actually zeroes them; on the CV_8U masks themselves
//                      cv2.bitwise_and runs as-is.
//  3. Vector field     cv2.Sobel(X) and cv2.Sobel(Y) over the heat image.
//  4. Wall masking     cv2.bitwise_and again, clearing the flow on walls.
//  5. Resampling       cv2.remap moves the units image downstream along the
//                      normalized vector field.
//
// There is no double buffering: every step writes back into the image it
// reads. The one exception is cv2.remap, which cannot sample the buffer it is
// writing; its output goes into scratch and is copied back into the canonical
// units image instead of being swapped in, so the units stay a single image.

// OpenCV image state of the pipeline. The images the pipeline does not own are
// the simulation's own Mats (its unit image, the solver's heat field, the goal
// splat), so the probes and the renderer read the very same buffers.
Pipeline :: struct {
	board:      cv.Mat(u8),
	walkable:   cv.Mat(u8),
	beyond:     cv.Mat(u8),
	goal_inv:   cv.Mat(u8),
	units_mask: cv.Mat(u8),
	goal_mask:  cv.Mat(u8),
	wall_mask:  cv.Mat(u8),

	// The images the pipeline does not own: views of the simulation's unit image
	// and remap scratch image.
	units:   cv.Mat(f32),
	scratch: cv.Mat(f32),

	// The seal and the Sobel of steps 3 and 4 live in nav.FlowField now, which
	// owns its scratch grids and views them as Mats itself (nav.update), so the
	// pipeline does not carry a second set of them.
	map_x:    cv.Mat(f32),
	map_y:    cv.Mat(f32),
}

// pipeline_setup views the simulation's images as OpenCV Mats and allocates
// the pipeline's own mask/gradient/map images. Called from reset, after the
// heat field, goal splat, units and flow field exist.
pipeline_setup :: proc(s: ^Simulation) {
	if s.pipe == nil {
		s.pipe = new(Pipeline)
	}
	p := s.pipe
	w := s.dims[0]
	h := s.dims[1]

	p.board = cv.mat_new(u8, h, w)
	p.walkable = cv.mat_new(u8, h, w)
	p.beyond = cv.mat_new(u8, h, w)
	p.goal_inv = cv.mat_new(u8, h, w)
	p.units_mask = cv.mat_new(u8, h, w)
	p.goal_mask = cv.mat_new(u8, h, w)
	p.wall_mask = cv.mat_new(u8, h, w)

	p.map_x = cv.mat_new(f32, h, w)
	p.map_y = cv.mat_new(f32, h, w)

	// These two are the simulation's own images: no copy, the same bytes the rest
	// of the game reads.
	p.units = s.units
	p.scratch = s.units_next
}

// pipeline_masks builds the one label image and thresholds it into the masks
// the rest of the pipeline consumes.
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

	// thresholds are exact label boundaries: 0 -> walls, >1 -> units and
	// goal, >2 -> goal.
	cv.threshold(&p.board, &p.walkable, 0.0, 1, cv.THRESH_BINARY)
	cv.threshold(&p.board, &p.beyond, 1.0, 1, cv.THRESH_BINARY)
	cv.threshold(&p.board, &p.goal_mask, 2.0, 1, cv.THRESH_BINARY)
	cv.threshold(&p.board, &p.goal_inv, 2.0, 1, cv.THRESH_BINARY_INV)
	cv.threshold(&p.board, &p.wall_mask, 0.0, 1, cv.THRESH_BINARY_INV)

	// units = (labels > 1) and (labels <= 2): the same label image feeds the
	// units and the objective.
	cv.and(&p.beyond, &p.goal_inv, &p.units_mask)
}

// pipeline_heat is step 2: the diffusion is the solver's own OpenCV step
// (cv2.filter2D, cv2.divide, cv2.setTo, cv2.add), so the pipeline and the probes
// run the exact same operator instead of two implementations of it.
pipeline_heat :: proc(s: ^Simulation) {
	heat.step(&s.heat, &s.goal, HEAT_SUBSTEPS)
}

// pipeline_prebake relaxes the seeded wash to the steady state of the screened
// Poisson operator before the first frame. Left alone, the wash is a free-space
// Green's function: smooth, wall-blind, and pointing straight at the goal
// through every wall. The steady state is the same operator constrained by the
// walls, and its only maximum sits at the goal, so its Sobel has to go around
// the maze. This is a loop over the diffusion step, but it runs on reset, so
// frame 0 already shows the converged field.
pipeline_prebake :: proc(s: ^Simulation) {
	pipeline_masks(s)
	heat.step(&s.heat, &s.goal, HEAT_PREBAKE)
}

// pipeline_flow is steps 3 and 4: the vector field is cv2.Sobel(X) and cv2.Sobel(Y)
// of the heat, and the wall mask clears it on walls. That is nav.update, which now
// owns the images it runs on (the seal, the padded gradient, the wall mask) so the
// same field the simulation probes is the one nav computes.
pipeline_flow :: proc(s: ^Simulation) {
	nav.update(&s.flow, &s.heat)
}

// resample_units is step 5: cv2.remap moves the units image downstream along
// the vector field. The map is built in scatter order, so it says where each
// source cell goes rather than where each destination comes from: a unit at
// (x,y) is written to the cell it reaches after one step, (x+vx, y+vy),
// rounded to the nearest cell. Building the map this way means a moved unit
// leaves nothing behind, so units stay point masses instead of smearing into
// a haze the way bilinear gather sampling does.
//
// Every destination that no unit claims is pointed at a wall cell, whose
// value is always zero. That is how the vacated cells are emptied without a
// clearing pass and without a second unit image in the simulation: the remap
// output lands in scratch and is copied back into the canonical units image.
// A step that would leave the map or enter a wall leaves its unit in place,
// so no unit is dragged through geometry.
resample_units :: proc(
	units: ^cv.Mat(f32),
	scratch: ^cv.Mat(f32),
	navmap: cv.Mat(f32),
	flow: ^nav.FlowField,
	map_x, map_y: ^cv.Mat(f32),
) {
	w := int(navmap.cols)
	h := int(navmap.rows)

	// A wall cell is never a valid destination, so sampling it always yields
	// zero. Use it as the trash cell for all unclaimed destinations.
	trash := 0
	for i in 0..<w * h {
		if navmap.data[i] != maps.WALKABLE {
			trash = i
			break
		}
	}
	// A negative x marks a destination that no unit has claimed yet.
	for i in 0..<int(map_x.rows * map_x.cols) {
		map_x.data[i] = -1
		map_y.data[i] = 0
	}

	// Pass 1: every occupied cell reserves itself. A unit therefore always has
	// a cell to fall back on, and a neighbour can never claim it out from under
	// it; without this reservation a unit that stays put can be erased by the
	// unit behind it, which is where the earlier mass loss came from.
	for i in 0..<w * h {
		if units.data[i] > 0 && navmap.data[i] == maps.WALKABLE {
			map_x.data[i] = f32(i % w)
			map_y.data[i] = f32(i / w)
		}
	}

	// Pass 2: a unit keeps its own cell and moves only into a cell nobody
	// holds. The flow converges on the goal, so this doubles as the congestion
	// rule: a unit whose destination is taken waits where it is instead of
	// being overwritten. The map stays one-to-one, so the remap conserves
	// every unit exactly.
	//
	// A queue has to be released in the order the units are facing, and a
	// single sweep in raster order always visits the front of the queue last,
	// which freezes the line. Repeating the sweep lets the vacated cells hand
	// the movement back along the queue, like a stream of units stepping
	// forward one after the other.
	for _ in 0..<8 {
		moved := 0
		for y in 0..<h {
			for x in 0..<w {
				i := x + y * w
				if units.data[i] <= 0 || navmap.data[i] != maps.WALKABLE {
					continue
				}
				// Already moved this frame: its reservation was released.
				if map_x.data[i] != f32(x) || map_y.data[i] != f32(y) {
					continue
				}
				dir := flow.field.data[i]
				// A cell with no flow is a local maximum of the heat field: the
				// unit there has nowhere to climb, so it stays put.
				if dir[0] == 0 && dir[1] == 0 {
					continue
				}
				dx := int(math.round(dir[0]))
				dy := int(math.round(dir[1]))
				// Flow shorter than half a cell still means "move that way", so
				// step on the dominant axis instead of rounding to zero.
				if dx == 0 && dy == 0 {
					ax := dir[0]
					if ax < 0 {
						ax = -ax
					}
					ay := dir[1]
					if ay < 0 {
						ay = -ay
					}
					if ax >= ay {
						if dir[0] > 0 {
							dx = 1
						} else {
							dx = -1
						}
					} else {
						if dir[1] > 0 {
							dy = 1
						} else {
							dy = -1
						}
					}
				}
				// The whole vector can point into a corner, where the diagonal
				// cell is a wall but one of the two axis neighbours is open.
				// Try the axes as well, which is how a unit slides along a wall
				// instead of freezing in front of it.
				dest := -1
				candidates := [3][2]int{{dx, dy}, {dx, 0}, {0, dy}}
				for c in candidates {
					if c[0] == 0 && c[1] == 0 {
						continue
					}
					nx := x + c[0]
					ny := y + c[1]
					if nx < 0 || ny < 0 || nx >= w || ny >= h {
						continue
					}
					j := nx + ny * w
					if navmap.data[j] != maps.WALKABLE || map_x.data[j] >= 0 {
						continue
					}
					dest = j
					break
				}
				// A unit can still face a wall dead on, where every candidate
				// above is blocked. Take the open neighbour that lies furthest
				// along the flow instead, so the unit slides sideways along the
				// wall rather than freezing in front of it.
				if dest < 0 {
					best := f32(-2)
					sides := [4][2]int{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}
					for c in sides {
						nx := x + c[0]
						ny := y + c[1]
						if nx < 0 || ny < 0 || nx >= w || ny >= h {
							continue
						}
						j := nx + ny * w
						if navmap.data[j] != maps.WALKABLE || map_x.data[j] >= 0 {
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
				map_x.data[i] = -1
				map_x.data[dest] = f32(x)
				map_y.data[dest] = f32(y)
				moved += 1
			}
		}
		if moved == 0 {
			break
		}
	}

	// Everything still unclaimed reads the trash cell, i.e. zero.
	for i in 0..<int(map_x.rows * map_x.cols) {
		if map_x.data[i] < 0 {
			map_x.data[i] = f32(trash % w)
			map_y.data[i] = f32(trash / w)
		}
	}

	// Nearest sampling keeps a unit on the single cell it reached. Bilinear
	// would split it across four cells and, applied every frame, would turn
	// each unit into a widening blur.
	cv.resample(units, scratch, map_x, map_y, cv.INTER_NEAREST)

	copy(units.data[:w * h], scratch.data[:w * h])
}

// pipeline_step is one frame of the OpenCV experiment: masks, heat diffusion,
// vector field, wall masking, resampling.
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

