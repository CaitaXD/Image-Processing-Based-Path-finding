package units

import "core:testing"
import "../grid"
import "../maps"

@(test)
sample_deterministic :: proc(t: ^testing.T) {
    m, e := grid.make(u8, 40, 30)
    assert(e == .None)

    for x in 0..<m.width {
        for y in 1..<m.height {
            grid.at(m, x, y)^ = maps.WALKABLE
        }
    }

    a := sample(m, 100, 7)
    b := sample(m, 100, 7)
    assert(len(a) == 100)
    assert(len(b) == 100)

    for i in 0..<len(a) {
        assert(a[i].pos[0] == b[i].pos[0] && a[i].pos[1] == b[i].pos[1],
            "same seed must produce identical placements")
    }

    for unit in a {
        assert(grid.at(m, unit.pos[0], unit.pos[1])^ == maps.WALKABLE,
            "unit must sit on a walkable cell")
    }

    c := sample(m, 5000, 9)
    assert(len(c) == 1160, "count must clamp to the number of walkable cells")
}

@(test)
sample_goal_deterministic :: proc(t: ^testing.T) {
    m, e := grid.make(u8, 40, 30)
    assert(e == .None)

    for x in 0..<m.width {
        for y in 0..<m.height {
            grid.at(m, x, y)^ = maps.WALKABLE
        }
    }
    grid.at(m, 5, 5)^ = maps.OBSTACLE

    a, ok_a := sample_goal(m, 7)
    b, ok_b := sample_goal(m, 7)
    assert(ok_a && ok_b)
    assert(a.pos == b.pos, "same seed must produce the same goal")
    assert(grid.at(m, a.pos[0], a.pos[1])^ == maps.WALKABLE, "goal must be walkable")

    empty, e2 := grid.make(u8, 8, 8)
    assert(e2 == .None)
    _, ok := sample_goal(empty, 1)
    assert(!ok, "no goal can be sampled from an all-obstacle grid")

    single, e3 := grid.make(u8, 4, 4)
    assert(e3 == .None)
    grid.at(single, 2, 3)^ = maps.WALKABLE
    g, ok_single := sample_goal(single, 42)
    assert(ok_single && g.pos == [2]int{2, 3}, "goal must be a walkable cell")
}
