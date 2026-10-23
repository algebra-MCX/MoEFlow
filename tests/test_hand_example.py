"""Literal numbers from docs/architecture.md §7; any change here must be mirrored in the doc."""

from moeflow.hand_example import run


def test_hand_example_matches_document():
    r = run()
    assert r["send_counts"] == [[2, 2], [1, 1]]
    assert r["send_token"] == [[0, 1, 1, 2], [0, 0]]
    assert r["send_slots"] == [[[0, 1], [1, -1], [-1, 1], [0, 1]], [[-1, 0], [1, -1]]]
    assert r["recv_counts"] == [[2, 1], [2, 1]]
    assert r["recv_slots"] == [[[0, 1], [1, -1], [-1, 0]], [[-1, 1], [0, 1], [1, -1]]]
    assert r["expert_offsets"] == [[0, 2, 4], [0, 1, 4]]
    assert r["expert_entries"] == [[(0, 0), (2, 1), (0, 1), (1, 0)], [(1, 0), (0, 1), (1, 1), (2, 0)]]
    assert r["partials"] == [[[3.5, 7.0], [12.0, -4.0], [0.25, 2.0]], [[6.0, -2.0], [17.5, 4.375], [4.0, 32.0]]]
    assert r["returned"] == [[[3.5, 7.0], [12.0, -4.0], [6.0, -2.0], [17.5, 4.375]], [[0.25, 2.0], [4.0, 32.0]]]
    assert r["out"] == [[[3.5, 7.0], [18.0, -6.0], [17.5, 4.375]], [[4.25, 34.0]]]
