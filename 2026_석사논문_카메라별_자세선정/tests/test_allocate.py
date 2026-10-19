import pytest

from campose.allocate import (
    allocate_poses,
    allocation_string,
    direction_counts,
    direction_fractions,
)

# Per-camera direction counts (front, back, oblique, side) of the sampled gallery
# poses that passed the quality filter in the original experiment, and the
# allocation that was used there (front/back/oblique/side).
ORIGINAL = {
    "market_c1": ((61, 75, 168, 147), "1/1/3/3"),
    "market_c2": ((162, 77, 128, 99), "3/1/2/2"),
    "market_c3": ((114, 56, 136, 156), "2/1/2/3"),
    "market_c4": ((43, 31, 153, 175), "1/1/3/3"),
    "market_c5": ((97, 27, 134, 150), "2/1/2/3"),
    "market_c6": ((74, 102, 166, 115), "1/2/3/2"),
    "duke_c1": ((48, 72, 248, 36), "1/1/5/1"),
    "duke_c2": ((69, 56, 104, 211), "1/1/2/4"),
    "duke_c3": ((33, 37, 171, 203), "1/1/3/3"),
    "duke_c4": ((221, 207, 23, 4), "4/3/1/0"),
    "duke_c5": ((81, 63, 148, 170), "1/1/3/3"),
    "duke_c6": ((124, 113, 163, 60), "2/2/3/1"),
    "duke_c7": ((44, 17, 162, 234), "1/1/2/4"),
    "duke_c8": ((26, 36, 147, 196), "1/1/3/3"),
    "msmt_c2": ((0, 145, 155, 131), "0/3/3/2"),
    "msmt_c6": ((186, 3, 153, 76), "4/0/3/1"),
    "msmt_c7": ((85, 22, 78, 232), "2/1/1/4"),
    "msmt_c13": ((11, 185, 156, 111), "1/3/2/2"),
    "msmt_c1": ((48, 54, 176, 186), "1/1/3/3"),
    "msmt_c3": ((25, 29, 132, 236), "1/1/2/4"),
    "msmt_c4": ((23, 168, 183, 92), "1/3/3/1"),
    "msmt_c5": ((23, 179, 148, 83), "1/3/3/1"),
    "msmt_c8": ((11, 92, 156, 120), "1/2/3/2"),
    "msmt_c9": ((20, 187, 154, 76), "1/3/3/1"),
    "msmt_c10": ((0, 141, 204, 121), "0/2/4/2"),
    "msmt_c11": ((11, 128, 179, 131), "1/2/3/2"),
    "msmt_c12": ((100, 25, 139, 204), "2/1/2/3"),
    "msmt_c14": ((78, 59, 193, 134), "2/1/3/2"),
    "msmt_c15": ((0, 45, 159, 211), "0/1/3/4"),
}
DIRS = ("front", "back", "oblique", "side")


def _counts(t):
    return dict(zip(DIRS, t))


@pytest.mark.parametrize("name", sorted(ORIGINAL))
def test_reproduces_original_allocations(name):
    counts, expected = ORIGINAL[name]
    assert allocation_string(allocate_poses(_counts(counts))) == expected


def test_spec_examples():
    common = {"front": 1, "back": 1, "oblique": 3, "side": 3}
    assert allocation_string(common) == "1/1/3/3"
    assert allocation_string(allocate_poses(_counts(ORIGINAL["market_c2"][0]))) == "3/1/2/2"
    assert allocation_string(allocate_poses(_counts(ORIGINAL["market_c6"][0]))) == "1/2/3/2"
    assert allocation_string(allocate_poses(_counts(ORIGINAL["duke_c4"][0]))) == "4/3/1/0"


@pytest.mark.parametrize("counts", [c for c, _ in ORIGINAL.values()] + [
    (1000, 0, 0, 0), (5, 5, 5, 5), (6, 0, 7, 900), (10, 10, 10, 10_000), (0, 0, 5, 0),
])
def test_sum_is_k(counts):
    for k in (4, 8, 12):
        alloc = allocate_poses(_counts(counts), k=k)
        assert sum(alloc.values()) == k
        assert all(v >= 0 for v in alloc.values())


def test_zero_ratio_direction_gets_zero():
    alloc = allocate_poses({"front": 0, "back": 141, "oblique": 204, "side": 121})
    assert alloc["front"] == 0
    assert sum(alloc.values()) == 8


def test_rare_direction_below_min_samples_gets_zero_but_frequent_minority_gets_one():
    # 4 side poses (< 5) -> 0; 23 oblique poses (5 %) -> still at least 1
    alloc = allocate_poses({"front": 221, "back": 207, "oblique": 23, "side": 4})
    assert alloc["side"] == 0
    assert alloc["oblique"] == 1


def test_every_available_direction_gets_minimum():
    alloc = allocate_poses({"front": 5, "back": 5, "oblique": 5, "side": 10_000})
    assert min(alloc.values()) == 1
    assert alloc["side"] == 5


def test_no_available_direction_gives_all_zero():
    assert sum(allocate_poses({"front": 2, "back": 0, "oblique": 1, "side": 4}).values()) == 0


def test_counts_and_fractions():
    c = direction_counts(["front", "side", "side", "unknown"])
    assert c == {"front": 1, "back": 0, "oblique": 0, "side": 2}
    f = direction_fractions(c)
    assert f["side"] == pytest.approx(2 / 3)
    assert direction_fractions({}) == {d: 0.0 for d in DIRS}
