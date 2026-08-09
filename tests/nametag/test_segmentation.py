from gromit.nametag.segmentation import Segment, segment_features


def test_single_segment_when_stable():
    feats = [[0.0, 0.0, 1.0, 1.0, 0.5]] * 6
    segs = segment_features(feats, threshold=0.05)
    assert segs == [Segment(start=0, end=5, representative=2)]


def test_split_on_change():
    feats = [[0.0, 0.0, 1.0, 1.0, 0.5]] * 3 + [[0.7, 0.05, 0.25, 0.25, 0.9]] * 3
    segs = segment_features(feats, threshold=0.1)
    assert len(segs) == 2
    assert segs[0].start == 0 and segs[0].end == 2
    assert segs[1].start == 3 and segs[1].end == 5


def test_representative_is_segment_midpoint():
    feats = [[0.0, 0.0, 1.0, 1.0, 0.5]] * 5
    segs = segment_features(feats, threshold=0.05)
    assert segs[0].representative == 2
