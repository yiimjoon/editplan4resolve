from VideoForge.broll.indexer import smart_frame_sampling


def test_sampling_invariants_short():
    frames = smart_frame_sampling(2.5)
    assert len(frames) == 3
    assert frames == sorted(frames)
    assert all(0.0 <= f <= 2.5 for f in frames)


def test_sampling_invariants_mid():
    frames = smart_frame_sampling(5.0)
    assert len(frames) == 5
    assert len(set(frames)) == len(frames)
    assert frames == sorted(frames)


def test_sampling_invariants_long():
    frames = smart_frame_sampling(20.0)
    assert len(frames) >= 6
    assert frames == sorted(frames)
    assert all(0.0 <= f <= 20.0 for f in frames)
