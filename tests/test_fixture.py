import numpy as np

from conftest import REAL_REGION


def test_fixture_has_hubs_and_no_bad_edges(region):
    z = np.load(region)
    pre, post = z["pre"], z["post"]
    assert pre.size == 600 and not (pre == post).any()
    assert len(set(zip(pre.tolist(), post.tolist()))) == 600
    indeg = np.bincount(post, minlength=120)
    assert indeg.max() > 5 * np.median(indeg)


def test_real_region_file_is_present():
    assert REAL_REGION.exists(), "run scripts/extract_subgraph.py first"
