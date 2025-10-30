from ligotools import readligo as rl
import numpy as np

H1 = "data/H-H1_LOSC_4_V2-1126259446-32.hdf5"
L1 = "data/L-L1_LOSC_4_V2-1126259446-32.hdf5"

def test_loaddata_h1_lengths():
    strain, time, chan = rl.loaddata(H1, 'H1')
    assert len(strain) == len(time)

def test_loaddata_l1_monotonic_time():
    strain, time, chan = rl.loaddata(L1, 'L1')
    assert np.all(np.diff(time) > 0)
