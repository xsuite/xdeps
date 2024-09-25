import xdeps as xd
import numpy as np


t=xd.Table.from_tfs("twiss_lhcb1.tfs")

## Column access

%timeit t._data['betx']
# 25.3 ns ± 0.307 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)

%timeit t['betx']
# 51.7 ns ± 1.18 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)

%timeit t.betx
# 232 ns ± 4.95 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)


## Row column access

%timeit t._data['betx'][5719]
# 51.8 ns ± 0.189 ns per loop (mean ± std. dev. of 7 runs, 10,000,000 loops each)

%timeit t['betx',5719]
# 218 ns ± 13.2 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)

%timeit t['betx',"ip5"]
# 299 ns ± 15.5 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)

## Select cols

%timeit t._select_cols(["betx","bety"])
# 21.1 μs ± 838 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

%timeit t.cols["betx","bety"]
#26.5 μs ± 1.54 μs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

%timeit t.cols["betx bety"]
#27.9 μs ± 106 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)


## Select rows

%timeit t._select_rows(slice(5719, 8644))
# 33.4 μs ± 1.31 μs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

%t.rows[5719:8644]
# 137 μs ± 3.73 μs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

%timeit t.rows["ip5":"ip6"]
# 146 μs ± 1.3 μs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)


