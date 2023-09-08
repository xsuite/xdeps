import operator
import time
from functools import partial

from xdeps.refs import BinOpRef
from xdeps.refsx import AddRef as R2

R1 = partial(BinOpRef, operator=operator.add)

print(R1(3, 4))
print(R2(3, R2(3, 4)))

print(R2(3, R2(3, 4))._get_value())

print(R2(3, 4) + R2(2, 3))


def mklist(n = 100000, Ref=R1, lbl=''):
    st = time.time()
    out = []
    for ii in range(n):
        ref = Ref(ii, ii) + 2 * ii + ii
        out.append(ref)
    print(f"{lbl} list {len(out):10}: {time.time()-st:10.6f} s")


def mkdict(n = 100000, Ref=R1, lbl=''):
    st = time.time()
    out = {}
    for ii in range(n):
        ref = Ref(ii, ii) + 2 * ii + ii
        out[ref] = n
    print(f"{lbl} dict {len(out):10}: {time.time()-st:10.6f} s")


mkdict(100000, R1, 'classic')
mkdict(100000, R2, 'cython')
mklist(100000, R1, 'classic')
mklist(100000, R2, 'cython')


