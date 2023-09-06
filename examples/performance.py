import time
from xdeps.refs import AddRef as R1
from xdeps.refsx import AddRef as R2

print(R1(3, 4))
print(R2(3, R2(3, 4)))

print(R2(3, R2(3, 4))._get_value())

print(R2(3, 4) + R2(2, 3))

def mklist(n, Ref, label):
    st = time.time()
    out = []
    for ii in range(n):
        out.append(Ref(ii, ii))
    print(f"list {label} {len(out)}: {time.time() - st} s")

def mkdict(n, Ref, label):
    st = time.time()
    out = {}
    for ii in range(n):
        out[Ref(ii, ii)] = n
    print(f"dict {label} {len(out)}: {time.time() - st} s")


mkdict(100000, R1, 'classic')
mkdict(100000, R2, 'cython')
mklist(100000, R1, 'classic')
mklist(100000, R2, 'cython')


