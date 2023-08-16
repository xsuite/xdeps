import xdeps as xd
import pickle

vv = {}
mgr = xd.Manager()
v = mgr.ref(vv, "v")

v['a']=3
v['b']=v['a']*2
v['a']=1.1

pickle.loads(pickle.dumps(mgr))

