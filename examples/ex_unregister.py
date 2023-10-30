from xdeps import Manager

ev = {"k": {}, "a": 3}
mgr = Manager()
e = mgr.ref(ev, "e")
e["k"]["a"] = e["a"] * 3
print(mgr.rdeps)
e["k"]["b"] = e["a"] * 2
print(mgr.rdeps)
mgr.unregister(e["k"]["a"])
print(mgr.rdeps)

mgr.verify()

ev = {"k": {}, "a": 3}
mgr = Manager()
e = mgr.ref(ev, "e")
e["k"]["b"] = e["a"] * 2
print(mgr.rdeps)
