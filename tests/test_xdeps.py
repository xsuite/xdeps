import xdeps

def test_set():
    v={'a':1, 'b':3}
    mgr=xdeps.Manager()
    r=mgr.ref(v)
    r['c']=0.1*r['a']+0.2*r['b']
    assert r['c']==0.1*r['a']+0.2*r['b']
    r['a']=1.2
    assert r['c']==0.1*r['a']+0.2*r['b']




