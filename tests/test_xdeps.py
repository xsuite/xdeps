import xdeps

def test_set():
    v={'a':1, 'b':3}
    mgr=xdeps.Manager()
    r=mgr.ref(v)
    r['c']=0.1*r['a']+0.2*r['b']
    assert r['c']==0.1*r['a']+0.2*r['b']
    r['a']=1.2
    assert r['c']==0.1*r['a']+0.2*r['b']


def test_attrref():
    class E:
        pass

    e=E()
    e.knl=[0,1,2,3]
    v={4:e,2:1.2}

    mgr=xdeps.Manager()
    v_=mgr.ref(v,'v')
    v_[4].knl[1]=v_[2]*3

    assert v[4].knl[1]==v[2]*3
    v_[2]=1.1
    assert v[4].knl[1]==v[2]*3



