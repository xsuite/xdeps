# copyright ############################### #
# This file is part of the Xdeps Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xdeps

def test_set():
    v={'a':1, 'b':3}
    mgr=xdeps.Manager()
    r=mgr.ref(v)
    r['c']=0.1*r['a']+0.2*r['b']
    assert v['c']==0.1*v['a']+0.2*v['b']
    r['a']=1.2
    assert v['c']==0.1*v['a']+0.2*v['b']


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


def test_inplace():
    m=xdeps.Manager()

    s={'a': 1, 'b': 2}
    s_=m.ref(s,'s')
    s_._exec('c=a+b')
    assert s['c']==s['a']+s['b']
    s_['c']+=1 ## s_['a']= s_['c']._expr + 1
    assert s['c']==s['a']+s['b']+1
    s_['a']+=1 ## s_['a']= s['a'] + 1
    assert s['c']==s['a']+s['b']+1


