import xdeps



m=xdeps.Manager()

s={}
s_=m.ref(s,'s')

s['a']=1
s['b']=2
s_._exec('c=a+b')


s_['c']

