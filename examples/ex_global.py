import xdeps

manager=xdeps.DepManager()
m=manager.refattr(globals(),'m')

a=b=0
m.c=0.1*m.a+0.2*m.b
m.a=2
print(c)
m.b=3
print(c)






