import xtrack as xt
import xdeps as xd


mgr = xd.Manager()

vv = mgr.ref(label="vv")
vv.a = 3
vv.m1 = xt.Multipole(order=12)
vv.m2 = xt.Multipole(order=12, _buffer=vv.m1._value._xobject._buffer)

vv.m1.knl[3] = vv.a + 1.1
vv.m2.knl[1] = vv.a + 1.3


xo1 = vv.m1._value._xobject
offset1 = xo1.knl._get_offset(3)
size1 = xo1.knl._itemtype._size

xo2 = vv.m2._value._xobject
offset2 = xo2.knl._get_offset(1)
size2 = xo2.knl._itemtype._size

buf = xo1._buffer
print(buf.buffer[offset1 : offset1 + size1].view(float))
print(buf.buffer[offset2 : offset2 + size2].view(float))

myupdate = mgr.gen_fun("myupdate", a=vv.a)
myupdate(a=12)

print(buf.buffer[offset1 : offset1 + size1].view(float))
print(buf.buffer[offset2 : offset2 + size2].view(float))

print(mgr.mk_fun("myupdate", a=vv.a))
