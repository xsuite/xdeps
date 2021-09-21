Expression
============

Main criteria
------------

An expression creates a dependency between data

Pull: compute on get
   - (+) easy to implement
   - (-) all data structure needs to be aware
   - (-) not feasible on GPUs

Push: compute on set
   + (+) more generic
   + (+) applicable to any external data structure
   - (-) difficult to implement and express

Push model
---------------
A reference is a tuple (owner, part) such as
  - `AttrRef(m,'a')` refers to m.a
  - `ItemRef(m,3)` refers to m[3]

A rule associate a function producing a value (action) that is associated to a target identified by a reference.
The actions has dependencies also identified by references.
Example `c:= a + b;`:

```python
Rule(
    target=AttrRef(m,'c'),
    action=lambda :m.a+m.b,
    dependencies=[AttrRef(m,'a'), AttrRef(m,'b')])
```

When a dependency change the action is executed and the target is update.
A manager collects actions, detects cycles and sort actions to respect dependencies.


```python
# user code
class M:
    pass

m=M() # a namespace
# end user code

from xdeps import manager

manager.register(
       Rule(
          target=AttrRef(m,'c'),
          action=lambda : m.a+m.b,
          dependencies=[AttrRef(m,'a'), AttrRef(m,'b')])
)
manager.apply_set(AttrRef(m,'a'),value)
# equivalent to
m.a=value
m.c=m.a+m.b
```


Questions:
- Is imposing one and only one action per target too restrictive?
- Can one model actions with implicit targets
Example
```python
def myaction(m):
   m.c=m.a+m.c

act=Action(
      args=(m),
      kwargs={},
      action=myaction,
      dependencies=[AttrRef(m,'a'), AttrRef(m,'b')])
)

act.execute() # act.action(*act.args,**act.kwargs)
```

Syntatic sugar
---------------

Syntax sugar can be used to simplify
- create references
- create actions
- trigger updates 

Without changing user classes

```python
#user code
class M:
    pass

m=M()
#user code

from xref import manager

mref=manager.ref(m) # build reference factory
#mref.c equivalent to AttrRef(m,'c')
mref.c = mref.a + mref.b # define and set rule
# alternative
mref+='c=a+b'
mref.a=3 # m.c will be updated
m.a=3 #nothing happens in target
del mref.c # delete rule
```

Allowing modyfing user class and restricting to parts without trailing `_`
```python
from xref import manager

@manager.class_  # decorate
class M:
    pass

m=M()
#m.c_ equivalent to AttrRef(m,'c')
m.c_ = m.a_ + m.b_ # define and set rule
m.a=3 # update m.a and m.c
del m.c_ # delete rule
```


Nested structure
-----------------------------------------
```python
class M():
   def __init__(**kwargs):
      self.__dict__.update(kwargs)

m=M()
from xdeps import manager
mref=manager.ref(m)
mref.c = mref.a
mref.a=3 # OK! triggers

mref.c = mref.a.b
mref.a=M(b=2) # triggers
mref.a.b=3 # triggers

```


Decorated classes

```python
m.c_ = m.a_       # recompute on setattr(m,'a')
m.c_ = m.a_.b     # recompute on setattr(m,'a') and setattr(m.a,'b') only if m.a is decorated 
m.c_ = m.a.b_     # recompute on settattr(m.a,'b') only if m.a is decorated
m.c_ = m.a_[3]    #  
```

