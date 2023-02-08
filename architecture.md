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
A reference is a tuple (owner, part, manager) such as
  - `AttrRef(m,'a')` refers to m.a
  - `ItemRef(m,3)` refers to m[3]

A rule associates a function producing a value (action) to a target identified by a reference.
The actions have dependencies also identified by references.
Example `c:= a + b;`:

```python
def setc(m):
    m.c=a+b

Tasks(
    action=setc,
    targets=[AttrRef(m,'c')],
    dependencies=[AttrRef(m,'a'), AttrRef(m,'b')]
)

```

When a dependency changes, the action is executed and the target is updated.
A manager collects actions, detects cycles and sort actions to respect dependencies.


```python
# user code
class M:
    pass

m=M() # a namespace
# end user code

from xdeps import manager
def setc(m):
    m.c=a+b


# set rule
manager.register(
    Tasks(
        action=setc,
        targets=[AttrRef(m,'c')],
        dependencies=[AttrRef(m,'a'), AttrRef(m,'b')]
    )

# propagate set
manager.set(AttrRef(m,'a'),value)

# equivalent to
m.a=value
m.c=m.a+m.b
```

Syntatic sugar
---------------

Syntax sugar can be used to simplify
- create references
- create actions
- trigger updates

```python
#user code
class M:
    pass

m=M()
#user code

from xref import manager

mref=manager.ref(m) # create an object controlling m

#mref.c equivalent to AttrRef(m,'c',manager)
mref.c = mref.a + mref.b # define and set rule
# alternative
mref._eval('c=a+b')

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

Nested structure (TBC)
-----------------------------------------
```python
class M():
   def __init__(self, **kwargs):
      self.__dict__.update(kwargs)

m=M()
import xdeps
mref=xdeps.Manager.ref(m)
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

