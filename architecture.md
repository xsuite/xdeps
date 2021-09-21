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

Push strategy:
   -  Rule defined by target, action, dependencies
       -  target=action() when dependencies change
       -  only one rule per target?
   -  Rule manager to detect cycle and order dependencies

Targets and dependencies can define using "references":
   - Defined by owner and part:
       - AttrRef(m,'a') refers to m.a
       - ItemRef(m,3) refers to m[3]

Actions:
   - Callable object with no arguments
        - generic but events needs to be provided by the users
   - Expresion of references e.g. AttrRef(m,'a')*3:
       - easy to write
       - easy to compute dependencies


Verbose code:

MADX:   `c:= a + b;``

```python
# user code
class M:
    pass

m=M() # a namespace
# user code

manager.register(
       Rule(
          target=AttrRef(m,'c'),
          action=(AttrRef(m,'a')+AttrRef(m,'b'))._get_value )
          dependencies=[AttrRef(m,'a'), AttrRef(m,'b')])

```
alternative with function

```python
m=M() # a namespace
manager.register(
       Rule(
          target=AttrRef(m,'c'),
          events=[AttrRef(m,'a'), AttrRef(m,'b')]
          action=lambda : m.a+m.b  )
)
```

trigger a computation

```
manager.apply_set(AttrRef(m,'a'),value)
```

Syntatic sugar
---------------


Syntax sugar with decoration:
- (+) intercept setting
- (+) simplify setting creation
- (-) decoration
Syntax sugar without decoration:
+ (+) no decoration
+ (-) explicit setting


Variant A without decoration:

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
del mref.c delete rule
```

Variant C without decoration

```python
from xref import manager, expr

class M:
    pass

m=M()

mref=manager.ref(m)
expr(,mref) #define and set rule
mref.a=3 # m.a and m.c will be updated
m.a=3 #only m.a will be updated
```


Variant B with decoration, condemn symbols with trailing `_`
```python
from xref import manager

@manager.class_  # decorate
class M:
    pass

m=M()
#m.c_ equivalent to AttrRef(m,'c')
m.c_ = m.a_ + m.b_ # define and set rule
m.a=3 # update m.a and m.c
```


Complication for structtured data
-----------------------------------------

Decorated classes

```python
m.c_ = m.a_       # recompute on setattr(m,a)
m.c_ = m.a_.b     # recompute on setattr(m,a) and setattr(m.a,b) 
m.c_ = m.a.b_     # recompute on setting `m.a.b_`
m.c_ = m.a_.b_    # what is it? (nothing probably)
m.c_ = m.a_[m.b_] # recompute on setting `m.a`
```

- `m.c_ = m.a_[m.b]`    : recompute on `m.a_` and value `m.b`
- `m.c_ = m.a_[m.b_]`   : recompute on `m.a_` and on`m.b`


Not decorated


Reference to nested structure

- `mref.target = mref.a`    : recompute on setting `m.a`
- `mref.target = mref.a.b`  : recompute on setting `m.a` and 
- `m.target_ = m.a.b_`  : recompute on setting `m.a.b_`
- `m.target_ = m.a_.b_` : what is it? (nothing probably)

Action on variable accessors

- `m.target_ = m.a_[m.b]`    : recompute on `m.a_` and value `m.b`
- `m.target_ = m.a_[m.b_]`   : recompute on `m.a_` and on`m.b`


