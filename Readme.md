# Xdeps

Value dependency manager with convenient syntax for expressions dependency.

## Example


```python
s={'a': 1, 'b': 2} # Assume a container exists

import xdeps
m=xdeps.Manager() # Create dependency manager
s_=m.ref(s,'s') # Create a reference to the container

s_['c']= s_['a']+s_['b'] # Now s['c'] depends on s_['a'] and s_['b']
s_._exec('d=c+b') # Same as above using a convenient function to save typing
s_['a']=1.1 # This updates 'a' and anything depending on 'a' such as 'c' and 'd'
assert s['c']==s['a']+s['b']
assert s['d']==s['c']+s['b']

s_['a']+=1.1 # Increment a
assert s['c']==s['a']+s['b']
assert s['d']==s['c']+s['b']

s_['c']+=s_['b']*2 # Modify expression
print(s_['c']._expr) # Inspect c-> ((s['a']+s['b'])+(s['b']*2))

set_ab=m.gen_fun('set_ab',a=s_['a'],b=s_['b']) # generate setter function
set_ab(2.3,1.2) # set a, b and dependecies
assert s['c']==s['a']+s['b']+s['b']*2
assert s['d']==s['c']+s['b']

dump=m.dump() # save the dependencies
m2=xdeps.Manager() # create a new manager
s2_=m.ref(s,'s') # set a reference to the container
m2.reload(dump) # reload dependencies

m2.plot_deps(backend='os') # Inspect depency graph
```

![Example](doc/example.png)

## Description

The library is based on a manager `mgr=xdeps.Manager()` that manages tasks and dependencies.

A `Task` is an instance that has:
- A `taskid`
- A set of `targets`
- A set of `dependecies`
- A method `run()` that uses the dependecies to update the targets.

A `Task` can be registered using `mgr.register(task)` or removed using `mgr.unregister(task)`.


Target and dependencies can be defined by a `Ref` object that can be obtained by `s_=mgr.ref(s,'s')` where `s` can be a dictionary or an instance. The label 's' is necessary for printing, but also to save and restore dependencies and must be unique.  `mgr.set_value(ref,value)` set the `value` into the reference `ref` and call the tasks that depends on `ref`.

A reference of a part of `s` could be obtained using standard Python syntax such as `s_.a` or `s_['b'][0].a` etc...

A `Ref` object could be used to define expressions such as `ex=s_.a*2+3` that can be evaluated using `ex._get_value()`.

A `Ref` of a cointainer can be assigned to an expression. This creates a special `ExprTask` that is registered in the manager. If an expressions was already created, it is replaced.

A `Ref` of a cointainer can be assigned to a value. In this case the value is assigned to the container that the tasks that depends on the reference are called in the correct ordering. Also in this case if an expressions was already created, it is deleted.



















