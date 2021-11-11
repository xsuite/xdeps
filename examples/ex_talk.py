v={} # create a container
v['a']=1 # with some data
v['b']=1

import xdeps # import library

mgr=xdeps.Manager() # create manager
r=mgr.ref(v,'v') # create reference to `v`

r['c'] # a reference
0.1*r['a']+0.3*r['b'] # an expression
r._eval('0.1*a+0.3*b') # same as above (shorter)

r['c']=0.1*r['a']+0.3*r['b'] # set a task
r._exec('c=0.1*a+0.3*b')  # same as above (shorter)
mgr.tasks[r['c']] # the task

r['a']=3 # change a and trigger a change in `c`
v['a']=2 # change a and nothing else

mgr.find_deps([r['a']]) # find dependencies
mgr.find_tasks([r['a']]) # find tasks to be executed
mgr.plot_tasks([r['a']]) # find tasks to be executed

