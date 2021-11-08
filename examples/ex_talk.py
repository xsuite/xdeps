v={} # create a container

import xdeps # import library

mgr=xdeps.Manager() # create manager
r=mgr.ref(v,'v') # create reference to `v`

v['a']=1
v['b']=1
r['c'] # a reference
0.1*r['a']+0.3*r['b'] # an expressionre
r._eval('0.1*a+0.3*b') # same as above (shorter)

r['c']=0.1*r['a']+0.3*r['b'] # set a task
r._eval('c=0.1*a+0.3*b')  # same as above (shorter)
mgr.tasks[r['c']] # the task

r['a']=3 # change a and trigger a change in `c`
v['a']=2 # change a and nothing else

mgr.find_deps([r['c']]) # find dependencies
mgr.find_tasks([r['c']]) # find tasks to be executed
mgr.plot_tasks([r['c']]) # find tasks to be executed






