# copyright ############################### #
# This file is part of the Xdeps Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from collections import defaultdict
from xdeps.sorting import toposort,depsort

def mpl_display_png(png):
    import io
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    sio = io.BytesIO()
    sio.write(png)
    sio.seek(0)
    img = mpimg.imread(sio)
    plt.imshow(img, aspect='equal')
    plt.xticks([])
    plt.yticks([])

def os_display_png(png):
    import os
    open("/tmp/out.png",'wb').write(png)
    os.system("(display /tmp/out.png;rm /tmp/out.png)&")

def ipy_display_png(png):
    from IPython.display import Image, display
    plt = Image(png)
    display(plt)

class Task:
    def __init__(self,targets,dependencies):
        self.targets=targets
        self.dependencies=dependencies

class TaskList:
    def __init__(self):
        self.tasks={}
        self.rdeps={} #source -> target
        self.tdeps={} #target -> actions

    def add_task(self,name,targets,dependencies):
        self.tasks[name]=Task(targets,dependencies)

    def del_task(self,name):
        self.tasks[name]

        del self.tasks[name]

    def _dfs(self,source,stack,visited,tvisited):
        visited.add(source)

        for task in self.tdeps[source]:
            for dep in task.targets:
                if dep not in visited:
                    self._dfs(dep,stack,visited)

        stack.appendleft(task)

    def to_pydot(self):
        from pydot import Dot, Node, Edge
        pdot = Dot("g", graph_type="digraph",rankdir="LR")
        for name, task in self.tasks.items():
            tn=Node(name, shape="circle")
            pdot.add_node(tn)
            for tt in task.targets:
                pdot.add_node(Node(str(tt), shape="square"))
                pdot.add_edge(Edge(name, str(tt), color="blue"))
            for tt in task.dependencies:
                pdot.add_node(Node(str(tt), shape="square"))
                pdot.add_edge(Edge(str(tt),tn, color="blue"))
        os_display_png(pdot.create_png())
        return pdot

    def descend(self,start):
        rdeps=defaultdict(set) # a: [b] == b depends on a
        for taskname, task in self.tasks.items():
            for dependency in task.dependencies:
                rdeps[dependency].add(taskname)
            for target in task.targets:
                rdeps[taskname].add(target)
        return toposort(rdeps,start)

    def descend2(self):
        deps=defaultdict(set)
        for taskname, task in self.tasks.items():
            for dependency in task.dependencies:
                deps[taskname].add(dependency)
            for target in task.targets:
                deps[target].add(taskname)
        return list(depsort(deps))






if __name__=="__main__":
    tl=TaskList()

    tl.add_task("a",[2,3],[0,1])
    tl.add_task("b",[5],[4,2])
    tl.add_task("c",[6],[5,3])
    tl.add_task("d",[5],[0,2])
    tl.add_task("f",[0],[6])
    tl.del_task("f")
    tl.to_pydot()

    print(tl.descend([1]))
    print(tl.descend([2]))
    print(tl.descend([1,2]))
