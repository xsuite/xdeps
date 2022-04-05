from dataclasses import dataclass, field
from collections import defaultdict
import logging

from .refs import ARef, Ref, ObjectAttrRef
from .refs import AttrRef, CallRef, ItemRef
from .utils import os_display_png, mpl_display_png, ipy_display_png
from .utils import AttrDict
from .sorting import toposort

logger = logging.getLogger(__name__)


class FuncWrapper:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        return CallRef(self.func, args, tuple(kwargs.items()))


class Task:
    taskid: object
    targets: set
    dependencies: set
    def run(self):
        raise NotImplemented

class GenericTask(Task):
    taskid: object
    action: object
    targets: set
    dependencies: set

    def __repr__(self):
        return f"<Task {self.taskid}:{self.dependencies}=>{self.targets}>"

    def run(self, *args):
        return self.action(*args)


class ExprTask(Task):
    def __init__(self, target, expr):
        self.taskid = target
        self.targets = target._get_dependencies()
        self.dependencies = expr._get_dependencies()
        self.expr = expr

    def __repr__(self):
        return f"   {self.taskid} = {self.expr}"

    def run(self):
        value = self.expr._get_value()
        self.taskid._set_value(value)

    def info(self):
        print(f"#  {self.taskid}._expr._get_dependencies()")
        for pp in self.expr._get_dependencies():
            print(f"   {pp} = {pp._get_value()}")
        print()


class InheritanceTask(Task):
    def __init__(self, children, parents):
        self.taskid = children
        self.targets = set([children])
        self.dependencies = set(parents)

    def __repr__(self):
        return f"{self.taskid} <- {self.parents}"

    def run(self, event):
        key, value, isattr = event
        for target in self.targets:
            if isattr:
                getattr(target, key)._set_value(value)
            else:
                target[key]._set_value(value)


class DepEnv:
    __slots__ = ("_data", "_")

    def __init__(self, data, ref):
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_", ref)

    def __getattr__(self, key):
        return getattr(self._data, key)

    def __getitem__(self, key):
        return self._data[key]

    def __setattr__(self, key, value):
        self._[key] = value

    def __setitem__(self, key, value):
        self._[key] = value

    def _eval(self, expr):
        return self._._eval(expr)


class Manager:
    """
    Value dependency manager:

    tasks: taskid -> task
    rdeps: ref -> set of all refs that depends on `ref`
    rtasks: taskid -> set all tasks whose dependencies are affected by taskid
    deptasks: ref -> all tasks that has ref as dependency
    tartasks: ref -> all tasks that has ref as target
    containers: label -> controlled container
    """

    def __init__(self):
        self.tasks = {}
        self.rdeps = defaultdict(set)
        self.rtasks = defaultdict(set)
        self.deptasks = defaultdict(set)
        self.tartasks = defaultdict(set)
        self.containers = {}

    def ref(self, container=None, label="_"):
        """Return a ref to an instance or dicr with a label.

        Label must be unique.
        """
        if container is None:
            container = AttrDict()
        objref = Ref(container, self, label)
        assert label not in self.containers
        self.containers[label] = objref
        return objref


    def set_value(self, ref, value):
        """Set a value pointed by a ref and execute all tasks that depends on ref.

        If the value is a Ref, create a new task from the ref.
        """
        logger.info("set_value %s %s",ref,value)
        if ref in self.tasks:
            self.unregister(ref)
        if isinstance(value, ARef):  # value is an expression
            self.register(ref, ExprTask(ref, value))
            value = value._get_value()  # to be updated
        ref._set_value(value)
        self._run_tasks(self.find_tasks(ref._get_dependencies()))

    def _run_tasks(self, tasks):
        for task in tasks:
            logger.info("Run %s",task)
            task.run()

    def register(self, taskid, task):
        """Register a new task identified by taskid"""
        #logger.info("register %s",taskid)
        self.tasks[taskid] = task
        for dep in task.dependencies:
            #logger.info("%s have an impact on %s",dep,task.targets)
            self.rdeps[dep].update(task.targets)
            #logger.info("%s is used by T:%s",dep,taskid)
            self.deptasks[dep].add(taskid)
            for deptask in self.tartasks[dep]:
                #logger.info("%s modifies deps of T:%s",deptask,taskid)
                self.rtasks[deptask].add(taskid)
        for tar in task.targets:
            #logger.info("%s is modified by T:%s",tar,taskid)
            self.tartasks[tar].add(taskid)
            for deptask in self.deptasks[tar]:
                #logger.info("T:%s modifies deps of T:%s",taskid,deptask)
                self.rtasks[taskid].add(deptask)

    def unregister(self, taskid):
        """Unregister the task identified by taskid"""
        task = self.tasks[taskid]
        for dep in task.dependencies:
            for target in task.targets:
                self.rdeps[dep].remove(target)
            self.deptasks[dep].remove(taskid)
        for tar in task.targets:
            self.tartasks[tar].remove(taskid)
            for deptask in self.deptasks[tar]:
                self.rtasks[taskid].remove(deptask)
        del self.tasks[taskid]

    def find_deps(self, start_set):
        """Find all refs that depends on ref in start_seps"""
        assert type(start_set) in (list, tuple, set)
        deps = toposort(self.rdeps, start_set)
        return deps

    def find_taskids_from_tasks(self, start_tasks=None):
        """Find all taskids whose dependencies are affected by the tasks in start_tasks"""
        if start_tasks is None:
            start_tasks = self.rtasks
        tasks = toposort(self.rtasks, start_tasks)
        return tasks

    def find_taskids(self, start_deps=None):
        """Find all taskids that depend on the refs in start_deps"""
        if start_deps is None:
            start_deps = self.rdeps
        start_tasks = set()
        for dep in start_deps:
            start_tasks.update(self.deptasks[dep])
        tasks = toposort(self.rtasks, start_tasks)
        return tasks

    def find_tasks(self, start_deps=None):
        """Find all tasks that depend on the refs in start_deps"""
        if start_deps is None:
            start_deps = self.rdeps
        return [self.tasks[taskid] for taskid in self.find_taskids(start_deps)]

    def mk_fun(self, name, **kwargs):
        """Write a python function that executes a set of tasks in order of dependencies:
            name: name of the functions
            kwards:
                the keys are used to defined the argument name of the functions
                the values are the refs that will be set
        """
        varlist, start = list(zip(*kwargs.items()))
        tasks = self.find_tasks(start)
        fdef = [f"def {name}({','.join(varlist)}):"]
        for vname, vref in kwargs.items():
            fdef.append(f"  {vref} = {vname}")
        for tt in tasks:
            fdef.append(f"  {tt}")
        fdef = "\n".join(fdef)
        return fdef

    def gen_fun(self,name, **kwargs):
        """Return a python function that executes a set of tasks in order of dependencies:
            name: name of the functions
            kwards:
                the keys are used to defined the argument name of the functions
                the values are the refs that will be set
        """
        fdef = self.mk_fun(name, **kwargs)
        gbl = {}
        lcl = {}
        gbl.update((k, r._owner) for k, r in self.containers.items())
        exec(fdef, gbl, lcl)
        return lcl[name]

    def plot_deps(self, start=None, backend="ipy"):
        """Plot a graph of task and target dependencies from start.

        Possible backend:
            mpl: generate a figure in matplotlib
            os: generate a file /tmp/out.png and use `display` to show it
            ipy: use Ipython facility for Jupyter notebooks
        """
        from pydot import Dot, Node, Edge

        if start is None:
            start = list(self.rdeps)
        pdot = Dot("g", graph_type="digraph", rankdir="LR")
        for task in self.find_tasks(start):
            tn = Node(" " + str(task.taskid), shape="circle")
            pdot.add_node(tn)
            for tt in task.targets:
                pdot.add_node(Node(str(tt), shape="square"))
                pdot.add_edge(Edge(tn, str(tt), color="blue"))
            for tt in task.dependencies:
                pdot.add_node(Node(str(tt), shape="square"))
                pdot.add_edge(Edge(str(tt), tn, color="blue"))
        png = pdot.create_png()
        if backend == "mpl":
            mpl_display_png(png)
        elif backend == "os":
            os_display_png(png)
        elif backend == "ipy":
            ipy_display_png(png)
        return pdot

    def plot_tasks(self, start=None, backend="ipy"):
        """Plot a graph of task dependencies

        Possible backend:
            mpl: generate a figure in matplotlib
            os: generate a file /tmp/out.png and use `display` to show it
            ipy: use Ipython facility for Jupyter notebooks
        """
        from pydot import Dot, Node, Edge

        if start is None:
            start = list(self.rdeps)
        pdot = Dot("g", graph_type="digraph", rankdir="LR")
        for task in self.find_tasks(start):
            tn = Node(str(task.taskid), shape="circle")
            pdot.add_node(tn)
            for dep in task.dependencies:
                for tt in self.tartasks[dep]:
                    pdot.add_edge(Edge(str(tt), tn, color="blue"))
        png = pdot.create_png()
        if backend == "mpl":
            mpl_display_png(png)
        elif backend == "os":
            os_display_png(png)
        elif backend == "ipy":
            ipy_display_png(png)
        return pdot


    def dump(self):
        """Dump in json all ExprTask defined in the manager
        """
        data = [
            (str(t.taskid), str(t.expr))
            for t in self.find_tasks(self.rdeps)
            if isinstance(t, ExprTask)
        ]
        return data

    def load(self, dump):
        """Reload the expressions in dump
        """
        for lhs, rhs in dump:
            lhs = eval(lhs, {}, self.containers)
            rhs = eval(rhs, {}, self.containers)
            task = ExprTask(lhs, rhs)
            self.register(task.taskid, task)

    def newenv(self, label="_", data=None):
        "Experimental"
        if data is None:
            data = AttrDict()
        ref = self.ref(data, label=label)
        return DepEnv(data, ref)

    def refattr(self, container=None, label="_"):
        "Experimental"
        if container is None:
            container = AttrDict()
        objref = ObjectAttrRef(container, self, label)
        assert label not in self.containers
        self.containers[label] = objref
        return objref
