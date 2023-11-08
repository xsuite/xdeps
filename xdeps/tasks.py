# copyright ############################### #
# This file is part of the Xdeps Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from collections import defaultdict
from copy import deepcopy
import logging
from typing import Set, Hashable

from .refs import BaseRef, MutableRef, ObjectAttrRef, Ref, RefCount
from .utils import plot_pdot
from .utils import AttrDict
from .sorting import toposort

logger = logging.getLogger(__name__)


def dct_merge(dct1, dct2):
    """Merge two dictionaries, in case of conflict dct2 takes precedence."""
    return {**dct1, **dct2}


def _check_root_owner(t, ref):
    """Check if a task `t`, or any of its parents, has `ref` as owner."""
    if hasattr(t, "_owner"):
        if t._owner is ref:
            return True
        else:
            return _check_root_owner(t._owner, ref)
    else:
        return False


class Task:
    """
    An (abstract) class representing a task.

    A task describes an action that modifies the values of a set of
    target references depending on the values stored in the set of
    references which are dependencies, and potentially its internal state.

    Properties
    ----------
    taskid: BaseRef
        The reference that identifies the task, usually an expression that
        depends on the targets and dependencies and specifies the action
        to be performed.
    targets: Set[BaseRef]
        The set of references that are modified by the task.
    dependencies: Set[BaseRef]
        The set of references that the task depends on.
    """

    taskid: Hashable
    targets: Set[BaseRef]
    dependencies: Set[BaseRef]

    def run(self):
        """Execute the task."""
        raise NotImplemented


class FunctionTask(Task):
    """Task that executes a function when its dependencies change.

    Parameters
    ----------
    taskid: Hashable
        The task identifier.
    action: callable
        The function to be executed.
    targets: Set[BaseRef]
        The set of references that are modified by the task.
    dependencies: Set[BaseRef]
        The set of references that the task depends on.
    """
    action: callable

    def __init__(
            self,
            taskid: Hashable,
            action: callable,
            targets: Set[BaseRef],
            dependencies: Set[BaseRef],
    ):
        self.taskid = taskid
        self.action = action
        self.targets = targets
        self.dependencies = dependencies

    def __repr__(self):
        return f"<Task {self.taskid}:{self.dependencies}=>{self.targets}>"

    def run(self):
        return self.action()


class LinearKnob(Task):
    def __init__(self, taskid, source, weights, targets):
        self.taskid = taskid
        self.source = source
        self.dependencies = {source}
        self.targets = targets
        self.prev_value = source._get_value()
        self.weights = weights

    def run(self):
        value = self.source._get_value()
        delta = value - self.prev_value

        for w, t in zip(self.weights, self.targets):
            # t += ... changes the local variable t in place, we must
            # use _set_value instead.
            t._set_value(t._get_value() + w * delta)

        self.prev_value = value


class ExprTask(Task):
    """
    Task that evaluates an expression `expr` and stores the result in `target`.

    Parameters
    ----------
    target: MutableRef
        The target reference where the result of the expression will be stored.
    expr: BaseRef
        The expression to be evaluated.
    """

    def __init__(self, target: MutableRef, expr: BaseRef):
        self.taskid = target
        self.targets = target._get_dependencies()
        self.dependencies = expr._get_dependencies()
        self.expr = expr

    def __repr__(self):
        return f"{self.taskid} = {self.expr}"

    def run(self):
        value = self.expr._get_value()
        self.taskid._set_value(value)

    def info(self):
        """Print information about the task."""
        print(f"#  {self.taskid}._expr._get_dependencies()")
        for pp in self.expr._get_dependencies():
            print(f"   {pp} = {pp._get_value()}")
        print()


class DepEnv:
    """Proxy for modification of refs and access to the underlying data source.

    A convenience class that can be passed both the original data object and
    a reference to it, so that deferred expressions can be assigned to the ref
    object, but data accesses are performed on the original data source.

    Parameters
    ----------
    data: object
        The original data object.
    ref: Ref
        The reference to the original data object.
    """

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
    """Value dependency manager.

    A manager registers external Python objects with Xdeps and orchestrates the
    actions performed by Xdeps. It keeps track of the dependencies between
    references and tasks, and executes tasks when their dependencies change.
    The graph of dependencies is stored in the following mappings:

    Properties
    ----------
    tasks: dict
        Maps taskid (in case of expressions that's target ref) to task.
    rdeps: dict
        Maps a ref to the set of all refs that (directly) depend on it.
    rtasks: dict
        Maps a task (identified by a taskid) to the set of all tasks (identified
        by their taskids) whose dependencies are the targets of the given task.
    deptasks: dict
        Maps a ref to all tasks that have ref as a direct dependency.
    tartasks: dict
        Maps ref to all tasks that have ref as direct target.
    containers: dict
        Maps label to the controlled container.
    """

    def __init__(self):
        self.tasks = {}
        self.containers = {}
        self.rdeps = defaultdict(RefCount)
        self.rtasks = defaultdict(RefCount)
        self.deptasks = defaultdict(RefCount)
        self.tartasks = defaultdict(RefCount)
        self._tree_frozen = False

    def ref(self, container=None, label="_"):
        """Return a ref to an instance (or dict) associated to a label.

        Label must be unique.
        """
        if container is None:
            container = AttrDict()
        objref = Ref(container,  label, self)
        assert label not in self.containers
        self.containers[label] = objref
        return objref

    def set_value(self, ref, value):
        """Set a value pointed by a ref and execute all tasks that depends on ref.

        If the value is a Ref, create a new task from the ref.
        """
        logger.info("set_value %s %s", ref, value)
        if ref in self.tasks:
            # TODO: Here we assume that if a ref identifies a task, it must be
            #  it's (only) target. What if that's not true?
            self.unregister(ref)
        if isinstance(value, BaseRef):  # value is an expression
            self.register(ExprTask(ref, value))
            value = value._get_value()  # to be updated
        ref._set_value(value)
        self._run_tasks(self.find_tasks(ref._get_dependencies()))

    def _run_tasks(self, tasks):
        for task in tasks:
            logger.info("Run %s", task)
            task.run()

    def register(self, task):
        """Register a new task identified by taskid"""
        # logger.info("register %s",taskid)
        if self._tree_frozen:
            raise ValueError("Expressions and references cannot be changed "
                             "because the tree is frozen (e.g. because "
                             "a variables cache is active)")
        taskid = task.taskid
        self.tasks[taskid] = task
        for dep in task.dependencies:
            # logger.info("%s have an impact on %s",dep,task.targets)
            self.rdeps[dep].extend(task.targets)
            # logger.info("%s is used by T:%s",dep,taskid)
            self.deptasks[dep].append(taskid)
            for deptask in self.tartasks[dep]:
                # logger.info("%s modifies deps of T:%s",deptask,taskid)
                self.rtasks[deptask].append(taskid)

        for tar in task.targets:
            # logger.info("%s is modified by T:%s",tar,taskid)
            self.tartasks[tar].append(taskid)
            other = self.deptasks[tar]
            for deptask in other:
                # logger.info("T:%s modifies deps of T:%s",taskid,deptask)
                self.rtasks[taskid].append(deptask)

    def unregister(self, taskid):
        """Unregister the task identified by taskid"""
        if self._tree_frozen:
            raise ValueError("Expressions and references cannot be changed "
                             "because the tree is frozen (e.g. because "
                             "a variables cache is active)")
        task = self.tasks[taskid]
        for dep in task.dependencies:
            for target in task.targets:
                if target in self.rdeps[dep]:
                    self.rdeps[dep].remove(target)
            if taskid in self.rtasks[dep]:
                self.rtasks[dep].remove(taskid)
            if taskid in self.deptasks[dep]:
                self.deptasks[dep].remove(taskid)
        for tar in task.targets:
            self.tartasks[tar].remove(taskid)
        if taskid in self.rtasks:
            del self.rtasks[taskid]
        del self.tasks[taskid]

    def freeze_tree(self):
        """Freeze the tree of expressions and references."""
        self._tree_frozen = True

    def unfreeze_tree(self):
        """Unfreeze the tree of expressions and references."""
        self._tree_frozen = False

    def find_deps(self, start_set):
        """Find all refs that depend on ref in `start_set`."""
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

    def iter_expr_tasks_owner(self, ref):
        """Return all ExprTask defintions that write registered container"""
        for t in self.find_tasks():
            # TODO check for all targets or limit to ExprTask
            if _check_root_owner(t.taskid, ref):
                yield str(t.taskid), str(t.expr)

    def copy_expr_from(self, mgr, name, bindings=None):
        """
        Copy expression from another manager

        name: one of toplevel container in mgr
        bindings: dictionary mapping old container names into new container refs
        """
        ref = mgr.containers[name]
        if bindings is None:
            cmbdct = self.containers
        else:
            cmbdct = dct_merge(self.containers, bindings)
        self.load(mgr.iter_expr_tasks_owner(ref), cmbdct)

    def mk_fun(self, name, **kwargs):
        """Write a python function that executes a set of tasks in order of dependencies:
        name: name of the functions
        kwargs:
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

    def gen_fun(self, name, **kwargs):
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

    def plot_deps(self, start=None, **kwargs):
        """Plot a graph of task and target dependencies from start.

        Possible backend:
            mpl: generate a figure in matplotlib
            os: generate a file /tmp/out.png and use `display` to show it
            ipy: use Ipython facility for Jupyter notebooks

        ftype: png, svg, or pdf
        """
        from pydot import Dot, Node, Edge

        if start is None:
            start = list(self.rdeps)
        pdot = Dot("g", graph_type="digraph", rankdir="LR")
        for task in self.find_tasks(start):
            tn = Node(str(task), label=str(task.taskid), shape="oval")
            pdot.add_node(tn)
            for tt in task.targets:
                pdot.add_node(Node(str(tt), shape="box"))
                pdot.add_edge(Edge(tn, str(tt), color="blue"))
            for tt in task.dependencies:
                pdot.add_node(Node(str(tt), shape="box"))
                pdot.add_edge(Edge(str(tt), tn, color="blue"))

        plot_pdot(pdot, **kwargs)
        return pdot

    def plot_tasks(self, start=None, **kwargs):
        """Plot a graph of task dependencies

        Possible backend:
            mpl: generate a figure in matplotlib
            os: generate a file /tmp/out.png and use `display` to show it
            ipy: use Ipython facility for Jupyter notebooks

        ftype: png, svg, or pdf
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

        plot_pdot(pdot, **kwargs)
        return pdot

    def dump(self):
        """Dump in json all ExprTask defined in the manager"""
        data = [
            (str(tt.taskid), str(tt.expr))
            # for t in self.find_tasks(self.rdeps)
            for tt in self.tasks.values()
            if isinstance(tt, ExprTask)
        ]
        return data

    def load(self, dump, dct=None):
        """Reload the expressions in `dump` using container in `dct`

        dump: List[Tuple[MutableRef, ARef]]
            List of pairs representing an expression lhs = rhs for each pair
            (lhs, rhs).
        dct: Optional[Dict[str, Ref]]
            Dictionary of named references of containers, if unspecified
            assume that all containers are in the manager.
        """

        if dct is None:
            dct = self.containers
        for lhs, rhs in dump:
            lhs = eval(lhs, {}, dct)
            rhs = eval(rhs, {}, dct)
            task = ExprTask(lhs, rhs)
            if lhs in self.tasks:
                self.unregister(lhs)
            self.register(task)

    def newenv(self, label="_", data=None):
        """Create a ref with a DepEnv environment in the manager."""
        if data is None:
            data = AttrDict()
        ref = self.ref(data, label=label)
        return DepEnv(data, ref)

    def refattr(self, container=None, label="_"):
        """Create a ref which translates attribute access to item access.

        Useful for accessing globals() like dictionaries.
        """
        if container is None:
            container = AttrDict()
        objref = ObjectAttrRef(container, label, self)
        assert label not in self.containers
        self.containers[label] = objref
        return objref

    def cleanup(self):
        """Remove empty sets from dicts."""
        for dct in self.rdeps, self.rtasks, self.deptasks, self.tartasks:
            for kk, ss in list(dct.items()):
                if len(ss) == 0:
                    del dct[kk]

    def copy(self):
        """Create a copy of in new manager."""
        other = Manager()
        other.containers = deepcopy(self.containers)
        other.tasks = deepcopy(self.tasks)
        other.rdeps = deepcopy(self.rdeps)
        other.rtasks = deepcopy(self.rtasks)
        other.deptasks = deepcopy(self.deptasks)
        other.tartasks = deepcopy(self.tartasks)
        return other

    def clone(self):
        """Regenerate a new manager.

        This differs to copy in that the internal data structures are
        regenerated from the tasks, instead of being copied. This ensures
        the consistency of the new manager at the expense of performance.
        """
        other = Manager()
        other.containers.update(self.containers)
        for task in self.tasks.values():
            other.register(task)
        other.cleanup()
        return other

    def verify(self, dcts=("rdeps", "rtasks", "deptasks", "tartasks")):
        """Verify the consistency of the manager.

        Parameters
        ----------
        dcts: Iterable[str]
            Names of the internal data structures to check.
        """
        self.cleanup()
        other = self.clone()
        for dct in dcts:
            odct = getattr(other, dct)
            sdct = getattr(self, dct)
            for kk, ss in list(sdct.items()):
                if set(ss) != set(odct[kk]):
                    print(f"{dct}[{kk}] not consistent")
                    print(f"{dct}[{kk}] self - check:", set(ss) - set(odct[kk]))
                    print(f"{dct}[{kk}] check - self:", set(odct[kk]) - set(ss))
                    raise (ValueError(f"{self} is not consistent in {dct}[{kk}]"))

    def refresh(self):
        """Regenerate the internal data structures from the tasks."""
        self.rdeps = defaultdict(RefCount)
        self.rtasks = defaultdict(RefCount)
        self.deptasks = defaultdict(RefCount)
        self.tartasks = defaultdict(RefCount)
        for task in self.tasks.values():
            self.register(task)
        self.cleanup()
