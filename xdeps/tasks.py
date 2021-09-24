from dataclasses import dataclass, field
import logging

from .refs import AttrRef, CallRef, Ref, ItemRef, ObjectRef
from .utils import os_display_png

logger=logging.getLogger(__name__)

def _traverse(ref, rdeps, lst, st):  # breath first sorting
    if ref in rdeps:
        for dep in rdeps[ref]:
            if dep not in st:
                lst.append(dep)
                st.add(dep)
            _traverse(dep.target, rdeps, lst, st)
    return lst


def traverse(ref, rdeps):
    return _traverse(ref, rdeps, [], set())

class FuncWrapper:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        return CallRef(self.func, args, tuple(kwargs.items()))


class Task:
    taskid: object
    action: object
    targets: object
    dependencies: tuple

    def __repr__(self):
        return f"<Task {taskid}:{self.dependencies}=>{self.targets}>"

    def run(self):
        logger.info(f"Run {self}")
        return self.action()


class ExprTask:
    def __init__(self,target,expr):
        self.taskid=target
        self.targets=[target]
        self.action=expr._get_value
        self.dependencies=expr._get_dependencies()
        self.expr=expr


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class DepManager:
    def __init__(self):
        self.tasks= {}
        self.deps = set()
        self.rdeps = {}

    def ref(self,m=None):
        if m is None:
            m=AttrDict()
        return ObjectRef(m,self)

    def set_value(self, ref, value):
        logger.info(f"set_value {ref} {value}")
        if isinstance(value,Ref):
            self.register(ref,ExprTask(ref,value))
        else:
            ref._set_value(value)

    def del_value(self,ref):
        self.unregister(ref)

    def register(self,taskid,task):
        self.tasks[taskid]=task

    def unregister(self,taskid):
        del self.tasks[taskid]

    def to_pydot(self):
        from pydot import Dot, Node, Edge
        pdot = Dot("g", graph_type="digraph",rankdir="LR")
        for taskid, task in self.tasks.items():
            tn=Node(str(taskid), shape="circle")
            pdot.add_node(tn)
            for tt in task.targets:
                pdot.add_node(Node(str(tt), shape="square"))
                pdot.add_edge(Edge(tn, str(tt), color="blue"))
            for tt in task.dependencies:
                pdot.add_node(Node(str(tt), shape="square"))
                pdot.add_edge(Edge(str(tt),tn, color="blue"))
        os_display_png(pdot.create_png())
        return pdot


    def class_(self, cls):
        cls_setattr=cls.__setattr__
        def _silent_setattr(self, name, value):
            cls_setattr(self, name, value)

        def __getattr__(self, name):
            if name.endswith("_"):
                orig = name[:-1]
                if hasattr(self, orig):
                    return AttrRef(self, orig, self._manager)
            raise AttributeError

        def __setattr__(self, name, value):
            ref = AttrRef(self, name, self._manager)
            if isinstance(value, Ref):
                self._dep_remove(name)
                dependencies = value._get_dependencies()
                dep = Rule(ref, tuple(dependencies), value._get_value)
                self._dep_add(name, dep)
                value = value._get_value()
            self._silent_setattr(name, value)
            self._manager.apply_set(ref)

        def _dep_add(self, name, dep):
            if not hasattr(self, "_deps"):
                self._deps = {}
            self._deps[name] = dep
            self._manager.register(dep)

        def _dep_remove(self, name):
            if hasattr(self, "_deps") and name in self._deps:
                self._manager.unregister(self._deps[name])
                del self._deps[name]

        for ff in (_silent_setattr, __setattr__, _dep_add, _dep_remove, __getattr__, __invert__):
            setattr(cls, ff.__name__, ff)

        setattr(cls, "_manager", self)
        return cls

    def fun_(self, fun):
        return FuncWrapper(fun)


manager=DepManager()
