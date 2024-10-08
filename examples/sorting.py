# copyright ############################### #
# This file is part of the Xdeps Package.   #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from collections import deque
from functools import reduce

def _dfs(graph, source, stack, visited):
    visited.add(source)

    for neighbour in graph.get(source,[]):
        if neighbour not in visited:
            _dfs(graph, neighbour, stack, visited)

    stack.appendleft(source)

def toposort(graph,start=None):
    stack = deque()
    visited = set()

    if start is None:
        start=reduce(set.union, graph.values(),graph.keys())

    for vertex in start:
        if vertex not in visited:
            _dfs(graph, vertex, stack, visited)

    return list(stack)

def toposort2(graph, start):
    seen = set()
    stack = []
    order = []
    q=start[:]
    while q:
        v = q.pop()
        if v not in seen:
            seen.add(v) # no need to append to path any more
            q.extend(graph.get(v,[]))
            while stack and v not in graph.get(stack[-1],[]):
                order.insert(0,stack.pop())
            stack.append(v)

    return (stack + order)   # new return value!




def depsort(deps):
    data={k: set(v) for k,v in deps.items()}
    extra_items_in_deps = reduce(set.union, data.values()) - set(data.keys())
    data.update({item:set() for item in extra_items_in_deps})
    while True:
        ordered = set(item for item,dep in data.items() if not dep)
        if not ordered:
            break
        yield list(ordered)
        data = {item: (dep - ordered) for item,dep in data.items()
                if item not in ordered}
    assert not data, "A cyclic dependency exists amongst %r" % data



def reverse_graph(dep_graph):
    """
    dep[4]=[3,1] means 4 depends on 3 and 1
    rdep[3]=[4]  means 3 is needed by 4
    rdep[1]=[4]  means 3 is needed by 4
    """
    rdeps={}
    for t,deps in dep_graph.items():
        for dd in deps:
          rdeps.setdefault(dd,[]).append(t)
    return rdeps

if __name__=="__main__":
    deps={
        3:[1,2],
        4:[1,3],
        5:[3,4],
    }

    rdeps=reverse_graph(deps)
    print(rdeps)
    print(toposort(rdeps))


    print(toposort2(rdeps,[1]))
    print(toposort2(rdeps,[2]))
    print(toposort2(rdeps,[1,2]))


    data = {
        'des_system':   'std synopsys std_cell_lib dw02 dw01 ramlib ieee'.split(),
        'dw01':         'ieee dware gtech'.split(),
        'dw02':         'ieee dware'.split(),
        'dw03':         'std synopsys dware dw02 dw01 ieee gtech'.split(),
        'dw04':         'ieee dw01 dware gtech'.split(),
        'dw05':         'ieee dware'.split(),
        'dw06':         'ieee dware'.split(),
        'dw07':         'ieee dware'.split(),
        'dware':        'ieee'.split(),
        'gtech':        'ieee'.split(),
        'ramlib':       'std ieee'.split(),
        'std_cell_lib': 'ieee'.split(),
        'synopsys':     (),
        }

    print(list(depsort(data)))

    rdeps=reverse_graph(data)
    print(toposort2(rdeps,['ieee']))

