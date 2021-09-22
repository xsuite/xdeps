from collections import deque

def dfs(graph, source, stack, visited):
    visited.add(source)

    for neighbour in graph.get(source,[]):
        if neighbour not in visited:
            dfs(graph, neighbour, stack, visited)

    stack.appendleft(source)

def topsort(graph):
    stack = deque()
    visited = set()

    for vertex in graph.keys():
        if vertex not in visited:
            dfs(graph, vertex, stack, visited)

    return list(stack)


def resolve(graph, start):
    seen = set()
    stack = []
    order = []    # order will be in reverse order at first
    while q:
        v = q.pop()
        if v not in seen:
            seen.add(v) # no need to append to path any more
            q.extend(graph.get(v,[]))

            while stack and v not in graph.get(stack[-1],[]): # new stuff here!
                order.append(stack.pop())
            stack.append(v)

    print(q)
    return (stack + order[::-1])[len(start):]   # new return value!


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

if __name__ == "__main__":
    deps={
        3:[1,2],
        4:[1,3],
        5:[3,4],
    }

    rdeps=reverse_graph(deps)
    print(rdeps)
    print(topsort(rdeps))


    print(resolve(rdeps,[1]))
    print(resolve(rdeps,[2]))
    print(resolve(rdeps,[1,2]))


    deps={
        3:[1,2],
        4:[1,3],
        5:[3,4],
        1:[5],
    }
    rdeps=reverse_graph(deps)
    print(topsort(rdeps))
    print(resolve(rdeps,[1]))

