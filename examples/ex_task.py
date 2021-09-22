from tasks import TaskList


tl=TaskList()

tl.add_task("a",[2,3],[0,1])
tl.add_task("b",[5],[4,2])
tl.add_task("c",[6],[5,3])
tl.add_task("d",[5],[0,2])
tl.add_task("f",[0],[6])
tl.del_task("f")
tl.to_pydot()

