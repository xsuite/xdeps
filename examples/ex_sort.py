from sorting import *

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
print(toposort3(rdeps,['ieee']))

