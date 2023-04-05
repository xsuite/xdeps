import xtrack as xt
line = xt.Line.from_json(
    './xtrack/test_data/hllhc15_noerrors_nobb/line_w_knobs_and_particle.json')

manager = line._var_management['manager']

for tt in list(manager.tasks):
    print(tt)
    manager.unregister(tt )
    manager.verify()

manager.unregister(line.vars['cosi0'])
manager.verify()
