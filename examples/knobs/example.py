import xtrack as xt

lhc=xt.Environment.from_json("../acc-models-lhc/xsuite/lhc.json")

from knobs import Knobs
knobs=Knobs(lhc)

knob=knobs.get_by_probing("on_x1_h")
knob=knobs.get_by_xdeps("on_x1_h")
knob=knobs.get_by_weight_names("on_x1_h")

knob2=knob.copy("myknob")

knobs.delete("myknob",verbose=True)
knobs.create(knob2)
knobs.get_by_probing("myknob")
knobs.check(knob2)
knobs.delete("myknob",verbose=True)
knobs.get_by_probing("myknob")
knobs.get_by_probing("on_x1_h")

knobs.create(knob2)
knobs.get_by_probing("myknob")
knob2.weights['acbxh1.r1']=0
knobs.update(knob2)
knobs.get_by_probing("myknob")
knobs.get_xdeps("myknob").weights['acbxh1.r1']


