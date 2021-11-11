from xsequence.lattice import Lattice
from xsequence.conversion_utils import conv_utils

madx_lattice = conv_utils.create_cpymad_from_file("fcc-ee_h.seq", 120)
lat = Lattice.from_cpymad(madx_lattice, 'l000013')

lat.params['energy'] = 120
lat.line = lat.sequence._get_line()
plat = lat.to_pyat()

import at
plat.radiation_off()
l0,q,qp,l = at.linopt(plat,refpts=range(len(plat)))
ax1,=plot(l.s_pos,l.beta[:,0])
ax2,=plot(l.s_pos,l.beta[:,1])

def update_twiss(plat,elements):
    plat.radiation_off()
    l0,q,qp,l = at.linopt(plat,refpts=range(len(plat)))
    ax1.set_ydata(l.beta[:,0])
    ax2.set_ydata(l.beta[:,1])

import xdeps
import logging
logging.basicConfig(level=logging.INFO)

plat.e=dict( (el.FamName,el) for el in plat)
plat.v = xdeps.utils.AttrDict()

manager = xdeps.Manager()
pref=manager.ref(plat,'plat')

pref.update_twiss=update_twiss
pref.up=pref.update_twiss(pref,pref.e)

pref.v.dk=0
pref.e['qc1l1.1'].K=-0.24949831119187935*(1+pref.v.dk)

pref.v.dk=plat.v.dk+0.0001

manager.plot_tasks(backend='os')


