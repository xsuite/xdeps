import xdeps
import numpy as np

class Knob:
    def __init__(self, name, value=0.0, weights={}):
        self.name = name
        self.value = value
        self.weights = weights  # dict of target_name: weight_value

    def copy(self,new_name=None):
        if new_name is None:
            new_name = self.name
        return Knob(new_name, self.value, self.weights.copy())

    def __repr__(self):
        return f"Knob(name={self.name!r}, value={self.value!r}, weights={self.weights!r})"

def termlist(ex, lst=[]):
    if isinstance(ex, xdeps.refs.AddExpr):
        return lst + termlist(ex._lhs) + termlist(ex._rhs)
    if isinstance(ex, xdeps.refs.SubExpr):
        if isinstance(ex._rhs, xdeps.refs.MulExpr):
            ex = ex._lhs + (-1 * ex._rhs._lhs) * ex._rhs._rhs
        else:
            ex = ex._lhs + (-1) * ex._rhs
        return lst + termlist(ex._lhs) + termlist(ex._rhs)
    else:
        return [ex]


def delete_term(ex, var):
    terms = []
    for term in termlist(ex):
        if isinstance(term, xdeps.refs.MulExpr) and term._rhs == var:
            continue
        else:
            terms.append(term)
    return sum(terms)

def pprint(expr):
    return "\n + ".join([str(t) for t in termlist(expr)])


class Knobs:
    def __init__(self, env, names = None):
        self.env = env
        self.names = [] if names is None else names
    
    @property
    def mgr(self):
        return self.env.ref_manager

    @property
    def _var_values(self):
        return self.env._xdeps_vref._owner

    @property
    def ref(self):
        return self.mgr.containers['vars']
    
    def delete(self, knobname, verbose=False, dry_run=False):
        direct_deps = list(self.mgr.rdeps.get(self.ref[knobname], {}))
        for dd in direct_deps:
            if verbose:
                print(f"Deleting dependency {dd}")
            oldexpr = dd._expr
            newexpr = delete_term(oldexpr, self.ref[knobname])
            if verbose:
                print(f" Old expr: {oldexpr}")
                print(f" New expr: {newexpr}")
            if not dry_run:
                self.mgr.set_value(dd, newexpr)
        direct_deps = list(self.mgr.rdeps.get(self.ref[knobname], {}))
        if not dry_run and len(direct_deps) > 0 :
            print(f"After deletion, knob {knobname} still has dependencies:")
            for dd in direct_deps:
                print(f" - {dd}")
            raise ValueError(f"Knob {knobname} still has dependencies after deletion")        

    def create(self, knob, verbose=False):
        """
        Create the knob in the model, deleting any previous definition.
        """
        knobname=knob.name
        self.delete(knobname, verbose=verbose, dry_run=False)
        if verbose:
            print(f"Creating knob {knobname} = {knob.value:15.6g}")
        self.env[knobname] = knob.value
        for wtarget, value in knob.weights.items():
            wname = f"{wtarget}_from_{knobname}"
            if verbose:
                print(f" Creating weight {wname} = {value:15.6g}")
            self.env[wname] = value
            if verbose:
                print(f"Creating expression {wtarget} += {wname} * {knobname}")
                print(f" Setting weight {wname} = {value:15.6g}")
            self.env.ref[wtarget] += self.env.ref[wname] * self.env.ref[knobname]
            self.env[wname] = value

    def update(self, knob, verbose=False):
        """
        Update the model with the knob weight values

        Check that the knob exists, that is k has dependent targets.
        If it exists, check that has the same structure
        else raise an error.
        """
        check = self.check(knob)
        knobname = knob.name

        if check is False:
            self.check(knob, verbose=True)
            raise ValueError(f"Knob {knobname} has different structure in {self}")
        else:  # update weight variables and expressions only if check is None
            for wtarget, value in knob.weights.items():
                wname = f"{wtarget}_from_{knobname}"
                if verbose and wname in self.env and self.env[wname] != value:
                    print(f"Update {wname} from {self.env[wname]:15.6g} to {value:15.6g}")
                self.env[wname] = value

    def get_by_xdeps(self, name, variant=None, verbose=False):
        if verbose:
            print(f"Getting knob {name} by xdeps with variant={variant}")
        mgr = self.env.ref_manager
        if name not in self.env:
            return None
        ref = self.env.ref[name]
        weight_names = [xx._key for xx in mgr.rdeps[ref]]
        var_values = self._var_values
        weights = {}
        tasks = mgr.tasks
        for wname in weight_names:
            expr = tasks[self.env.ref[wname]].expr
            if verbose:
                print(f"Weight {wname}")
            for term in termlist(expr):
                if isinstance(term, xdeps.refs.MulExpr) and term._rhs == ref:
                    if np.isscalar(term._lhs):
                        value = float(term._lhs)
                    else:
                        value = var_values[term._lhs._key]
                        if verbose:
                            print(f"   Term: {term}")
                            print(f"   Weight: {value}")
                    weights[wname] = value
        value = self.env[ref._key]
        return Knob(name, value, weights)

    def get_by_probing(self, name, variant=None, verbose=False):
        if verbose:
            print(f"Getting knob {name} by probing with variant={variant}")
        weights = {}
        oldvars = self._var_values.copy()
        oldvalue = self._var_values[name]
        self.env[name] = oldvalue + 1
        for k in self._var_values:
            vnew = self._var_values[k]
            if hasattr(vnew, "__sub__"):
                dvar = self._var_values[k] - oldvars[k]
                if dvar != 0:
                    weights[k] = dvar
                    if verbose:
                        print(f"Weight {k} = {dvar:15.6g}")
        del weights[name]
        self.env[name] = oldvalue
        return Knob(name, oldvalue, weights)

    def get_by_weight_names(self, name, variant=None, verbose=False):
        if verbose:
            print(f"Getting knob {name} by weight names with variant={variant}")
        weights = {}
        value = self._var_values[name]
        wname = f"_from_{name}"
        for k in self._var_values:
            if k.endswith(wname):
                weights[k.split(wname)[0]] = self._var_values[k]
                if verbose:
                    print(f"Weight {k.split(wname)[0]} = {self._var_values[k]:15.6g}")
        return Knob(name, value, weights)

    def check(self, knob, verbose=False):
        """
        Return True has the expeceted structure
        Return False has a different structure
        """
        knobname = knob.name
        deps = self.mgr.rdeps.get(self.ref[knobname], {})
        depnames = {dep._key for dep in deps}
        if verbose:
            print(f"Check knob {knobname}")
            for dep in deps:
                print(f"- {dep._key} {dep._expr}")
        if depnames != set(knob.weights.keys()):
            if verbose:
                knob_weights = set(knob.weights.keys())
                print(f"Model weights `{depnames}` != knob weights `{knob_weights}`")
            return False
        else:
            return True
