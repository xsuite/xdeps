import numpy as np

from xdeps import Table

data = {
    "name": np.array(["ip1", "ip2", "ip2", "ip3", "tab$end"]),
    "s": np.array([1.0, 2.0, 2.1, 3.0, 4.0]),
    "betx": np.array([4.0, 5.0, 5.1, 6.0, 7.0]),
    "bety": np.array([2.0, 3.0, 3.1, 4.0, 9.0]),
}


t = Table(data)


def test_len():
    assert len(t.betx) == len(data["betx"])


def test_getitem_col():
    assert id(t["betx"]) == id(data["betx"])
    assert t["betx+sqrt(bety)"][1] == (t.betx + np.sqrt(t.bety))[1]


def test_getitem_col_row():
    assert t["betx", 0] == data["betx"][0]
    assert t["betx", "ip1"] == data["betx"][0]
    assert t["betx", "ip2"] == data["betx"][1]
    assert t["betx", ("ip2", 0)] == data["betx"][1]
    assert t["betx", "ip2::1"] == data["betx"][2]
    assert t["betx", ("ip2", 1)] == data["betx"][2]
    assert t["betx", "ip2<<1"] == data["betx"][0]
    assert t["betx", ("ip2", 0, -1)] == data["betx"][0]
    assert t["betx", "ip2::1>>1"] == data["betx"][3]
    assert t["betx", ("ip2", 1, 1)] == data["betx"][3]
    assert t["betx", "ip2::-1"] == data["betx"][2]
    assert t["betx", ("ip2", -1, 1)] == data["betx"][3]
    assert np.all(t["betx", 0:2] == data["betx"][0:2])
    assert np.all(t["betx", "ip1":"ip3"] == data["betx"][0:4])
    assert np.all(t["betx", "ip1":"ip2::1"] == data["betx"][0:3])
    assert np.all(t["betx", ["ip3","ip2::1"]] == data["betx"][[3,2]])


def test_cols():
    assert isinstance(t.cols["betx"], Table)
    assert t.cols["betx", "bety"].betx[0] == t.betx[0]


def test_row_selection():
    assert t.rows[t.s > 1, 1].betx[0] == data["betx"][t.s > 1][1]


def test_row_selection_names():
    assert t.rows["ip1"].betx[0] == data["betx"][0]
    assert t.rows["ip[23]"].betx[0] == data["betx"][1]
    assert t.rows["ip.*::1"].betx[0] == data["betx"][1]
    assert t.rows["notthere"]._nrows == 0
    assert t.rows[["ip1", "ip2"]].betx[0] == data["betx"][0]


def test_row_selection_names_with_rows():
    assert t.rows["ip2"].betx[0] == data["betx"][1]
    assert t.rows["ip[23]"].betx[0] == data["betx"][1]
    assert t.rows["ip.*::1"].betx[0] == data["betx"][1]
    assert t.rows["notthere"]._nrows == 0
    assert t.rows[["ip1", "ip2"]].betx[1] == data["betx"][1]


def test_row_selection_ranges():
    assert t.rows[ 1:4:3].betx[0] == data["betx"][1]
    assert t.rows[ 1.5:2.5:"s"].betx[0] == data["betx"][1]
    assert t.rows[ "ip1":"ip3"].betx[2] == data["betx"][2]
    assert t.rows[ "ip2::1<<1":"ip2::1>>1"].betx[0] == data["betx"][1]
    assert t.rows[ "ip1":"ip3":"name"].betx[0] == data["betx"][0]
    assert t.rows[None].betx[0] == data["betx"][0]
    assert t.rows[:].betx[0] == data["betx"][0]


def test_row_selection_ranges_with_rows():
    assert t.rows[1:4:3].betx[0] == data["betx"][1]
    assert t.rows[1.5:2.5:"s"].betx[0] == data["betx"][1]
    assert t.rows["ip1":"ip3"].betx[2] == data["betx"][2]
    assert t.rows["ip2>>-1":"ip2>>+1"].betx[0] == data["betx"][0]
    assert t.rows["ip1":"ip3":"name"].betx[0] == data["betx"][0]
    assert t.rows[None].betx[0] == data["betx"][0]
    assert t.rows[:].betx[0] == data["betx"][0]


def test_numpy_string():
    tab = Table(dict(name=np.array(["a", "b$b"]), val=np.array([1, 2])))
    assert tab["val", tab.name[1]] == 2
