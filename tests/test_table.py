import numpy as np
import pytest

from xdeps import Table

data = {
    "name": np.array(["ip1", "ip2", "ip2", "ip3", "tab$end"]),
    "s": np.array([1.0, 2.0, 2.1, 3.0, 4.0]),
    "betx": np.array([4.0, 5.0, 5.1, 6.0, 7.0]),
    "bety": np.array([2.0, 3.0, 3.1, 4.0, 9.0]),
}


## Table tests

t = Table(data)

def test_table_initialization():
    # Valid initialization
    data = {"name": np.array(["a", "b", "c"]), "value": np.array([1, 2, 3])}
    table = Table(data)
    assert len(table) == 3
    assert table._col_names == ["name", "value"]

    # Invalid initialization (non-numpy array)
    data_invalid = {"name": ["a", "b", "c"], "value": [1, 2, 3]}
    try:
        Table(data_invalid)
    except ValueError as e:
        assert str(e) == "Column `name` is not a numpy array"

    # Invalid initialization (different column lengths)
    data_invalid_length = {"name": np.array(["a", "b"]), "value": np.array([1, 2, 3])}
    try:
        Table(data_invalid_length)
    except ValueError as e:
        assert str(e) == "Columns have different lengths"


def test_table_split_names():
    assert t._split_name_count_offset("example") == ("example", None, 0)
    assert t._split_name_count_offset("example::5") == ("example", 5, 0)
    assert t._split_name_count_offset("example<<3") == ("example", None, -3)
    assert t._split_name_count_offset("example::5<<3") == ("example", 5, -3)
    assert t._split_name_count_offset("example::5<<-3") == ("example", 5, 3)
    assert t._split_name_count_offset("example<<-3") == ("example", None, 3)
    assert t._split_name_count_offset("example::10<<0") == ("example", 10, 0)
    assert t._split_name_count_offset("example::0<<-1") == ("example", 0, 1)
    assert t._split_name_count_offset("example<<0") == ("example", None, 0)
    assert t._split_name_count_offset("example::0<<0") == ("example", 0, 0)
    assert t._split_name_count_offset("example>>3") == ("example", None, 3)
    assert t._split_name_count_offset("example::5>>3") == ("example", 5, 3)
    assert t._split_name_count_offset("example::5>>-3") == ("example", 5, -3)
    assert t._split_name_count_offset("example>>-3") == ("example", None, -3)
    assert t._split_name_count_offset("example::10>>0") == ("example", 10, 0)
    assert t._split_name_count_offset("example::0>>-1") == ("example", 0, -1)
    assert t._split_name_count_offset("example>>0") == ("example", None, 0)
    assert t._split_name_count_offset("example::0>>0") == ("example", 0, 0)


def test_table_len():
    assert len(t.betx) == len(data["betx"])


def test_table_getitem_col():
    assert id(t["betx"]) == id(data["betx"])
    assert t["betx+sqrt(bety)"][1] == (t.betx + np.sqrt(t.bety))[1]
    try:
        t["bbb"]
    except NameError as e:
        assert str(e) == "name 'bbb' is not defined"


def test_table_getitem_col_row():
    assert t["betx", 0] == data["betx"][0]
    assert t["betx", "ip1"] == data["betx"][0]
    assert t["betx", "ip2"] == data["betx"][1]
    assert t["betx", ("ip2", 0)] == data["betx"][1]
    assert t["betx", ("ip2", 1)] == data["betx"][2]
    assert t["betx", ("ip2",-1)] == data["betx"][2]
    assert t["betx", "ip2::0"] == data["betx"][1]
    assert t["betx", "ip2::1"] == data["betx"][2]
    assert t["betx", "ip2::-1"] == data["betx"][2]
    assert t["betx", ("ip2", 1)] == data["betx"][2]
    assert t["betx", "ip2<<1"] == data["betx"][0]
    assert t["betx", ("ip2", 0, -1)] == data["betx"][0]
    assert t["betx", "ip2::1>>1"] == data["betx"][3]
    assert t["betx", ("ip2", 1, 1)] == data["betx"][3]
    assert t["betx", "ip2::-1"] == data["betx"][2]
    assert t["betx", ("ip2", -1, 1)] == data["betx"][3]
    assert np.all(t["betx", 0:2] == data["betx"][0:2])
    assert np.all(t["betx", "ip1":"ip3"] == data["betx"][0:4])
    assert np.all(t["betx", "ip1":"ip3":"name"] == data["betx"][0:4])
    assert np.all(t["betx", "ip1":"ip2::1"] == data["betx"][0:3])
    assert np.all(t["betx", ["ip3", "ip2::1"]] == data["betx"][[3, 2]])
    assert np.all(t["betx", [3, 2]] == data["betx"][[3, 2]])
    assert np.all(
        t["betx", [True, False, True, False, True]] == data["betx"][[0, 2, 4]]
    )
    with pytest.raises(KeyError) as e:
        t["betx", "notthere"]
        assert str(e) == "Cannot find 'notthere' in column 'name'"


def test_table_getitem_edge_cases():
    with pytest.raises(KeyError) as e:
        t[:]
        assert str(e).startswith("Invalid arguments")
    
    with pytest.raises(KeyError) as e:
        t[()]
        assert str(e).startswith("Empty selection")

    assert t[("betx", 1)] == t["betx",1]

    assert t[("betx",)][2] == t["betx"][2]

    


def test_table_numpy_string():
    tab = Table(dict(name=np.array(["a", "b$b"]), val=np.array([1, 2])))
    assert tab["val", tab.name[1]] == 2


def test_table_column_access():
    data = {"name": np.array(["a", "b", "c"]), "value": np.array([1, 2, 3])}
    table = Table(data)

    # Direct column access
    assert np.array_equal(table["name"], data["name"])
    assert np.array_equal(table["value"], data["value"])

    # Column expression
    assert np.array_equal(table["value * 2"], data["value"] * 2)


def test_table_sort_columns():
    data = {"name": np.array(["a", "b", "c"]), "value": np.array([1, 2, 3])}
    table = Table(data)

    # Sort columns
    assert table._col_names == ["name", "value"]
    table._col_names = ["value", "name"]
    assert table._col_names == ["value", "name"]


def test_table_column_assignment():
    data = {"name": np.array(["a", "b", "c"]), "value": np.array([1, 2, 3])}
    data_orig = {"name": np.array(["a", "b", "c"]), "value": np.array([1, 2, 3])}
    table = Table(data)

    # Direct column assignment
    table["value"] = table["value"] * 2
    assert np.array_equal(table["value"], data["value"])
    assert np.array_equal(table["value"], data_orig["value"] * 2)

    # Direct column assignment
    table["value"] = 1
    assert np.array_equal(table["value"], np.array([1, 1, 1]))
    assert np.array_equal(data["value"], np.array([1, 1, 1]))


def test_table_del_column():
    data = {"name": np.array(["a", "b", "c"]), "value": np.array([1, 2, 3])}
    table = Table(data)

    # Delete column
    del table["value"]
    assert "value" not in table._col_names


def test_table_from_methods():
    data = {"name": np.array(["a", "b", "c"]), "value": np.array([1, 2, 3])}
    table = Table(data)

    # Show method
    table.show()

    # From pandas
    import pandas as pd

    df = pd.DataFrame(data)
    table_from_pandas = Table.from_pandas(df)
    assert len(table_from_pandas) == 3

    # From CSV
    df.to_csv("test.csv", index=False)
    table_from_csv = Table.from_csv("test.csv")
    assert len(table_from_csv) == 3

    # From rows
    rows = [
        {"name": "a", "value": 1},
        {"name": "b", "value": 2},
        {"name": "c", "value": 3},
    ]
    table_from_rows = Table.from_rows(rows)
    assert len(table_from_rows) == 3

    # To pandas
    df_from_table = table._df
    assert df_from_table.equals(df)


def test_table_concatenate():
    data1 = {"name": np.array(["a", "b", "c"]), "value": np.array([1, 2, 3])}
    data2 = {"name": np.array(["d", "e", "f"]), "value": np.array([4, 5, 6])}
    table1 = Table(data1)
    table2 = Table(data2)

    # Concatenate
    concatenated_table = table1 + table2
    assert len(concatenated_table) == 6
    assert np.array_equal(concatenated_table["value"], np.array([1, 2, 3, 4, 5, 6]))

def test_table_transpose():
    data = {"name": np.array(["a", "b", "c"]), "value": np.array([1, 2, 3])}
    table = Table(data)

    transposed_table = table._t
    assert transposed_table._col_names == ["columns", "row0", "row1", "row2"]


def test_edge_cases():
    # Empty table
    data_empty = {"name": np.array([]), "value": np.array([])}
    table_empty = Table(data_empty)
    assert len(table_empty) == 0

    # Non-uniform column lengths
    data_non_uniform = {"name": np.array(["a", "b"]), "value": np.array([1, 2, 3])}
    try:
        Table(data_non_uniform)
    except ValueError as e:
        assert str(e) == "Columns have different lengths"


def test_table_show():
    data = {"name": np.array(["a", "b", "c"]), "value": np.array([1, 2, 3])}
    table = Table(data)
    assert table.show(output=str) == "name value\na        1\nb        2\nc        3"


## Table cols tests

def test_cols():
    assert isinstance(t.cols["betx"], Table)
    assert t.cols["betx", "bety"].betx[0] == t.betx[0]

def test_cols_iter():
    data = {
        "name": np.array(["a", "b", "c"]),
        "c1": np.array([1, 2, 3]),
        "c2": np.array([4, 5, 6]),
        "s1":1,
        "s2":2,
    }
    with pytest.raises(ValueError) as err:
        Table(data)
        assert str(err) == "Column `s1` is not a numpy array"

    table = Table(data, col_names=["name", "c1", "c2"])

    for col in table.cols:
        assert col in table._col_names

    assert list(table.cols) == ["name", "c1", "c2"]

def test_cols_get_index_unique():
    assert len(set(t.cols.get_index_unique())) == len(t)

def test_cols_expression():
    data = {
        "name": np.array(["a", "b", "c"]),
        "c1": np.array([1, 2, 3]),
        "c2": np.array([4, 5, 6]),
    }
    table = Table(data)
    assert np.array_equal(table.cols["c1+c2"]["c1+c2"], data["c1"] + data["c2"])
    assert np.array_equal(table.cols["c1+1.34"]["c1+1.34"], table["c1"]+1.34)

## Table rows tests

def test_rows_get_index():
    assert t.rows.get_index("ip2::1") == 2
    assert t.rows.get_index("ip2") == 1
    assert t.rows.get_index(("ip2",1)) == 2

def test_rows_is_repeated():
    assert not t.rows.is_repeated("ip1")
    assert t.rows.is_repeated("ip2")
    assert not t.rows.is_repeated("ip3")
    assert not t.rows.is_repeated("tab$end")

def test_rows_selection_indices():
    assert t.rows[1].betx[0] == data["betx"][1]
    assert np.array_equal(t.rows[[2, 1]].betx, data["betx"][[2, 1]])


def test_rows_selection_names():
    assert t.rows["ip2"].betx[0] == data["betx"][1]
    assert t.rows["ip[23]"].betx[0] == data["betx"][1]
    assert t.rows["ip.*::1"].betx[0] == data["betx"][2]
    assert len(t.rows["notthere"]) == 0
    assert t.rows[["ip1", "ip2"]].betx[1] == data["betx"][1]


def test_rows_selection_ranges():
    assert t.rows[1:4:3].betx[0] == data["betx"][1]
    assert t.rows[1.5:2.5:"s"].betx[0] == data["betx"][1]
    assert t.rows["ip1":"ip3"].betx[2] == data["betx"][2]
    assert t.rows["ip2::1<<1":"ip2::1>>1"].betx[0] == data["betx"][1]
    assert t.rows["ip1":"ip3":"name"].betx[0] == data["betx"][0]
    assert t.rows[:].betx[0] == data["betx"][0]


def test_rows_multiple_selection():
    assert t.rows[t.s > 1, 1].betx[0] == data["betx"][t.s > 1][1]

def test_rows_mask():
    assert np.array_equal(t.betx[t.rows.mask[:, t.s > 1]], data["betx"][t.s > 1])
    assert np.array_equal(t.betx[t.rows.mask[[2, 1]]], data["betx"][[1, 2]])



## Table without index

def test_table_no_index():
    data = {
        "name": np.array(["a", "b", "c"]),
        "c1": np.array([1, 2, 3]),
        "c2": np.array([4, 5, 6]),
    }
    table = Table(data, index=None)
    assert "name" in table._col_names
    assert "c1" in table._col_names
    assert "c2" in table._col_names
    assert table._index is None

    assert table["c1", 1] == data["c1"][1]
    assert np.array_equal(table["c1"], data["c1"])
    assert np.array_equal(table["c1", [1, 2]], data["c1"][[1,2]])




