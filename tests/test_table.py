import numpy as np

from xdeps import Table

data = {
    "name": np.array(["ip1", "ip2", "ip2", "ip3", "tab$end"]),
    "s": np.array([1.0, 2.0, 2.1, 3.0, 4.0]),
    "betx": np.array([4.0, 5.0, 5.1, 6.0, 7.0]),
    "bety": np.array([2.0, 3.0, 3.1, 4.0, 9.0]),
}

t = Table(data)

def test_table_initialization():
    # Valid initialization
    data = {
        "name": np.array(["a", "b", "c"]),
        "value": np.array([1, 2, 3])
    }
    table = Table(data)
    assert len(table) == 3
    assert table._col_names == ["name", "value"]

    # Invalid initialization (non-numpy array)
    data_invalid = {
        "name": ["a", "b", "c"],
        "value": [1, 2, 3]
    }
    try:
        Table(data_invalid)
    except ValueError as e:
        assert str(e) == "Column `name` is not a numpy array"

    # Invalid initialization (different column lengths)
    data_invalid_length = {
        "name": np.array(["a", "b"]),
        "value": np.array([1, 2, 3])
    }
    try:
        Table(data_invalid_length)
    except ValueError as e:
        assert str(e) == "Columns have different lengths"

def test_len():
    assert len(t.betx) == len(data["betx"])

def test_getitem_col():
    assert id(t["betx"]) == id(data["betx"])
    assert t["betx+sqrt(bety)"][1] == (t.betx + np.sqrt(t.bety))[1]
    try:
        t["bbb"]
    except NameError as e:
        assert str(e) == "name 'bbb' is not defined"


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
    assert np.all(t["betx", [3,2]] == data["betx"][[3,2]])


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



def test_column_access():
    data = {
        "name": np.array(["a", "b", "c"]),
        "value": np.array([1, 2, 3])
    }
    table = Table(data)

    # Direct column access
    assert np.array_equal(table["name"], data["name"])
    assert np.array_equal(table["value"], data["value"])

    # Column expression
    assert np.array_equal(table["value * 2"], data["value"] * 2)


def test_row_access():
    data = {
        "name": np.array(["a", "b", "c"]),
        "value": np.array([1, 2, 3])
    }
    table = Table(data)

    # Integer row access
    assert table["value", 0] == 1

    # String row access
    assert table["value", "a"] == 1

    # Tuple row access
    assert table["value", ("a", 0)] == 1

    # Slice row access
    assert np.array_equal(table["value", 0:2], data["value"][0:2])

    # List row access
    assert np.array_equal(table["value", ["a", "b"]], data["value"][0:2])


def test_table_methods():
    data = {
        "name": np.array(["a", "b", "c"]),
        "value": np.array([1, 2, 3])
    }
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
    rows = [{"name": "a", "value": 1}, {"name": "b", "value": 2}, {"name": "c", "value": 3}]
    table_from_rows = Table.from_rows(rows)
    assert len(table_from_rows) == 3

    # Transpose
    transposed_table = table._t
    assert transposed_table._col_names == ["columns", "row0", "row1", "row2"]

    # Concatenate
    concatenated_table = table + table
    assert len(concatenated_table) == 6

    # To pandas
    df_from_table = table._df
    assert df_from_table.equals(df)


def test_edge_cases():
    # Empty table
    data_empty = {
        "name": np.array([]),
        "value": np.array([])
    }
    table_empty = Table(data_empty)
    assert len(table_empty) == 0

    # Non-uniform column lengths
    data_non_uniform = {
        "name": np.array(["a", "b"]),
        "value": np.array([1, 2, 3])
    }
    try:
        Table(data_non_uniform)
    except ValueError as e:
        assert str(e) == "Columns have different lengths"


