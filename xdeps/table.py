import os
import pathlib
import re
from typing import Collection
from collections import namedtuple

import numpy as np

gblmath = {"np": np}
for k, fu in np.__dict__.items():
    if type(fu) is np.ufunc:
        gblmath[k] = fu


def is_iterable(obj):
    return hasattr(obj, "__iter__") and not isinstance(obj, str)


def eval_col(col, view):
    try:
        cc = eval(col, gblmath, view)
    except NameError:
        raise KeyError(
            f"Column `{col}` could not be found or " "is not a valid expression"
        )
    return cc


def _to_str(arr, digits, fixed="g", max_len=None):
    """Covert an array to an array of string representations.

    Parameters
    ----------
    arr : array-like
        Array to convert.
    digits : int
        Number of digits to use for floats.
    fixed : str
        If 'g', use general format (total number of digits is `digits`, if
        necessary use exponent notation). If 'f', use fixed point notation
        (exactly `digits` digits after the decimal point).
    max_len : int or None
        If not None, truncate strings to this length.
    """
    if len(arr.shape) > 1:
        # If array of arrays, just show the shape
        out = np.repeat(f"<array of shape {arr.shape[1:]}>", arr.shape[0])
    elif arr.dtype.kind in "SU":
        # Keep strings
        out = arr
    elif arr.dtype.kind in "iu":
        # Print out integers in full
        out = np.char.mod("%d", arr)
    elif arr.dtype.kind in "f":
        # Format floats.
        # First generate a format string like "%.<digits>g", then use it on
        # each element.
        add = 3 if fixed == "f" else 7
        fmt = "%%%d.%d%s" % (digits + add, digits, fixed)
        out = np.char.mod(fmt, arr)
    elif arr.dtype.kind == "O" and isinstance(arr[0], Collection):
        # If array of collections (array with dtype=object) or list, give shape
        lengths = []
        for entry in arr:
            if isinstance(entry, str):
                lengths.append(entry)
            elif isinstance(entry, np.ndarray):
                lengths.append(f"<array of shape {entry.shape}>")
            elif isinstance(entry, Collection):
                lengths.append(f"<collection of length {len(entry)}>")
            else:
                lengths.append(str(entry))
        out = np.array(lengths)
    else:
        # Any other flat array: stringify.
        out = arr.astype("U")

    if max_len is not None:
        old_out = out.copy()
        out = []
        for ii in range(len(old_out)):
            out.append(
                str(
                    old_out[ii][:max_len] + " ..."
                    if len(old_out[ii]) > max_len
                    else old_out[ii]
                )
            )
        out = np.array(out)
    return out


class Indices:
    """Class returing indices of the table.
    t.indices[1] -> row
    t.indices['a'] -> pattern
    r.indices[10:20]-> range
    r.indices['a':'b'] -> name range
    r.indices['a':'b':'myname'] -> name range with 'myname' column
    r.indices[-2:2:'x'] -> name value range with 'x' column
    r.indices[-2:2:'x',...] -> & combinations
    """

    def __init__(self, table):
        self.table = table

    def __getitem__(self, rows):
        if not isinstance(rows, tuple):  # multiple arguments
            rows = [rows]
        view = _View(self.table, self.table._get_row_indices(rows[0]))
        for row in rows[1:]:
            view = _View(view, self.table._get_row_indices(row))
        return view.get_indices()


class Mask:
    def __init__(self, table):
        self.table = table

    def __getitem__(self, key):
        return self.table._get_name_mask(key, self.table._index)


class _RowView:
    def __init__(self, table):
        self.table = table
        self.mask = Mask(table)

    def __getitem__(self, rows):
        if not isinstance(rows, tuple):  # multiple arguments
            rows = [rows]
        return self.table._get_sub_table(rows, None)

    def __iter__(self):
        res_type = namedtuple("Row", self.table._col_names)
        for ii in range(len(self.table)):
            yield res_type(*[self.table[cc, ii] for cc in self.table._col_names])

    def at(self, index, as_dict=False):
        if as_dict:
            return {cc: self.table[cc, index] for cc in self.table._col_names}
        else:
            res_type = namedtuple("Row", self.table._col_names)
            return res_type(*[self.table[cc, index] for cc in self.table._col_names])

    def transpose(self):
        return self.table._t

    def __repr__(self):
        return f"RowView: {len(self.table)} rows, {len(self.table.cols)} cols"

    def __len__(self):
        return len(self.table)

    def head(self, n=5):
        return self[:n]

    def tail(self, n=5):
        return self[-n:]

    def reverse(self):
        return self[::-1]


class _ColView:
    def __init__(self, table):
        self.table = table

    def __getitem__(self, cols):
        return self.table._get_sub_table([], cols)

    @property
    def names(self):
        return self.table._col_names

    def __repr__(self):
        return "ColView: " + " ".join(self.table._col_names)

    def keys(self):
        return iter(self.table._col_names)

    def values(self):
        return (self.table._data[cc] for cc in self.table._col_names)

    def items(self):
        return ((cc, self.table._data[cc]) for cc in self.table._col_names)

    def __contains__(self, key):
        return key in self.table._col_names

    def __iter__(self):
        return iter(self.table._col_names)

    def __len__(self):
        return len(self.table._col_names)

    def transpose(self):
        return self.table._t


class _View:
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __getitem__(self, k):
        return self.data[k][self.index]

    def __len__(self):
        return len(self.data)

    def get(self, k, default=None):
        if k == "__tracebackhide__":  # to avoid issues in ipython
            return None
        return self.data.get(k, default)[self.index]

    def __repr__(self):
        return f"<{sum(self.index)} rows>"

    def __contains__(self, k):
        return k in self.data

    def eval(self, col):
        try:
            cc = eval(col, gblmath, self)
        except NameError:
            raise KeyError(
                f"Column `{col}` could not be found or " "is not a valid expression"
            )
        return cc

    def get_indices(self):
        if not hasattr(self.data, "_get_indices"):
            return np.arange(len(self.data))[self.index]
        else:
            return self.data._get_indices()[self.index]


class Table:
    """Table class managing list of columns and scalars values. Columns are numpy
    arrays with same lengths. A special column can be assigned as string index.

    The table can be accessed by columns, rows or by a combination of both. The table
    can be transposed and shown in a human readable format.

    Main extraction API:

    - `table[col]` : get a column (numpy array) or a scalar value
    - `table[col,row]` : get a value at a specific row (integer or string index)
    - `table.rows[<sel1>,<sel2>,...]` : get a view of the table by rows. <sel> can be:
        - a numerical index
        - a string regex pattern
        - a slice (start:stop:step) or a range (low:high:column)
        - a combination of above equivalent to table.rows[<sel1>].rows[<sel2>].rows[...]
    - `table.cols[<col1>,<col2>,...]`: get a view of the table by columns, where column can be also valid expression of column names.

    A string index can include a count and an offset. The count is the number of the
    match in the table and the offset is the row offset. The count and offset are
    separated by `sep_count` (default `::`) and `offset_sep` (default `<<` and `>>`)
    respectively.

    Other methods:

    - `table.show()` : show the table in a human readable format


    Internal implementation details:
    - `table._data` : dictionary of scalars and columns (np.arrays)
    - `table._col_names` : list of column names
    - `table._index` : column name used as index
    - `table._index_cache` : cache for index column
    - `table._count_cache` : cache for count in index column
    - `table._sep_count` : separator for count in index column
    - `table._sep_previous` : separator for previous index column
    - `table._sep_next` : separator for next index column

    Supported hidden methods
    - `get_sub_table...`

    """

    def __init__(
        self,
        data,
        col_names=None,
        index="name",
        sep_count="::",
        sep_previous="<<",
        sep_next=">>",
        cast_strings=True,
        regex_flags=re.IGNORECASE,
    ):
        _data = data.copy()
        _col_names = list(data.keys() if col_names is None else col_names)
        for kk in _col_names:
            vv = data[kk]
            if not hasattr(vv, "dtype"):
                raise ValueError(f"Column `{kk}` is not a numpy array")
            else:
                if cast_strings and vv.dtype.kind in "SU":
                    vv = np.array(vv, dtype=object)
            _data[kk] = vv

        nrows = set(len(_data[cc]) for cc in _col_names)
        if len(nrows) > 1:
            for cc in _col_names:
                print(f"Length {cc:r} = {len(_data[cc])}")
            raise ValueError("Columns have different lengths")

        if index is not None and index not in _col_names:
            raise ValueError(f"Index column `{index}` not found in columns")

        # special init due to setattr redefinition
        init = {
            #            "header": header,
            "mask": Mask(self),  ## to be deprecated
            "cols": _ColView(self),
            "rows": _RowView(self),
            "_data": data.copy(),
            "_col_names": _col_names,
            "_index": index,
            "_index_cache": None,
            "_count_cache": None,
            "_sep_count": sep_count,
            "_sep_previous": sep_previous,
            "_sep_next": sep_next,
            "_regex_flags": regex_flags,
        }

        for kk, vv in init.items():
            object.__setattr__(self, kk, vv)

    @property
    def _nrows(self):
        return len(self._data[self._col_names[0]])

    def _get_row_where_col(self, col, row, count=0, offset=0):
        # generally slower than _get_row_col_fast
        return np.where(col == row)[0][count] + offset

    def _make_cache(self):
        col = self._data[self._index]
        dct = {}
        count = {}
        for ii, nn in enumerate(col):
            cc = count.get(nn, -1) + 1
            dct[(nn, cc)] = ii
            count[nn] = cc
        for kk in count:
            count[kk] += 1
        return dct, count

    def _get_cache(self):
        if self._index_cache is None:
            _index_cache, _count_cache = self._make_cache()
            object.__setattr__(self, "_index_cache", _index_cache)
            object.__setattr__(self, "_count_cache", _count_cache)
        return self._index_cache, self._count_cache

    def _get_row_cache(self, row, count, offset):
        """Get the index of a row by name and repetition."""
        cache, count_dict = self._get_cache()
        if count is None:
            count = 0
        if count < 0:
            count += count_dict[row]
        idx = cache.get((row, count))
        return idx + offset if idx is not None else None

    def _get_row_cache_raise(self, row, count, offset):
        idx = self._get_row_cache(row, count, offset)
        if idx is None:
            raise KeyError(f"Cannot find '{row}' in column '{self._index}'")
        return idx

    def _split_name_count_offset(self, name):
        """Split a name::count<<offset into name, count and offset from selector."""
        ss = name.split(self._sep_previous)
        name = ss[0]
        offset = 0 if len(ss) == 1 else -int(ss[1])
        ss = name.split(self._sep_next)
        name = ss[0]
        offset += 0 if len(ss) == 1 else int(ss[1])
        ss = name.split(self._sep_count)
        name = ss[0]
        count = None if len(ss) == 1 else int(ss[1])
        return name, count, offset

    def _get_regexp_indices(self, regexp, col):
        """Get indices using string selector on the index column.

        Examples:
           "mq.*::1<<1" -> all elements preceding second matches of mq.*
        """
        name, count, offset = self._split_name_count_offset(regexp)
        regexpr = re.compile(name, flags=self._regex_flags)
        lst = []
        cnt = -1
        for ii, nn in enumerate(self._data[col]):
            if regexpr.fullmatch(nn):
                cnt += 1
                if count is None or count == cnt:
                    lst.append(ii)
        return np.array(lst, dtype=int) + offset

    def _get_row_index(self, row):
        """
            get the row index from:
        - integer
        - name::count<<offset, name::count>>offset
        - (name, count, offset) or (name, count)
        """
        if isinstance(row, int):
            return row
        elif isinstance(row, str):
            row, count, offset = self._split_name_count_offset(row)
            return self._get_row_cache_raise(row, count, offset)
        elif isinstance(row, tuple):
            return self._get_row_cache(*row)
        else:
            raise ValueError(f"Invalid row {row}")

    def _get_row_indices(self, row):
        """
            get the row indices from:
        - integer
        - regexp::count<<offset, name::count>>offset
        - (name, count, offset) or (name, count)
        - slice
        - list of the above
        """
        if isinstance(row, slice):
            ia = row.start
            ib = row.stop
            ic = row.step
            if isinstance(ia, str) or isinstance(ib, str):  # name matching
                if ic is None:
                    if ia is not None:
                        ia = self._get_row_index(ia)
                    if ib is not None:
                        ib = self._get_row_index(ib) + 1
                else:
                    if ia is not None:
                        ia = self._get_row_where_col(self._data[ic], ia)
                    if ib is not None:
                        ib = self._get_row_where_col(self._data[ic], ib) + 1
                return slice(ia, ib)
            elif isinstance(ic, str):  # range matching
                col = self._data[ic]
                if ia is None and ib is None:
                    return slice(None)
                elif ia is not None and ib is None:
                    return np.where(col <= ib)[0]
                elif ib is not None and ia is None:
                    return np.where(col >= ia)[0]
                else:
                    return np.where((col >= ia) & (col <= ib))[0]
            else:  # plain slice
                return row
        elif is_iterable(row):  # could be a mask
            if hasattr(row, "dtype") and row.dtype is np.dtype("bool"):
                return np.where(row)[0]
            else:
                out = []
                for rr in row:
                    if isinstance(rr, str):
                        out.append(self._get_row_index(rr))
                    elif isinstance(rr, int):
                        out.append(rr)
                    else:
                        raise ValueError(f"Invalid row {rr}")
                return np.array(out)
        elif isinstance(row, str):
            return self._get_regexp_indices(row, self._index)
        else:
            return [self._get_row_index(row)]

    def _get_name_mask(self, name, col):
        """Get mask using string selector on a column."""
        mask = np.zeros(self._nrows, dtype=bool)
        indices = self._get_row_indices(name)
        mask[indices] = True
        return mask

    def _get_sub_table(self, rows, cols):
        view = self._data
        # create row view
        for row in rows:
            if row is not None:
                view = _View(view, self._get_row_indices(row))
        # select columms
        if cols is None or cols == slice(None, None, None):
            col_list = self._col_names
        elif isinstance(cols, str):
            col_list = cols.split()
        elif is_iterable(cols):
            col_list = list(cols)
        else:
            raise ValueError(f"Invalid column: {cols}")

        if self._index not in col_list:
            col_list.insert(0, self._index)
        data = {}
        for cc in col_list:
            data[cc] = eval(cc, gblmath, view)
        for kk in self.keys(exclude_columns=True):
            data[kk] = self._data[kk]
        return self.__class__(
            data,
            col_names=col_list,
            index=self._index,
            sep_count=self._sep_count,
            sep_previous=self._sep_previous,
            sep_next=self._sep_next,
            regex_flags=self._regex_flags,
        )

    def _get_sub_table_from_indices(self, indices):
        data = {}
        for cc in self._col_names:
            data[cc] = self._data[cc][indices]
        return self.__class__(
            data,
            col_names=self._col_names,
            index=self._index,
            sep_count=self._sep_count,
            sep_previous=self._sep_previous,
            sep_next=self._sep_next,
            regex_flags=self._regex_flags,
        )

    def __getitem__(self, args):
        """Extract data from the table.

        If one argument is given, it can be a column name or a valid expression
        of column names. If multiple arguments are given, the first argument is
        the column name or expression and the following arguments are row
        selectors.
        """
        if isinstance(args, str):
            try:
                return self._data[args]
            except KeyError:
                return eval(args, gblmath, self._data)
        if type(args) is tuple:  # multiple args
            if len(args) == 0:
                col = None
                row = None
            elif len(args) == 1:
                col = args[0]
                row = None
            elif len(args) == 2:
                col = args[0]
                row = args[1]
                try:
                    col = self._data[col]
                except KeyError:
                    col = eval(col, gblmath, self._data)
                if isinstance(row, str):
                    cache, count = self._get_cache()
                    idx = cache.get((row, 0))
                    if idx is None:
                        name, count, offset = self._split_name_count_offset(row)
                        idx = self._get_row_cache(name, count, offset)
                elif isinstance(row, tuple):
                    cache, count = self._get_cache()
                    idx = cache.get(row)
                    if idx is None:
                        idx = self._get_row_cache(*row)
                elif isinstance(row, slice):
                    idx = self._get_row_indices(row)
                elif isinstance(row, list):
                    idx = self._get_row_indices(row)
                else:
                    idx = row
                return col[idx]
            else:
                raise ValueError(f"Too many arguments {args} for table {id(self)}.")
        raise ValueError(f"Invalid arguments {args} for table {id(self)}.")

    def show(
        self,
        rows=None,
        cols=None,
        maxrows=None,
        maxwidth=None,
        output=None,
        digits=6,
        fixed="g",
        header=True,
        max_col_width=None,
    ):
        if rows is None and cols is None:
            view = self
        else:
            view = self._get_sub_table(rows, cols)

        col_list = view._col_names

        # index always first
        if self._index in col_list:
            col_list.remove(self._index)
        col_list.insert(0, self._index)

        cut = -1
        viewrows = len(view)
        if maxrows is not None and output is None and viewrows > maxrows:
            cut = maxrows // 2

        if maxwidth == "auto":
            try:
                maxwidth = os.get_terminal_size().columns - 5
                if maxwidth < 10 or maxwidth > 10000:
                    raise ValueError("Terminal width too big or too small.")
            except (OSError, ValueError):
                maxwidth = 100
        elif maxwidth == "full" or maxwidth is None:
            maxwidth = 100000000

        data = []
        width = 0
        fmt = []
        header_line = []
        for cc in col_list:
            if cc in view:
                coldata = view[cc]
            else:
                coldata = eval(cc, gblmath, view)
            if cut > 0:
                coldata = np.r_[coldata[:cut], coldata[cut:]]
            coltype = coldata.dtype.kind
            col = _to_str(coldata, digits, fixed, max_len=max_col_width)
            colwidth = int(col.dtype.str[2:])
            if len(cc) > colwidth:
                colwidth = len(cc)
            width += colwidth + 1
            if width < maxwidth:
                if coltype in "SUO":
                    fmt.append("%%-%ds" % (colwidth))
                else:
                    fmt.append("%%%ds" % colwidth)
                header_line.append(fmt[-1] % str(cc))
                data.append(col)
            else:
                header_line.append("...")
                break

        result = []
        if header:
            result.append(" ".join(header_line))
        for ii in range(len(col)):
            row = " ".join([ff % col[ii] for ff, col in zip(fmt, data)])
            result.append(row)
            if ii == cut:
                result.append("...")
        result = "\n".join(result)
        if output is None:
            print(result)
        elif output is str:
            return result
        elif hasattr(output, "write"):
            output.write(result)
        else:
            output = pathlib.Path(output)
            with open(output, "w") as fh:
                fh.write(result)

    @classmethod
    def from_pandas(cls, df, index=None, lowercase=False):
        if index is None:
            index = df.index.name
            if index is None and "NAME" in df.columns:
                index = "NAME"
        if lowercase:
            df.columns = df.columns.str.lower()
            index = index.lower()
            for cc in df.columns:
                if df[cc].dtype.kind in "SUO":
                    df[cc] = df[cc].str.lower()
        col_names = list(df.columns)
        data = {cc: df[cc].values for cc in col_names}
        if hasattr(df, "headers"):
            for cc, dd in df.headers.items():
                cc = cc.lower() if lowercase else cc
                if cc in data:
                    cc = cc + "_header"
                if lowercase and isinstance(dd, str):
                    dd = dd.lower()
                data[cc] = dd
        return cls(data, col_names=col_names, index=index)

    @classmethod
    def from_csv(cls, filename, index=None, col_names=None, **kwargs):
        import pandas as pd

        df = pd.read_csv(filename, **kwargs)
        if index is None and "NAME" in df.columns:
            index = "NAME"

        return cls.from_pandas(df, index=index, col_names=col_names)

    @classmethod
    def from_rows(cls, rows, col_names=None, index=None):
        if hasattr(rows[0], "_asdict"):  # namedtuple
            if col_names is None:
                col_names = list(rows[0]._fields)
            data = {cc: np.array([getattr(rr, cc) for rr in rows]) for cc in col_names}
        else:
            if col_names is None:
                col_names = list(rows[0].keys())
            data = {cc: np.array([rr[cc] for rr in rows]) for cc in col_names}
        return cls(data, col_names=col_names, index=index)

    @classmethod
    def from_tfs(cls, filename, index=None, lowercase=True):
        from tfs import read_tfs

        df = read_tfs(filename)
        return cls.from_pandas(df, index=index, lowercase=lowercase)

    @property
    def _t(self):
        """Transpose the table."""
        data = {"columns": np.array(self._col_names)}
        for nn in range(len(self)):
            data[f"row{nn}"] = np.array([str(self[cc][nn]) for cc in self._col_names])
        return Table(data, index="columns", col_names=list(data.keys()))

    @property
    def _df(self):
        return self.to_pandas()

    def __contains__(self, key):
        return self._data.__contains__(key)

    def __delitem__(self, key, val):
        if key in self._col_names:
            self._col_names.remove(key)
        del self._data[key]

    def __dir__(self):
        return super().__dir__() + list(self._data.keys())

    def __iter__(self):
        return self._data.__iter__()

    def __getattr__(self, key):
        if key in self._data:
            return self._data[key]
        else:
            raise AttributeError(f"Cannot find `{key}` in table")

    def __len__(self):
        k = self._col_names[0]
        return len(self._data[k])

    def __setitem__(self, key, val):
        if key == self._index:
            object.__setattr__(self, "_index_cache", None)
            object.__setattr__(self, "_count_cache", None)
        if key in self.__dict__:
            object.__setattr__(self, key, val)
        elif key in self._col_names:
            self._data[key][:] = val
        else:
            self._data[key] = val
            if hasattr(val, "__iter__") and len(val) == self._nrows:
                self._col_names.append(key)

    __setattr__ = __setitem__

    def __repr__(self):
        n = self._nrows
        c = len(self._col_names)
        ns = "s" if n != 1 else ""
        cs = "s" if c != 1 else ""
        out = [f"{self.__class__.__name__}: {n} row{ns}, {c} col{cs}"]
        if self._nrows < 30:
            out.append(self.show(output=str, maxwidth="auto"))
        else:
            out.append(self.rows[:10].show(output=str, maxwidth="auto"))
            out.append("...")
            out.append(self.rows[-10:].show(header=False, output=str, maxwidth="auto"))
            # out.append("Use `table.show()` to see the full table.")
        return "\n".join(out)

    def __neg__(self):
        return self.rows.reverse()

    def _concatenate_table(self, table):
        """Concatenate a table to the table."""
        for col in table._col_names:
            self._data[col] = np.concatenate([self._data[col], table._data[col]])
        return self

    def __add__(self, other):
        if isinstance(other, Table):
            res = self._copy()
            return res._concatenate_table(other)
        else:
            raise ValueError("Can only concatenate Table with Table")

    def __mul__(self, num):
        res = self._copy()
        for col in res._col_names:
            res._data[col] = np.concatenate([res._data[col]] * num)
        return res

    def _copy(self):
        return self.__class__(
            self._data.copy(),
            col_names=self._col_names,
            index=self._index,
            sep_count=self._sep_count,
            sep_previous=self._sep_previous,
            sep_next=self._sep_next,
        )

    def items(self):
        return self._data.items()

    def keys(self, exclude_columns=False):
        if exclude_columns:
            return [kk for kk in self._data.keys() if kk not in self._col_names]
        return self._data.keys()

    def values(self):
        return self._data.values()

    def pop(self, key):
        res = self._data.pop(key)
        if key in self._col_names:
            self._col_names.remove(key)
        return res

    def _update(self, data):
        """Update the table with new data."""
        if hasattr(self, "_data"):
            data = self._data
        for name, value in data.items():
            if hasattr(value, "__len__") and len(value) == len(self):
                if name not in self._data:
                    self._col_names.append(name)
            self._data[name] = value

    def _append_row(self, row):
        """Append a row to the table."""
        for col in self._col_names:
            self._data[col] = np.r_[self._data[col], [row[col]]]

    def to_pandas(self, index=None, columns=None):
        if columns is None:
            columns = self._col_names

        import pandas as pd

        df = pd.DataFrame(self._data, columns=self._col_names)
        if index is not None:
            df.set_index(index, inplace=True)
        return df
