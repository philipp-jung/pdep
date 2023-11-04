import pandas as pd
from typing import Tuple, Dict
from itertools import combinations
from collections import namedtuple

PdepTuple = namedtuple("PdepTuple", ["pdep", "gpdep"])


def calculate_frequency(df: pd.DataFrame, col: int):
    """
    Calculates the frequency of a value to occur in colum
    col on dataframe df.
    """
    counts = df.iloc[:, col].value_counts()
    return counts.to_dict()


def calculate_counts_dict(df: pd.DataFrame, order=1) -> dict:
    """
    Calculates a dictionary d that contains the absolute counts of how
    often values in the lhs occur with values in the rhs in the table df.

    The dictionary has the structure
    d[lhs_columns][rhs_column][lhs_values][rhs_value],
    where lhs_columns is a tuple of one or more lhs_columns, and
    rhs_column is the columns whose values are determined by lhs_columns.

    Pass an `order` argument to indicate how many columns in the lhs should
    be investigated. If order=1, only unary relationships are taken into account,
    if order=2, only binary relationships are taken ito account, and so on.
    """
    i_cols = list(range(df.shape[1]))
    d = {comb: {cc: {} for cc in i_cols} for comb in combinations(i_cols, order)}
    for lhs_cols in d:
        d[lhs_cols]["value_counts"] = {}

    for row in df.itertuples(index=True):
        i_row, row = row[0], row[1:]
        for lhs_cols in combinations(i_cols, order):
            lhs_vals = tuple(row[lhs_col] for lhs_col in lhs_cols)

            # increase counts of values in the LHS, accessed via d[lhs_columns]['value_counts']
            if d[lhs_cols]["value_counts"].get(lhs_vals) is None:
                d[lhs_cols]["value_counts"][lhs_vals] = 1.0
            else:
                d[lhs_cols]["value_counts"][lhs_vals] += 1.0

            # update conditional counts
            for rhs_col in i_cols:
                if rhs_col not in lhs_cols:
                    if d[lhs_cols][rhs_col].get(lhs_vals) is None:
                        d[lhs_cols][rhs_col][lhs_vals] = {}
                    rhs_val = row[rhs_col]
                    if d[lhs_cols][rhs_col][lhs_vals].get(rhs_val) is None:
                        d[lhs_cols][rhs_col][lhs_vals][rhs_val] = 1.0
                    else:
                        d[lhs_cols][rhs_col][lhs_vals][rhs_val] += 1.0
    return d


def expected_pdep(
    n_rows: int,
    counts_dict: dict,
    order: int,
    A: Tuple[int, ...],
    B: int,
) -> float | None:
    """
    Calculates the expected value of pdep(A,B).
    """
    pdep_B = pdep(n_rows, counts_dict, order, tuple([B]))

    if pdep_B is None:
        return None

    if len(A) == 1:
        n_distinct_values_A = len(counts_dict[1][A]["value_counts"])
    elif len(A) > 1:
        n_distinct_values_A = len(counts_dict[order][A][B])
    else:
        raise ValueError("A needs to contain one or more attribute names")

    return pdep_B + (n_distinct_values_A - 1) / (n_rows - 1) * (1 - pdep_B)


def pdep(
    n_rows: int,
    counts_dict: dict,
    order: int,
    A: Tuple[int, ...],
    B: int | None = None,
) -> float | None:
    """
    Calculates the probabilistic dependence (pdep) between a left hand side A,
    which consists of one or more attributes, and an optional right hand side B,
    which consists of one attribute.

    If B is None, calculate pdep(A), that is the probability that two randomly
    selected records from A will have the same value.
    """
    sum_components = []

    if B is not None:  # pdep(A,B)
        counts_dict = counts_dict[order][A][B]
        for lhs_val, rhs_dict in counts_dict.items():  # lhs_val same as A_i
            lhs_counts = sum(rhs_dict.values())  # same as a_i
            for rhs_val, rhs_counts in rhs_dict.items():  # rhs_counts same as n_ij
                sum_components.append(rhs_counts**2 / lhs_counts)
        return sum(sum_components) / n_rows

    elif len(A) == 1:  # pdep(A)
        counts_dict = counts_dict[1]
        counts_dict = counts_dict[A]["value_counts"]
        for lhs_val, lhs_rel_frequency in counts_dict.items():
            sum_components.append(lhs_rel_frequency**2)
        return sum(sum_components) / n_rows**2

    else:
        raise ValueError(
            "Wrong data type for A or B, or wrong order. A "
            "should be a tuple of a list of column names of df, "
            "B should be name of a column or None. If B is None, "
            "order must be 1."
        )


def gpdep(
    n_rows: int,
    counts_dict: dict,
    A: Tuple[int, ...],
    B: int,
    order: int,
) -> PdepTuple | None:
    """
    Calculates the *genuine* probabilistic dependence (gpdep) between
    a left hand side A, which consists of one or more attributes, and
    a right hand side B, which consists of exactly one attribute.
    """
    pdep_A_B = pdep(n_rows, counts_dict, order, A, B)
    epdep_A_B = expected_pdep(n_rows, counts_dict, order, A, B)

    if pdep_A_B is not None and epdep_A_B is not None:
        gpdep_A_B = pdep_A_B - epdep_A_B
        return PdepTuple(pdep_A_B, gpdep_A_B)
    return None


def calc_all_gpdeps(
    counts_dict: dict, df: pd.DataFrame, detected_cells: Dict[Tuple, str], order: int
) -> Dict[Tuple, Dict[int, PdepTuple]]:
    """
    Calculate all gpdeps in dataframe df, with an order implied by the depth
    of counts_dict.
    """
    n_rows, n_cols = df.shape
    lhss = set([x for x in counts_dict[order].keys()])
    rhss = list(range(n_cols))
    gpdeps = {lhs: {} for lhs in lhss}
    for lhs in lhss:
        for rhs in rhss:
            gpdeps[lhs][rhs] = gpdep(
                n_rows, counts_dict, detected_cells, lhs, rhs, order
            )
    return gpdeps
