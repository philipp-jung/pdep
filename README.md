# pdep: Calculate (g)pdep Measure On Dataframe

Developed by Gregory Piatetsky-Shapiro and Christopher J. Matheus[^0] and published in 1993, the pdep-measure
is a way to determine the relative significance of different attributes in predicting the target attribute.
is the probability that two randomly selected rows will have the same value for column A, given that they have the
same value for the target attribute.
The pdep-measure between two columns, pdep(A,B), however, is the probability that two randomly selected rows will
have the same values for columns A and B, given that they have the same value for the target attribute.

## How to install it
Install [poetry](https://python-poetry.org/docs/#installing-with-the-official-installer) on your system.
Then, clone the repo, navigate into it an run `poetry install`.
Proceed to open activate a virtual environment with `poetry shell`, then run `python test_pdep.py`.

## How to use it
Have a look in the tests to see how to use the package.
Note that `counts_dict` is a nested dictionary that needs to be constructed before calculating
pdep values.
The parameter `order` references the number of attributes in the left-hand-side.
For example, an FD `A,B -> C` is of `order=2`, while `A -> C` is of `order=1`.

## How to test it
To test, run `python test_pdep.py`.

[^0]: https://www.researchgate.net/publication/230818124_Measuring_data_dependencies_in_large_databases
