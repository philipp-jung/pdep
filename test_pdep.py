import unittest
import pandas as pd
import pdep


class TestPdep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5, 6, 7],
                "name": ["Natalie", "Alice", "Tim", "Bob", "Bob", "Alice", "Bob"],
                "zip": [14193, 14193, 14880, 14882, 14882, 14880, 14193],
                "city": [
                    "Berlin",
                    "Berlin",
                    "Potsdam",
                    "Potsdam",
                    "Potsdam",
                    "Potsdam",
                    "Berln",
                ],
            }
        )
        cls.counts_dict = {1: pdep.calculate_counts_dict(cls.df, order=1)}
        cls.n_rows = cls.df.shape[0]

        # make references to columns more lisible
        cls.zip_code = tuple([2])
        cls.city = 3

    def test_pdep_a(self):
        """
        Test that pdep(city) is computed correctly.
        """
        pdep_city = round(
            pdep.pdep(self.n_rows, self.counts_dict, 1, tuple([self.city])), 2
        )
        self.assertEqual(pdep_city, 0.43)

    def test_pdep_a_b(self):
        """
        Test that pdep(zip,city) is computed correctly.
        """
        pdep_zip_city = round(
            pdep.pdep(self.n_rows, self.counts_dict, 1, self.zip_code, self.city), 2
        )
        self.assertEqual(pdep_zip_city, 0.81)

    def test_expected_pdep_a_b(self):
        """
        Test that E[pdep(zip,city)] is computed correctly.
        """
        epdep_zip_city = round(
            pdep.expected_pdep(
                self.n_rows, self.counts_dict, 1, self.zip_code, self.city
            ),
            2,
        )
        self.assertEqual(epdep_zip_city, 0.62)


if __name__ == "__main__":
    unittest.main()
