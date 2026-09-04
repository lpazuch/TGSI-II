from __future__ import annotations

import unittest

from tgsi_pipeline.utils import parse_float


class ParseFloatTest(unittest.TestCase):
    def test_parse_decimal_with_dot(self) -> None:
        self.assertEqual(parse_float("80.52"), 80.52)

    def test_parse_decimal_with_comma(self) -> None:
        self.assertEqual(parse_float("80,52"), 80.52)

    def test_parse_thousand_separator(self) -> None:
        self.assertEqual(parse_float("1.234,56"), 1234.56)


if __name__ == "__main__":
    unittest.main()
