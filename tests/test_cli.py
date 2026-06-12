from __future__ import annotations

import unittest
from unittest import mock

from donate import cli


class CliTests(unittest.TestCase):
    def test_packaged_command_defaults_to_web_wizard(self) -> None:
        with mock.patch("donate.cli.web_mod.main", return_value=0) as run:
            self.assertEqual(cli.main([]), 0)
        run.assert_called_once_with([])

    def test_terminal_mode_keeps_terminal_wizard(self) -> None:
        with mock.patch("donate.cli.donate_main", return_value=0) as run:
            self.assertEqual(cli.main(["--terminal", "--all"]), 0)
        run.assert_called_once_with(["--all"])

    def test_web_flags_route_to_web_wizard(self) -> None:
        with mock.patch("donate.cli.web_mod.main", return_value=0) as run:
            self.assertEqual(cli.main(["--no-open", "--web-port", "9000"]), 0)
        run.assert_called_once_with(["--no-open", "--port", "9000"])


if __name__ == "__main__":
    unittest.main()
