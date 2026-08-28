#!/usr/bin/env python3

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "package_rocm_libraries.sh"
FAMILIES = (
    "libggml.so",
    "libggml-base.so",
    "libggml-cpu.so",
    "libggml-hip.so",
)


class PackageRocmLibrariesTest(unittest.TestCase):
    def run_script(self, *args: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(SCRIPT), *(str(arg) for arg in args)],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_stages_all_families_and_preserves_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source"
            destination = root / "destination"
            nested = source / "nested" / "lib"
            nested.mkdir(parents=True)
            destination.mkdir()

            (nested / "libggml.so.0").write_bytes(b"ggml")
            (nested / "libggml.so").symlink_to("libggml.so.0")
            (source / "libggml-base.so.0").write_bytes(b"base")
            (nested / "libggml-cpu.so").write_bytes(b"cpu")
            (nested / "libggml-hip.so.0").write_bytes(b"hip")
            (nested / "libggml-hip.so").symlink_to("libggml-hip.so.0")

            (nested / "libggml-cuda.so").write_bytes(b"cuda")
            (nested / "libamdhip64.so.6").write_bytes(b"rocm userspace")
            (nested / "README.txt").write_text("unrelated", encoding="utf-8")

            result = self.run_script(source, destination)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                {path.name for path in destination.iterdir()},
                {
                    "libggml.so",
                    "libggml.so.0",
                    "libggml-base.so.0",
                    "libggml-cpu.so",
                    "libggml-hip.so",
                    "libggml-hip.so.0",
                },
            )
            self.assertTrue((destination / "libggml.so").is_symlink())
            self.assertEqual(os.readlink(destination / "libggml.so"), "libggml.so.0")
            self.assertTrue((destination / "libggml-hip.so").is_symlink())
            self.assertEqual(
                os.readlink(destination / "libggml-hip.so"), "libggml-hip.so.0"
            )

    def test_rejects_each_missing_family_with_diagnostic(self) -> None:
        for missing_family in FAMILIES:
            with self.subTest(missing_family=missing_family):
                with tempfile.TemporaryDirectory() as temp_dir:
                    root = Path(temp_dir)
                    source = root / "source"
                    destination = root / "destination"
                    source.mkdir()
                    destination.mkdir()

                    for family in FAMILIES:
                        if family != missing_family:
                            (source / family).write_bytes(family.encode())

                    result = self.run_script(source, destination)

                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(missing_family, result.stderr)
                    self.assertEqual(list(destination.iterdir()), [])

    def test_rejects_invalid_arguments_and_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            existing = root / "existing"
            missing = root / "missing"
            existing.mkdir()

            wrong_count = self.run_script(existing)
            missing_source = self.run_script(missing, existing)
            missing_destination = self.run_script(existing, missing)

            self.assertNotEqual(wrong_count.returncode, 0)
            self.assertIn("SOURCE_ROOT DESTINATION", wrong_count.stderr)
            self.assertNotEqual(missing_source.returncode, 0)
            self.assertIn(str(missing), missing_source.stderr)
            self.assertNotEqual(missing_destination.returncode, 0)
            self.assertIn(str(missing), missing_destination.stderr)


if __name__ == "__main__":
    unittest.main()
