#!/usr/bin/env python3

from pathlib import Path
import re
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"
ROCM_TARGETS = (
    "gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;"
    "gfx1150;gfx1151;gfx1200;gfx1201"
)
LINUX_MATRIX = [
    ("cpu", "x64", "ubuntu-22.04"),
    ("cpu", "arm64", "ubuntu-22.04-arm"),
    ("vulkan", "x64", "ubuntu-22.04"),
    ("vulkan", "arm64", "ubuntu-24.04-arm"),
    ("cuda", "x64", "ubuntu-24.04"),
    ("rocm", "x64", "ubuntu-24.04"),
]


class RocmReleaseWorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = WORKFLOW.read_text(encoding="utf-8")
        linux_job = re.search(
            r"(?ms)^  build-linux:\n(?P<body>.*?)(?=^  build-macos:)", cls.workflow
        )
        if linux_job is None:
            raise AssertionError("build-linux job is missing")
        cls.linux_job = linux_job.group("body")

        matrix = re.search(
            r"(?ms)^        include:\n(?P<body>.*?)(?=^    steps:)", cls.linux_job
        )
        if matrix is None:
            raise AssertionError("build-linux matrix include block is missing")
        cls.matrix = matrix.group("body")

    def step(self, name: str) -> str:
        match = re.search(
            rf"(?ms)^      - name: {re.escape(name)}\n(?P<body>.*?)(?=^      - name:|\Z)",
            self.linux_job,
        )
        self.assertIsNotNone(match, f"Linux workflow step is missing: {name}")
        return match.group("body")

    def rocm_branch(self, step: str) -> str:
        match = re.search(
            r'''(?ms)^          if \[ "\$\{\{ matrix\.backend \}\}" = "rocm" \]; then\n'''
            r"(?P<body>.*?)^          fi$",
            step,
        )
        self.assertIsNotNone(match, "ROCm branch is missing")
        return match.group("body")

    def assert_rocm_elf_loop(self, package: str, primary_elfs: str) -> None:
        branch = self.rocm_branch(package)
        self.assertIn('dynamic=$(readelf -d "$elf")', branch)
        self.assertIn(
            "paths=$(sed -nE 's/.*\\((RPATH|RUNPATH)\\).*\\[(.*)\\]/\\2/p'",
            branch,
        )
        self.assertIn("if [ \"$paths\" != '$ORIGIN' ]; then", branch)
        self.assertIn("dependencies=$(ldd \"$elf\")", branch)
        self.assertIn("if grep -q 'not found' <<<\"$dependencies\"; then", branch)
        self.assertIn(
            "mapfile -t ggml_libraries < <(find \"$BUNDLE\" -type f "
            "-name 'libggml*.so*' -print)",
            branch,
        )

        loop = re.search(
            rf'''(?ms)^            for elf in {re.escape(primary_elfs)} "\$\{{ggml_libraries\[@\]\}}"; do\n'''
            r"(?P<body>.*?)^            done$",
            branch,
        )
        self.assertIsNotNone(loop, "staged ELF validation loop is missing")
        loop_body = loop.group("body")
        self.assertEqual(loop_body.count('check_origin_rpath "$elf"'), 1)
        self.assertEqual(loop_body.count('check_linkage "$elf"'), 1)

    def test_linux_matrix_preserves_every_backend_and_adds_rocm(self) -> None:
        entries = []
        for match in re.finditer(
            r"(?ms)^          - backend: (?P<backend>\w+)\n(?P<body>.*?)(?=^          - backend:|\Z)",
            self.matrix,
        ):
            body = match.group("body")
            arch = re.search(r"^            arch: (\w+)$", body, re.MULTILINE)
            runner = re.search(r"^            runner: ([\w.-]+)$", body, re.MULTILINE)
            self.assertIsNotNone(arch)
            self.assertIsNotNone(runner)
            entries.append((match.group("backend"), arch.group(1), runner.group(1)))
        self.assertEqual(entries, LINUX_MATRIX)

    def test_rocm_uses_official_724_repository_and_required_packages(self) -> None:
        install = self.step("Install ROCm 7.2.4")
        self.assertIn("if: matrix.backend == 'rocm'", install)
        self.assertIn("https://repo.radeon.com/rocm/rocm.gpg.key", install)
        self.assertIn("https://repo.radeon.com/rocm/apt/7.2.4 noble main", install)
        packages = re.search(r"sudo apt-get install -y (?P<packages>[^\n]+)", install)
        self.assertIsNotNone(packages)
        self.assertEqual(
            packages.group("packages").split(),
            ["hipcc", "hip-dev", "rocm-device-libs", "hipblas-dev", "rocblas-dev"],
        )

    def test_rocm_targets_and_cmake_flags_are_exact(self) -> None:
        entry = re.search(
            r"(?ms)^          - backend: rocm\n(?P<body>.*?)(?=^          - backend:|\Z)",
            self.matrix,
        )
        self.assertIsNotNone(entry)
        body = entry.group("body")
        cmake_args = re.search(
            r'^            cmake_args: "([^"]+)"$', body, re.MULTILINE
        )
        self.assertIsNotNone(cmake_args)
        self.assertEqual(
            cmake_args.group(1),
            "-DPARAKEET_GGML_HIP=ON -DGGML_HIP_NO_VMM=ON "
            "-DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ "
            "-DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm-7.2.4",
        )
        targets = re.search(r'^            gpu_targets: "([^"]+)"$', body, re.MULTILINE)
        self.assertIsNotNone(targets)
        self.assertEqual(targets.group(1), ROCM_TARGETS)

        for name in ("Configure", "Configure (shared C-API lib)"):
            configure = self.step(name)
            rocm = self.rocm_branch(configure)
            self.assertIn('-DBUILD_SHARED_LIBS=ON', rocm)
            self.assertIn('-DCMAKE_INSTALL_RPATH=\\$ORIGIN', rocm)
            self.assertIn('-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON', rocm)
            self.assertIn('${GPU_TARGETS:+"-DGPU_TARGETS=${GPU_TARGETS}"}', configure)
            self.assertIn('if [ "${{ matrix.backend }}" = "cuda" ]; then', configure)
            self.assertIn('-DCMAKE_BUILD_RPATH=\\$ORIGIN', configure)

    def test_binary_archive_stages_complete_payload_and_checks_every_elf(self) -> None:
        package = self.step("Package")
        branch = self.rocm_branch(package)
        self.assertIn(
            'scripts/package_rocm_libraries.sh build/third_party/ggml/src "$BUNDLE"',
            branch,
        )
        for path in (
            "parakeet-cli",
            "parakeet-server",
            "LICENSE",
            "README.md",
            "libggml.so",
            "libggml-base.so",
            "libggml-cpu.so",
            "libggml-hip.so",
        ):
            self.assertRegex(branch, rf'test -[xef] "\$BUNDLE/{re.escape(path)}"')
        self.assert_rocm_elf_loop(
            package, '"$BUNDLE/parakeet-cli" "$BUNDLE/parakeet-server"'
        )
        self.assertIn('out=$("$BUNDLE/parakeet-cli" 2>&1 || true)', branch)
        self.assertIn('out=$("$BUNDLE/parakeet-server" --help 2>&1 || true)', branch)
        self.assertIn(
            'BUNDLE="parakeet-${{ steps.ver.outputs.version }}-bin-linux-'
            '${{ matrix.backend }}-${{ matrix.arch }}"',
            package,
        )
        self.assertIn('tar -czf "$BUNDLE.tar.gz" "$BUNDLE"', package)

    def test_capi_archive_stages_complete_payload_and_checks_every_elf(self) -> None:
        package = self.step("Package (shared C-API lib)")
        branch = self.rocm_branch(package)
        self.assertIn(
            'scripts/package_rocm_libraries.sh '
            'build-shared/third_party/ggml/src "$BUNDLE"',
            branch,
        )
        for path in (
            "libparakeet.so",
            "parakeet_capi.h",
            "LICENSE",
            "README.md",
            "libggml.so",
            "libggml-base.so",
            "libggml-cpu.so",
            "libggml-hip.so",
        ):
            self.assertRegex(branch, rf'test -[xef] "\$BUNDLE/{re.escape(path)}"')
        self.assert_rocm_elf_loop(package, '"$BUNDLE/libparakeet.so"')
        self.assertIn(
            'BUNDLE="parakeet-${{ steps.ver.outputs.version }}-lib-linux-'
            '${{ matrix.backend }}-${{ matrix.arch }}"',
            package,
        )
        self.assertIn('tar -czf "$BUNDLE.tar.gz" "$BUNDLE"', package)

    def test_both_archives_use_existing_artifact_and_release_upload_flow(self) -> None:
        upload_binary = self.step("Upload artifact")
        upload_library = self.step("Upload shared C-API lib artifact")
        self.assertIn("name: ${{ steps.pack.outputs.bundle }}", upload_binary)
        self.assertIn("path: ${{ steps.pack.outputs.bundle }}.tar.gz", upload_binary)
        self.assertIn("name: ${{ steps.pack_lib.outputs.bundle }}", upload_library)
        self.assertIn("path: ${{ steps.pack_lib.outputs.bundle }}.tar.gz", upload_library)
        self.assertIn('gh release upload "$TAG"', self.workflow)
        self.assertIn("dist/*", self.workflow)


if __name__ == "__main__":
    unittest.main()
