#!/usr/bin/env python3

from pathlib import Path
import re
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
WORKFLOW = (REPO_ROOT / ".github" / "workflows" / "docker.yml").read_text(
    encoding="utf-8"
)
ROCM_TARGETS = (
    "gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;"
    "gfx1150;gfx1151;gfx1200;gfx1201"
)


def docker_stage(name: str) -> str:
    match = re.search(
        rf"(?ms)^FROM [^\n]+ AS {re.escape(name)}\n(?P<body>.*?)(?=^FROM |\Z)",
        DOCKERFILE,
    )
    if match is None:
        raise AssertionError(f"Docker stage is missing: {name}")
    return match.group("body")


class RocmDockerfileTest(unittest.TestCase):
    def test_build_stage_installs_official_rocm_724_toolchain_conditionally(self) -> None:
        build = docker_stage("build")
        commands = build.replace("\\\n", " ")
        self.assertIn("ARG ROCM_VERSION=7.2.4", build)
        self.assertIn("ARG ENABLE_ROCM=0", build)
        self.assertRegex(
            commands,
            r'(?s)if \[ "\$ENABLE_ROCM" = "1" \]; then.*?'
            r"https://repo\.radeon\.com/rocm/rocm\.gpg\.key.*?"
            r"https://repo\.radeon\.com/rocm/apt/\$\{ROCM_VERSION\} noble main.*?"
            r"Pin-Priority: 600.*?"
            r"apt-get install -y --no-install-recommends\s+"
            r"hipcc\s+hip-dev\s+rocm-device-libs\s+hipblas-dev\s+rocblas-dev",
        )

    def test_build_passes_exact_targets_and_hip_flags_and_stages_all_libraries(self) -> None:
        build = docker_stage("build")
        self.assertIn('ARG GPU_TARGETS=""', build)
        self.assertIn('${GPU_TARGETS:+"-DGPU_TARGETS=${GPU_TARGETS}"}', build)
        self.assertIn("-DGGML_NATIVE=OFF", build)
        self.assertIn(
            'scripts/package_rocm_libraries.sh build/third_party/ggml/src /install/lib',
            build,
        )
        for library in ("libggml.so", "libggml-base.so", "libggml-cpu.so", "libggml-hip.so"):
            self.assertIn(f'test -e "/install/lib/{library}"', build)

    def test_runtime_installs_rocm_userspace_and_final_stages_copy_payload(self) -> None:
        runtime_base = docker_stage("runtime-base")
        commands = runtime_base.replace("\\\n", " ")
        self.assertIn("ARG ROCM_VERSION=7.2.4", runtime_base)
        self.assertIn("ARG ENABLE_ROCM=0", runtime_base)
        self.assertRegex(
            commands,
            r'(?s)if \[ "\$ENABLE_ROCM" = "1" \]; then.*?'
            r"https://repo\.radeon\.com/rocm/apt/\$\{ROCM_VERSION\} noble main.*?"
            r"Pin-Priority: 600.*?"
            r"apt-get install -y --no-install-recommends\s+"
            r"rocm-hip-runtime\s+rocm-hip-libraries",
        )
        self.assertIn("COPY --from=build /install/lib/ /usr/local/lib/", runtime_base)
        self.assertIn(
            "COPY --from=build /install/bin/parakeet-cli /usr/local/bin/",
            docker_stage("runtime"),
        )
        self.assertIn(
            "COPY --from=build /install/bin/parakeet-server /usr/local/bin/",
            docker_stage("runtime-server"),
        )


class RocmDockerWorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        setup = re.search(
            r"(?ms)^  setup:\n(?P<body>.*?)(?=^  build:)", WORKFLOW
        )
        if setup is None:
            raise AssertionError("setup job is missing")
        cls.setup = setup.group("body")
        build = re.search(r"(?ms)^  build:\n(?P<body>.*?)(?=^  merge:)", WORKFLOW)
        if build is None:
            raise AssertionError("build job is missing")
        cls.build = build.group("body")
        merge = re.search(r"(?ms)^  merge:\n(?P<body>.*)\Z", WORKFLOW)
        if merge is None:
            raise AssertionError("merge job is missing")
        cls.merge = merge.group("body")

    def test_matrix_has_one_amd64_rocm_entry_and_prs_remain_cpu_only(self) -> None:
        rocm = re.search(r"(?m)^\s*ROCM='(?P<entry>[^']+)'$", self.setup)
        self.assertIsNotNone(rocm, "ROCM matrix declaration is missing")
        entry = rocm.group("entry")
        self.assertIn('"variant":"rocm"', entry)
        self.assertIn('"arch":"amd64"', entry)
        self.assertNotIn('"arch":"arm64"', entry)
        self.assertIn('"runner":"ubuntu-24.04"', entry)
        self.assertIn('"enable_rocm":"1"', entry)
        self.assertIn(f'"gpu_targets":"{ROCM_TARGETS}"', entry)
        self.assertIn(
            '"cmake_args":"-DBUILD_SHARED_LIBS=ON -DPARAKEET_GGML_HIP=ON '
            '-DGGML_HIP_NO_VMM=ON '
            '-DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ '
            '-DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm-7.2.4"',
            entry,
        )
        self.assertIn('matrix={\\"include\\":[${CPU}]}', self.setup)
        self.assertIn('matrix={\\"include\\":[${CPU},${CUDA},${ROCM}]}', self.setup)
        self.assertRegex(
            self.setup,
            r'(?s)if \[ "\$\{\{ github\.event_name \}\}" = "pull_request" \]; then.*?'
            r'\$\{CPU\}.*?else.*?\$\{CPU\},\$\{CUDA\},\$\{ROCM\}',
        )

    def test_both_image_builds_forward_rocm_arguments_on_linux_amd64(self) -> None:
        self.assertEqual(self.build.count("platforms: linux/${{ matrix.arch }}"), 2)
        self.assertEqual(self.build.count("ENABLE_ROCM=${{ matrix.enable_rocm }}"), 2)
        self.assertEqual(self.build.count("ROCM_VERSION=7.2.4"), 2)
        self.assertEqual(self.build.count("GPU_TARGETS=${{ matrix.gpu_targets }}"), 2)

    def test_merge_emits_cli_and_server_rocm_manifests_with_all_tag_families(self) -> None:
        for image in ("cli", "server"):
            entry = re.search(
                rf"(?ms)^          - image: {image}\n"
                r"            variant: rocm\n"
                r'            suffix: "-rocm"$',
                self.merge,
            )
            self.assertIsNotNone(entry, f"{image} ROCm merge entry is missing")
        self.assertIn("suffix=${{ matrix.suffix }},onlatest=true", self.merge)
        self.assertIn("type=raw,value=latest,enable={{is_default_branch}}", self.merge)
        self.assertIn("type=ref,event=tag", self.merge)
        self.assertIn("type=sha", self.merge)
        self.assertIn(
            "pattern: digests-${{ matrix.image }}-${{ matrix.variant }}-*", self.merge
        )
        self.assertIn("latest${{ matrix.suffix }}", self.merge)

    def test_push_tag_and_dispatch_events_are_enabled_while_existing_variants_remain(self) -> None:
        self.assertRegex(WORKFLOW, r"(?ms)^on:\n  push:\n    branches: \[master\]\n    tags: \['v\*'\]\n  pull_request:\n  workflow_dispatch:")
        for image, variant, suffix in (
            ("cli", "cpu", ""),
            ("cli", "cuda", "-cuda"),
            ("server", "cpu", ""),
            ("server", "cuda", "-cuda"),
        ):
            self.assertRegex(
                self.merge,
                rf'(?m)^          - image: {image}\n'
                rf"            variant: {variant}\n"
                rf'            suffix: "{re.escape(suffix)}"$',
            )


if __name__ == "__main__":
    unittest.main()
