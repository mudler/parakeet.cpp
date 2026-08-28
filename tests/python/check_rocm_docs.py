#!/usr/bin/env python3

from pathlib import Path
import re
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
README = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
SERVER_README = (REPO_ROOT / "examples" / "server" / "README.md").read_text(
    encoding="utf-8"
)
RELEASE_WORKFLOW = (
    REPO_ROOT / ".github" / "workflows" / "release.yml"
).read_text(encoding="utf-8")
DOCKER_WORKFLOW = (
    REPO_ROOT / ".github" / "workflows" / "docker.yml"
).read_text(encoding="utf-8")


def workflow_rocm_targets(workflow: str) -> str:
    match = re.search(
        r'gpu_targets(?:\\?"|):\s*(?:\\?")(?P<targets>gfx[^\"]+)', workflow
    )
    if match is None:
        raise AssertionError("ROCm GPU targets are missing from the workflow")
    return match.group("targets")


class RocmDocumentationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.release_targets = workflow_rocm_targets(RELEASE_WORKFLOW)
        cls.docker_targets = workflow_rocm_targets(DOCKER_WORKFLOW)

    def test_release_matrix_and_archive_names_include_linux_x64_rocm(self) -> None:
        self.assertRegex(
            README,
            r"(?m)^\| Linux x64 \|[^\n]*\brocm\b[^\n]*\|$",
        )
        for archive in (
            "parakeet-<version>-bin-linux-rocm-x64.tar.gz",
            "parakeet-<version>-lib-linux-rocm-x64.tar.gz",
        ):
            self.assertIn(archive, README)

    def test_thin_archive_contract_names_rocm_72_and_official_install_docs(self) -> None:
        self.assertRegex(README, r"(?is)thin archives?.{0,240}ROCm 7\.2")
        self.assertIn("https://rocm.docs.amd.com/projects/install-on-linux/", README)

    def test_documented_targets_match_both_workflows_and_describe_gpu_families(self) -> None:
        self.assertEqual(self.release_targets, self.docker_targets)
        self.assertIn(self.release_targets, README)
        for coverage in ("Instinct", "Radeon", "Ryzen AI", "Strix Halo"):
            self.assertIn(coverage, README)
        self.assertIn("Use the Vulkan or CPU archive as a fallback", README)

    def test_source_build_documents_exact_release_flags(self) -> None:
        rocm_entry = re.search(
            r"(?ms)^\s*- backend: rocm\n(?P<body>.*?)(?=^\s*- backend:|^\s*steps:)",
            RELEASE_WORKFLOW,
        )
        self.assertIsNotNone(rocm_entry)
        cmake_args = re.search(
            r'^\s*cmake_args: "(?P<args>[^"]+)"$',
            rocm_entry.group("body"),
            re.MULTILINE,
        )
        self.assertIsNotNone(cmake_args)
        for flag in cmake_args.group("args").split():
            self.assertIn(flag, README)
        self.assertIn(f'-DGPU_TARGETS="{self.release_targets}"', README)
        self.assertIn("-DGGML_NATIVE=OFF", README)

    def test_runtime_selection_and_fallback_are_documented(self) -> None:
        for document in (README, SERVER_README):
            self.assertIn("ROCm0", document)
            self.assertIn("PARAKEET_DEVICE=ROCm0", document)
            self.assertIn("PARAKEET_DEVICE=cpu", document)

    def test_rocm_docker_tags_and_device_access_match_published_variants(self) -> None:
        for image in ("cli", "server"):
            self.assertRegex(
                DOCKER_WORKFLOW,
                rf"(?m)^\s*- image: {image}\n\s*variant: rocm\n\s*suffix: \"-rocm\"$",
            )
        for document in (README, SERVER_README):
            for tag in ("latest-rocm", "<version>-rocm", "<sha>-rocm"):
                self.assertIn(tag, document)
            for option in (
                "--device=/dev/kfd",
                "--device=/dev/dri",
                "--group-add video",
            ):
                self.assertIn(option, document)
        self.assertIn("ghcr.io/mudler/parakeet.cpp-cli:latest-rocm", README)
        self.assertIn("ghcr.io/mudler/parakeet.cpp-server:latest-rocm", README)
        self.assertIn("ghcr.io/mudler/parakeet.cpp-server:latest-rocm", SERVER_README)

    def test_rocm_examples_cover_model_mounting_and_server_model_fetching(self) -> None:
        self.assertRegex(
            README,
            r"(?s)-v \"\$PWD/models:/models:ro\".*?parakeet\.cpp-cli:latest-rocm",
        )
        for document in (README, SERVER_README):
            self.assertRegex(
                document,
                r"(?s)parakeet\.cpp-server:latest-rocm.*?--model tdt_ctc-110m",
            )
            self.assertRegex(
                document,
                r"(?s)-v \"\$PWD/model\.gguf:/model\.gguf:ro\".*?"
                r"parakeet\.cpp-server:latest-rocm.*?--model /model\.gguf",
            )

    def test_strix_halo_correctness_and_indicative_timings_are_recorded(self) -> None:
        transcript = (
            "Well, I don't wish to see it any more, observed Phoebe, turning away "
            "her eyes. It is certainly very like the old portrait."
        )
        for document in (README, SERVER_README):
            self.assertIn(transcript, document)
            for timing in ("34.505 ms", "53.914 ms", "68.015 ms"):
                self.assertIn(timing, document)
            self.assertRegex(document, r"(?is)indicative.{0,120}not guaranteed")

    def test_no_document_claims_that_only_cpu_and_cuda_images_exist(self) -> None:
        stale_claims = (
            r"(?i)each comes in a CPU and a CUDA variant",
            r"(?i)\(CPU by default, `:latest-cuda` for the CUDA build\)",
            r"(?i)each (?:image )?ships only CPU and CUDA",
        )
        for document in (README, SERVER_README):
            for claim in stale_claims:
                self.assertNotRegex(document, claim)


if __name__ == "__main__":
    unittest.main()
