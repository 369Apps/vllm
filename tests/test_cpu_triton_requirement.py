# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU wheel metadata must not pin triton with a direct URL.

``setup.py`` copies ``requirements/cpu.txt`` into ``install_requires``.
Comments and ``--`` installer options are dropped, and ``-r`` lines are
expanded. A ``triton @ https://...`` line becomes a URL ``Requires-Dist``,
which ``uv`` refuses and which an air-gapped install cannot fetch.
"""

import subprocess
import sys
from pathlib import Path

from packaging.requirements import Requirement

REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS_DIR = REPO_ROOT / "requirements"
CPU_REQUIREMENTS = REQUIREMENTS_DIR / "cpu.txt"
TRITON_CPU_FIND_LINKS = (
    "https://github.com/triton-lang/triton-cpu/releases/expanded_assets/v3.8.0"
)


def _read_requirements(filename: str) -> list[str]:
    """Return requirement lines the way ``setup.py`` feeds ``install_requires``."""
    text = (REQUIREMENTS_DIR / filename).read_text()
    resolved: list[str] = []
    for line in text.splitlines():
        if line.startswith("-r "):
            resolved += _read_requirements(line.split()[1])
        elif (
            not line.startswith("--")
            and not line.startswith("#")
            and line.strip() != ""
        ):
            resolved.append(line)
    return resolved


def test_cpu_triton_pin_is_a_version_requirement() -> None:
    """The x86 triton-cpu pin is a version plus a platform marker.

    ``--find-links`` stays in the requirements file so image and source
    installs can still download the wheel, but it is not a package requirement.
    """
    text = CPU_REQUIREMENTS.read_text()
    assert f"--find-links {TRITON_CPU_FIND_LINKS}" in text

    triton_reqs = [
        req for req in _read_requirements("cpu.txt") if req.strip().startswith("triton")
    ]
    assert triton_reqs == ['triton==3.8.0+cpu ; platform_machine == "x86_64"']
    for req in triton_reqs:
        assert Requirement(req).url is None


def test_cpu_wheel_metadata_has_no_direct_url(tmp_path: Path) -> None:
    """Rendered METADATA records triton by version and contains no URL."""
    requirements = _read_requirements("cpu.txt")
    setup_py = tmp_path / "setup.py"
    setup_py.write_text(
        "from setuptools import setup\n"
        "setup(name='cpu-req-stub', version='0.0.0', "
        f"install_requires={requirements!r})\n"
    )
    subprocess.run(
        [sys.executable, "setup.py", "dist_info"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    metadata_path = tmp_path / "cpu_req_stub-0.0.0.dist-info" / "METADATA"
    metadata = metadata_path.read_text()

    assert "://" not in metadata
    assert " @ " not in metadata

    parsed = [
        Requirement(line.split(":", 1)[1].strip())
        for line in metadata.splitlines()
        if line.startswith("Requires-Dist:")
    ]
    triton_reqs = [req for req in parsed if req.name == "triton"]
    assert len(triton_reqs) == 1
    triton_req = triton_reqs[0]
    assert triton_req.url is None
    assert str(triton_req.specifier) == "==3.8.0+cpu"
    assert str(triton_req.marker) == 'platform_machine == "x86_64"'
    assert all(req.url is None for req in parsed)
