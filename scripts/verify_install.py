"""Verify an already built wheel in a temporary, offline virtual environment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import venv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-dir", type=Path, default=Path(__file__).resolve().parents[1] / "dist")
    args = parser.parse_args()
    wheels = sorted(args.wheel_dir.resolve().glob("model_sentinel-*.whl"))
    if len(wheels) != 1:
        parser.error(f"expected exactly one model_sentinel wheel in {args.wheel_dir}; found {len(wheels)}")

    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    with tempfile.TemporaryDirectory(prefix="model-sentinel-install-") as directory:
        root = Path(directory)
        venv_path = root / "venv"
        outside = root / "outside-checkout"
        outside.mkdir()
        venv.EnvBuilder(with_pip=True).create(venv_path)
        binary_dir = venv_path / ("Scripts" if os.name == "nt" else "bin")
        python = binary_dir / ("python.exe" if os.name == "nt" else "python")
        command = binary_dir / ("model-sentinel.exe" if os.name == "nt" else "model-sentinel")

        def run(*arguments: str) -> None:
            subprocess.run(arguments, cwd=outside, env=environment, check=True)

        run(str(python), "-I", "-m", "pip", "install", "--no-index", "--no-deps", "--disable-pip-version-check", str(wheels[0]))
        run(str(command), "--version")
        run(str(command), "--help")
        run(str(python), "-I", "-c", "\n".join([
            "from pathlib import Path",
            "import subprocess, sys",
            "import model_sentinel",
            "from model_sentinel.pack import DEFAULT_PACK, load_pack",
            "assert Path(model_sentinel.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())",
            "pack = load_pack()",
            "assert pack['schema_version'] == 1 and len(pack['cases']) == 32",
            "subprocess.run([sys.executable, '-I', '-m', 'model_sentinel', 'validate-pack', str(DEFAULT_PACK)], check=True)",
        ]))
    print("Installed wheel, CLI, and bundled probe pack verified.")


if __name__ == "__main__":
    main()
