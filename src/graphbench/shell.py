"""Resolve Bash and build the unchanged upstream shell invocation portably."""

import os
import shutil
from pathlib import Path


def find_bash() -> str:
    configured = os.environ.get("GRAPHBENCH_BASH")
    if configured:
        executable = shutil.which(configured) or configured
        if not Path(executable).is_file():
            raise FileNotFoundError(f"GRAPHBENCH_BASH does not exist: {configured}")
        return str(executable)
    if os.name == "nt":
        # Prefer Git's Bash over the Windows/WSL launcher named bash.exe.
        git = shutil.which("git")
        candidates = []
        if git:
            install = Path(git).resolve().parent.parent
            candidates += [install / "bin/bash.exe", install / "usr/bin/bash.exe"]
        for variable in ("ProgramFiles", "LOCALAPPDATA"):
            if os.environ.get(variable):
                candidates.append(
                    Path(os.environ[variable])
                    / (
                        "Programs/Git/bin/bash.exe"
                        if variable == "LOCALAPPDATA"
                        else "Git/bin/bash.exe"
                    )
                )
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
    else:
        executable = shutil.which("bash")
        if executable:
            return executable
    raise FileNotFoundError(
        "Bash is required only for official --fresh-pipeline. Install Git Bash (Windows) or bash/gzip/coreutils (Unix), or set GRAPHBENCH_BASH."
    )


def injection_command(bash, python, script, dataset, percentage):
    # All user paths travel through argv; the shell program contains no interpolation.
    launcher = 'interpreter="$1"; export interpreter; shift; python3() { set -o pipefail; "$interpreter" "$@" | tr -d "\\r"; }; export -f python3; bash "$@"'
    return [
        str(bash),
        "-c",
        launcher,
        "official-injection",
        Path(python).as_posix(),
        script,
        dataset,
        percentage,
    ]
