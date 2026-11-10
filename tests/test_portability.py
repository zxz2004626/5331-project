"""Exercise real entry points after relocating sources to a path with spaces."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from graph_anomaly.artifacts import (
    EVALUATION_SOURCES,
    GENERATION_SOURCES,
    artifact_path,
    source_digest,
)
from graph_anomaly.cli import COMMANDS, pipeline_steps
from graph_anomaly.config import ROOT, project_root
from graph_anomaly.official import find_bash, injection_command
from graph_anomaly.results import aggregate_results, load_split_counts
from graph_anomaly.verify import generation_hash, implementation_hash


@pytest.fixture
def moved_project(tmp_path):
    moved = tmp_path / "relocated project with spaces"
    shutil.copytree(
        ROOT / "src", moved / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info")
    )
    shutil.copytree(ROOT / "configs", moved / "configs")
    elsewhere = tmp_path / "another working directory"
    elsewhere.mkdir()
    environment = os.environ.copy()
    environment.pop("GRAPH_ANOMALY_ROOT", None)
    environment["PYTHONPATH"] = str(moved / "src")
    environment["PYTHONIOENCODING"] = "utf-8"
    return moved, elsewhere, environment


def test_cli_uses_relocated_sources_outside_the_checkout(moved_project):
    moved, elsewhere, environment = moved_project
    for args in (["-m", "graph_anomaly", "run", "--dry-run"],):
        result = subprocess.run(
            [sys.executable, *args],
            cwd=elsewhere,
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "--fresh-pipeline" in result.stdout
        assert "--paired-timing" in result.stdout
        assert not (moved / "vendor").exists()
        assert not (moved / "results").exists()
    result = subprocess.run(
        [sys.executable, "-m", "graph_anomaly", "--root", str(moved), "generate", "--help"],
        cwd=elsewhere,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--workers" in result.stdout


@pytest.mark.parametrize("command", COMMANDS)
def test_each_command_help_is_side_effect_free(moved_project, command):
    moved, elsewhere, environment = moved_project
    subprocess.run(
        [sys.executable, "-m", "graph_anomaly", command, "--help"],
        cwd=elsewhere,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert not (moved / "vendor").exists()
    assert not (moved / "results").exists()


def test_source_hashes_survive_newlines_but_include_extracted_dependencies(tmp_path):
    sources = tmp_path / "src/graph_anomaly"
    sources.mkdir(parents=True)
    for name in set(GENERATION_SOURCES + EVALUATION_SOURCES):
        (sources / name).write_bytes(b"# example\nvalue = 1\n")
    generation = source_digest(sources, GENERATION_SOURCES)
    evaluation = source_digest(sources, EVALUATION_SOURCES)
    for path in sources.iterdir():
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    assert source_digest(sources, GENERATION_SOURCES) == generation
    assert source_digest(sources, EVALUATION_SOURCES) == evaluation
    assert generation_hash(tmp_path) == generation
    assert implementation_hash(tmp_path, EVALUATION_SOURCES) == evaluation
    (sources / "models.py").write_bytes(b"# model implementation changed\n")
    assert source_digest(sources, EVALUATION_SOURCES) != evaluation
    assert source_digest(sources, GENERATION_SOURCES) == generation


def test_project_root_override_and_missing_configuration(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/experiment.json").write_text("{}")
    monkeypatch.setenv("GRAPH_ANOMALY_ROOT", str(tmp_path))
    assert project_root() == tmp_path.resolve()
    monkeypatch.setenv("GRAPH_ANOMALY_ROOT", str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="configs/experiment.json"):
        project_root()


def test_official_modes_are_separate_and_paths_remain_argv():
    steps = pipeline_steps()
    official = [step for step in steps if step[0] == "official"]
    assert official == [
        ["official"],
        ["official", "--paired-timing"],
        ["official", "--fresh-pipeline"],
        ["official", "--verify-output"],
    ]
    command = injection_command(
        "path with spaces/bash", "path with spaces/python", "inject.sh", "movielens", "5"
    )
    assert command[4:] == ["path with spaces/python", "inject.sh", "movielens", "5"]
    assert "path with spaces" not in command[2]


def test_report_counts_can_come_from_audited_metadata(tmp_path):
    audit = tmp_path / "results/validation/final_audit.json"
    audit.parent.mkdir(parents=True)
    counts = [{"normal": 100, "anomaly": 5}] * 3
    audit.write_text(
        json.dumps(
            {
                "cache_checksums_and_counts": [
                    {
                        "scenario": "main",
                        "dataset": "movielens",
                        "seed": 11,
                        "generator": "G0",
                        "split_counts": counts,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert load_split_counts(tmp_path, "movielens") == counts


def test_legacy_windows_artifact_paths_work_after_relocation(tmp_path):
    expected = tmp_path / "results/evaluation/main/runs/example.json"
    assert artifact_path(tmp_path, "results/evaluation/main/runs/example.json") == expected
    assert artifact_path(tmp_path, r"results\evaluation\main\runs\example.json") == expected


def test_bash_child_script_receives_the_current_python_interpreter(tmp_path):
    try:
        bash = find_bash()
    except FileNotFoundError:
        pytest.skip("Bash unavailable; official shell integration needs Git Bash on Windows")
    script = tmp_path / "script with spaces.sh"
    script.write_text(
        "python3 -c 'import sys; print(sys.executable); print(sys.argv[1:])' \"$@\"\n",
        encoding="utf-8",
        newline="\n",
    )
    result = subprocess.run(
        injection_command(bash, sys.executable, script.as_posix(), "movielens", "5"),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "['movielens', '5']" in result.stdout
    assert Path(result.stdout.splitlines()[0]).resolve() == Path(sys.executable).resolve()


def test_aggregation_does_not_erase_historical_tables_without_current_caches(tmp_path):
    output = tmp_path / "results/evaluation/main"
    runs = output / "runs"
    runs.mkdir(parents=True)
    (runs / "historical.json").write_text(
        json.dumps({"status": "complete", "evaluation_source_sha256": "historical-implementation"})
    )
    original = "historical,metrics\n1,2\n"
    (output / "metrics.csv").write_text(original, encoding="utf-8")
    with pytest.raises(ValueError, match="summary tables were preserved"):
        aggregate_results(output)
    assert (output / "metrics.csv").read_text(encoding="utf-8") == original
