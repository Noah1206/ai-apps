from pathlib import Path

from busan_lab.environment import inspect_project_environment


def test_project_environment_accepts_only_pinned_local_venv(tmp_path: Path) -> None:
    root = tmp_path / "project"
    environment = root / ".venv"
    environment.mkdir(parents=True)
    (root / ".python-version").write_text("3.13.14\n", encoding="utf-8")
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")

    report = inspect_project_environment(
        root,
        python_prefix=environment,
        python_version="3.13.14",
        executable_lookup=lambda name: f"/tools/{name}",
    )

    assert report.passed is True


def test_project_environment_rejects_global_python(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / ".venv").mkdir(parents=True)
    (root / ".python-version").write_text("3.13.14\n", encoding="utf-8")
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")

    report = inspect_project_environment(
        root,
        python_prefix=tmp_path / "global-python",
        python_version="3.14.5",
        executable_lookup=lambda name: f"/tools/{name}",
    )

    assert report.passed is False
    failed = {check.name for check in report.checks if not check.passed}
    assert failed == {"project_venv", "python_version"}
