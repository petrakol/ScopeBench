from __future__ import annotations

import json
import sys
from pathlib import Path

from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scopebench.cli import app


def test_quickstart_writes_artifacts(tmp_path):
    runner = CliRunner()
    out_dir = tmp_path / "artifacts"

    result = runner.invoke(app, ["quickstart", "--json", "--out-dir", str(out_dir)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["decision"] in {"ALLOW", "ASK", "DENY"}
    assert payload["dataset"]["count"] > 0
    assert "artifacts" in payload
    assert (out_dir / "result.json").exists()
    assert (out_dir / "summary.md").exists()


def test_plugin_try_returns_commands():
    runner = CliRunner()
    result = runner.invoke(app, ["plugin-try", "robotics", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["domain"] == "robotics"
    assert "plugin-generate" in payload["commands"]["generate"]
    assert payload["smoke_eval"]["decision"] in {"ALLOW", "ASK", "DENY"}


def test_plugin_try_unknown_domain_fails():
    runner = CliRunner()
    result = runner.invoke(app, ["plugin-try", "unknown-domain"])

    assert result.exit_code != 0
    assert "Unknown plugin marketplace domain" in result.output
