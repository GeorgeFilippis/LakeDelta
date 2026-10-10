"""The Earth Engine smoke-test script itself runs end to end (offline harness)."""

import importlib.util
import os
import sys

from ee_dryrun import dry_run
import test_ee_dryrun as T
from lakedelta import ee_data as E

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_smoke_script_offline(monkeypatch, tmp_path, capsys):
    class R:
        content = b"GIF89a-not-really"

        def raise_for_status(self):
            return None

    monkeypatch.setattr(E.requests, "get", lambda url, timeout=0: R())
    spec = importlib.util.spec_from_file_location("smoke", os.path.join(ROOT, "scripts", "ee_smoke_test.py"))
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    monkeypatch.setattr(smoke, "init", lambda args: None)
    monkeypatch.setattr(sys, "argv", ["smoke", "--project", "x", "--full", "--out", str(tmp_path)])
    with dry_run(T.responder):
        try:
            smoke.main()
        except SystemExit as e:
            code = e.code
    out = capsys.readouterr().out
    assert code == 0, out
    assert (tmp_path / "mornos_report.pdf").exists()
