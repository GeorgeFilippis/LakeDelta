"""Earth Engine sign-in. Falls back to demo mode when no credentials exist."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping

EE_SCOPES = [
    "https://www.googleapis.com/auth/earthengine",
    "https://www.googleapis.com/auth/cloud-platform",
]


@dataclass
class EEStatus:
    ok: bool
    mode: str          # "service-account", "user" or "demo"
    message: str
    project: str | None = None


def _secret_table(secrets: Mapping[str, Any] | None) -> dict | None:
    if secrets is None:
        return None
    for key in ("earth_engine", "gcp_service_account", "ee"):
        try:
            table = secrets[key]
        except Exception:  # noqa: BLE001 - st.secrets raises its own errors
            continue
        try:
            return dict(table)
        except Exception:  # noqa: BLE001
            continue
    return None


def initialize(secrets: Mapping[str, Any] | None = None) -> EEStatus:
    """Initialise Earth Engine from (in order) Streamlit secrets, environment, local login."""
    if os.environ.get("LAKEDELTA_DEMO", "").strip() == "1":
        return EEStatus(False, "demo", "Demo mode was requested with LAKEDELTA_DEMO=1.")
    try:
        import ee
    except ImportError:
        return EEStatus(False, "demo", "The earthengine-api package is not installed.")

    info = _secret_table(secrets)
    if info is None and os.environ.get("EE_SERVICE_ACCOUNT_JSON"):
        try:
            info = json.loads(os.environ["EE_SERVICE_ACCOUNT_JSON"])
        except json.JSONDecodeError as exc:
            return EEStatus(False, "demo", f"EE_SERVICE_ACCOUNT_JSON is not valid JSON ({exc}).")

    project = os.environ.get("EE_PROJECT") or (info or {}).get("project_id")
    try:
        if info:
            from google.oauth2 import service_account

            if "private_key" in info and "\\n" in info["private_key"]:
                info["private_key"] = info["private_key"].replace("\\n", "\n")
            creds = service_account.Credentials.from_service_account_info(info, scopes=EE_SCOPES)
            ee.Initialize(credentials=creds, project=project)
            return EEStatus(True, "service-account",
                            f"Signed in as {info.get('client_email', 'service account')}.", project)
        if project:
            ee.Initialize(project=project)
            return EEStatus(True, "user", f"Signed in with local credentials (project {project}).",
                            project)
    except Exception as exc:  # noqa: BLE001 - surface any auth failure to the UI
        return EEStatus(False, "demo", f"Earth Engine sign-in failed: {exc}")
    return EEStatus(False, "demo", "No Earth Engine credentials were found.")
