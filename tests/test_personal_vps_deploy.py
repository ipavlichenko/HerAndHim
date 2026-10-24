from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "deploy" / "personal-vps" / "render_pilot_config.py"
IDENTITY_SCRIPT = ROOT / "deploy" / "personal-vps" / "install_identity.py"


def _load():
    spec = importlib.util.spec_from_file_location("render_pilot_config", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _load_identity_installer():
    spec = importlib.util.spec_from_file_location("install_identity", IDENTITY_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _secrets(tmp_path: Path) -> tuple[Path, set[str]]:
    directory = tmp_path / "secrets"
    directory.mkdir(mode=0o700)
    values = {
        "openrouter_api_key": "or-private-value",
        "openai_audio_api_key": "oa-private-value",
        "google_cloud_tts_api_key": "gc-private-value",
        "telegram_bot_token": "tg-private-value",
        "telegram_owner_chat_id": "123456789",
    }
    for name, value in values.items():
        path = directory / name
        path.write_text(value + "\n", encoding="utf-8")
        path.chmod(0o600)
    return directory, set(values.values())


def test_config_is_owner_only_exact_route_and_bounded(tmp_path):
    module = _load()
    secret_dir, _ = _secrets(tmp_path)
    config = module.build_config(secret_dir)

    telegram = config["channels"]["telegram"]
    assert telegram["allowedUsers"] == [123456789]
    assert config["proactive"]["chatId"] == 123456789
    assert config["selfie"]["chatId"] == 123456789
    assert config["web"]["accessToken"]
    assert config["tools"]["runCommand"] is False
    assert config["heartbeat"]["enabled"] is False
    assert config["agent"]["language"] == "ru"
    assert config["companion"] == {
        "companionName": "Сара",
        "companionAge": 28,
        "companionCountry": "RU",
        "companionRegion": "Moscow",
        "companionOccupation": "product designer",
    }
    assert config["user"]["timezone"] == "Europe/Moscow"
    assert config["persona"]["timezone"] == "Europe/Moscow"
    assert config["selfie"]["schedule"] == []

    purposes = config["outboundPolicy"]["purposes"]
    assert purposes["dialogue"]["model"] == "mistralai/mistral-large-4-0"
    assert purposes["dialogue"]["activeProvider"] == "mistral/eu"
    assert purposes["dialogue"]["maxInputTokensEstimated"] == 6000
    assert purposes["dialogue"]["maxCallsPerTurn"] == 2
    assert purposes["vision"]["model"] == "openai/gpt-6.1-sol"
    assert purposes["vision"]["activeProvider"] == "azure/eu"
    assert purposes["memory_consolidation"]["activeProvider"] == "google-vertex/europe"
    assert purposes["relationship_analysis"]["activeProvider"] == "google-vertex/europe"
    assert purposes["image_generation"]["maxCostUsdPerCall"] > 0
    assert all(row.get("maxCostUsdPerDay", 1) > 0 for row in purposes.values())
    assert config["quota"] == {"diskMb": 2048, "dailyMessages": 100, "dailyPhotos": 4}


def test_secret_permissions_fail_closed(tmp_path):
    module = _load()
    secret_dir, _ = _secrets(tmp_path)
    path = secret_dir / "openrouter_api_key"
    path.chmod(0o644)
    try:
        module.build_config(secret_dir)
    except PermissionError as exc:
        assert "0600" in str(exc)
    else:
        raise AssertionError("world-readable secret must be rejected")


def test_secret_values_do_not_appear_in_compose_or_ops(tmp_path):
    _, secret_values = _secrets(tmp_path)
    deployment = ROOT / "deploy" / "personal-vps"
    inspected = [deployment / "compose.yml", deployment / "sara-ops.sh", deployment / "README.md"]
    text = "\n".join(path.read_text(encoding="utf-8") for path in inspected)
    assert not any(value in text for value in secret_values)
    compose = (deployment / "compose.yml").read_text(encoding="utf-8")
    assert "127.0.0.1:7788:7788" in compose
    assert "no-new-privileges:true" in compose
    assert "max-size" in compose
    assert "init: true" not in compose
    assert "TZ: Europe/Moscow" in compose


def test_host_hardening_keeps_private_dashboard_tunnel_available():
    deployment = ROOT / "deploy" / "personal-vps"
    sshd = (deployment / "00-00-sara-sshd.conf").read_text(encoding="utf-8")
    assert "PasswordAuthentication no" in sshd
    assert "PermitRootLogin prohibit-password" in sshd
    assert "AllowTcpForwarding local" in sshd
    assert "GatewayPorts no" in sshd

    provision = (deployment / "provision-host.sh").read_text(encoding="utf-8")
    assert "sshd -t" in provision
    assert "systemctl reload ssh" in provision
    assert "systemctl enable --now sara-backup.timer" in provision

    backup = (deployment / "backup.py").read_text(encoding="utf-8")
    assert "os.umask(0o077)" in backup
    assert "BACKUPS.chmod(0o700)" in backup
    assert "archive.chmod(0o600)" in backup
    assert "manifest_path.chmod(0o600)" in backup


def test_runtime_secret_filter_redacts_credentials_and_owner_id():
    from herandhim.main import _RuntimeSecretFilter

    record = logging.LogRecord(
        "test", logging.INFO, __file__, 1,
        "token=%s owner=123456789", ("tg-private-value",), None,
    )
    assert _RuntimeSecretFilter(["tg-private-value", "123456789"]).filter(record)
    assert record.getMessage() == "token=[REDACTED] owner=[REDACTED]"


def test_explicit_empty_selfie_schedule_disables_fixed_slots(monkeypatch):
    from herandhim.scheduler import selfie_task

    monkeypatch.setattr(
        selfie_task.config,
        "get",
        lambda *args, **kwargs: [] if args == ("selfie", "schedule") else kwargs.get("default"),
    )
    scheduler = selfie_task.SelfieScheduler.__new__(selfie_task.SelfieScheduler)
    assert scheduler._read_schedule() == []


def test_identity_install_removes_conflicts_and_archives_them(tmp_path):
    module = _load_identity_installer()
    data = tmp_path / "data"
    persona_dir = data / "context" / "persona"
    persona_dir.mkdir(parents=True)
    (persona_dir / "boyfriend.md").write_text("conflict", encoding="utf-8")
    (persona_dir / "girlfriend.md").write_text("conflict", encoding="utf-8")

    archive = module.install(
        ROOT / "deploy" / "personal-vps" / "seed",
        data,
        uid=__import__("os").getuid(),
        gid=__import__("os").getgid(),
    )

    assert sorted(path.name for path in persona_dir.iterdir()) == [
        "appearance.md",
        "persona.md",
    ]
    appearance = (persona_dir / "appearance.md").read_text(encoding="utf-8")
    assert "hazel-green" in appearance
    assert "same facial identity" in appearance
    assert (archive / "context/persona/boyfriend.md").read_text() == "conflict"
    assert (archive / "context/persona/girlfriend.md").read_text() == "conflict"
    assert (data / "data/cities/RU.json").is_file()
    outbound = data / "context/outbound/dialogue_identity.md"
    assert outbound.is_file()
    assert outbound.stat().st_size < 3_000
    assert "Сара" in outbound.read_text(encoding="utf-8")

