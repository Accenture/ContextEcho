import hashlib
from unittest import mock

from donate import device_id as did


def test_device_id_deterministic_and_one_way():
    with mock.patch.object(did, "hardware_identifier", return_value="abc-hw-uuid"), \
            mock.patch.object(did.getpass, "getuser", return_value="alice"):
        first = did.device_id()
        second = did.device_id()
    assert first == second
    expected = hashlib.sha256(b"contextecho-device-v1|abc-hw-uuid|alice").hexdigest()[:32]
    assert first == expected
    # raw hardware id never appears in the output
    assert "abc-hw-uuid" not in first


def test_device_id_differs_by_user_on_shared_machine():
    with mock.patch.object(did, "hardware_identifier", return_value="abc-hw-uuid"):
        with mock.patch.object(did.getpass, "getuser", return_value="alice"):
            a = did.device_id()
        with mock.patch.object(did.getpass, "getuser", return_value="bob"):
            b = did.device_id()
    assert a != b


def test_no_hardware_identifier_means_no_device_id():
    with mock.patch.object(did, "hardware_identifier", return_value=""):
        assert did.device_id() == ""


def test_is_valid_device_id():
    assert did.is_valid_device_id("a" * 32)
    assert did.is_valid_device_id("0123456789abcdef0123456789abcdef")
    assert not did.is_valid_device_id("")
    assert not did.is_valid_device_id("A" * 32)  # uppercase rejected
    assert not did.is_valid_device_id("a" * 31)
    assert not did.is_valid_device_id(None)
    assert not did.is_valid_device_id(123)


def test_macos_parser_extracts_uuid():
    ioreg = '    "IOPlatformUUID" = "D9E2F8A1-1234-5678-9ABC-DEF012345678"\n'
    with mock.patch.object(did.subprocess, "run") as run:
        run.return_value = mock.Mock(stdout=ioreg)
        assert did._macos_hardware_uuid() == "d9e2f8a1-1234-5678-9abc-def012345678"
