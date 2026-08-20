import pytest
from fastapi import HTTPException

from seismic_lab.api import parse_csv


def test_parse_csv_accepts_exact_three_channel_waveform():
    payload = (
        "east_west,north_south,vertical\n"
        + "0.1,0.2,0.3\n" * 6000
    ).encode()
    assert parse_csv(payload).shape == (6000, 3)


def test_parse_csv_rejects_missing_channel():
    payload = ("east_west,north_south\n" + "0.1,0.2\n" * 6000).encode()
    with pytest.raises(HTTPException, match="channel columns"):
        parse_csv(payload)


def test_parse_csv_rejects_wrong_duration():
    payload = (
        "east_west,north_south,vertical\n"
        + "0.1,0.2,0.3\n" * 5999
    ).encode()
    with pytest.raises(HTTPException, match="exactly 6000 rows"):
        parse_csv(payload)
