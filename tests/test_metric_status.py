import pytest

from app.services.heart_agent import HeartMRIAgent


@pytest.fixture
def agent():
    return HeartMRIAgent.__new__(HeartMRIAgent)


@pytest.mark.parametrize(
    "key,value,expected",
    [
        ("LV_EDV", 155.0, "normal"),
        ("LV_EDV", 155.01, "high"),
        ("LV_EDV", 55.99, "low"),
        ("RV_CO", 4.0, "normal"),
        ("LV_BS_02_mean", 14.7, "high"),
        ("LV_SP_16_mean", 5.9, "low"),
        ("LV_EF", 39.9, "severely_reduced"),
        ("RV_EF", 66.0, "elevated"),
        ("RV_BS_01", 5.6, "unknown"),
        ("LV_TP_17_mean", 3.0, "unknown"),
    ],
)
def test_metric_status_uses_displayed_reference_ranges(agent, key, value, expected):
    assert agent._get_metric_status(key, value) == expected


def test_report_does_not_hardcode_wall_thickness_as_normal(agent):
    report = agent._build_report_data(
        {"LV_BS_01_mean": 14.7, "LV_SP_16_mean": 5.5, "RV_BS_01": 2.5},
        {},
        {},
    )
    items = {
        item["key"]: item
        for section in report["sections"]
        for item in section["items"]
    }
    assert items["LV_BS_01_mean"]["status"] == "high"
    assert items["LV_SP_16_mean"]["status"] == "low"
    assert items["RV_BS_01"]["status"] == "unknown"
