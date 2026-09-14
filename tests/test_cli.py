import pytest

from real_estate_ai.cli import main


def test_cli_displays_property_score(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main()

    output = capsys.readouterr().out

    assert output == ("Property: DXB-1001\nMatch score: 91.45/100\n")
