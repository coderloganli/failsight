import zipfile

from failsight.download import archive_url, extract_daily_csvs


def test_archive_url():
    base = "https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/"
    assert archive_url(base, "Q1_2026") == f"{base}data_Q1_2026.zip"


def test_extract_flattens_daily_csvs_and_skips_metadata(tmp_path):
    archive = tmp_path / "data_Q1_2026.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("data_Q1_2026/2026-01-01.csv", "date,serial_number\n")
        zf.writestr("__MACOSX/data_Q1_2026/._2026-01-01.csv", "junk")
        zf.writestr("data_Q1_2026/README.txt", "notes")
    written = extract_daily_csvs(archive, tmp_path / "raw")
    assert [p.name for p in written] == ["2026-01-01.csv"]
    assert (tmp_path / "raw" / "2026-01-01.csv").read_text() == "date,serial_number\n"
