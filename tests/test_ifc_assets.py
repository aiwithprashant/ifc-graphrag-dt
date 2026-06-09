from pipeline.ifc_assets import is_valid_ifc


def test_is_valid_ifc_rejects_html(tmp_path):
    path = tmp_path / "bad.ifc"
    path.write_bytes(b"<html>not an IFC file</html>" * 10_000)
    assert not is_valid_ifc(path)


def test_is_valid_ifc_accepts_step_header(tmp_path):
    path = tmp_path / "model.ifc"
    path.write_bytes(b"ISO-10303-21;\n" + b" " * 100_000)
    assert is_valid_ifc(path)
