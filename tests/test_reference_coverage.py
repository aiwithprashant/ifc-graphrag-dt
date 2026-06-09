from benchmark.reference_coverage import normalize_ifc_type


def test_normalize_ifc_type():
    assert normalize_ifc_type("IfcPump[duty]") == "IfcPump"
    assert normalize_ifc_type("IfcWall") == "IfcWall"
