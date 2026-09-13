def test_package_imports() -> None:
    import bonsai

    assert bonsai.__all__ == []
