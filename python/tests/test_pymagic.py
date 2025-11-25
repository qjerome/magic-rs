import pytest
import os
import tempfile
from pure_magic_rs import MagicDb, Magic


@pytest.fixture
def magic_db():
    return MagicDb()


@pytest.fixture
def sample_png_path():
    # Create a minimal PNG file for testing
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        # Minimal PNG header
        f.write(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
        )
        f.flush()
        yield f.name
    os.unlink(f.name)


def test_magic_db_new(magic_db):
    assert isinstance(magic_db, MagicDb)


def test_first_magic_buffer(magic_db):
    # Test with a PNG buffer
    png_buffer = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    result = magic_db.first_magic_buffer(png_buffer, None)
    assert isinstance(result, Magic)
    assert "PNG" in result.message
    assert result.mime_type == "image/png"


def test_first_magic_file(magic_db, sample_png_path):
    result = magic_db.first_magic_file(sample_png_path)
    assert isinstance(result, Magic)
    assert "PNG" in result.message
    assert result.mime_type == "image/png"


def test_best_magic_buffer(magic_db):
    png_buffer = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    result = magic_db.best_magic_buffer(png_buffer)
    assert isinstance(result, Magic)
    assert "PNG" in result.message
    assert result.mime_type == "image/png"


def test_best_magic_file(magic_db, sample_png_path):
    result = magic_db.best_magic_file(sample_png_path)
    assert isinstance(result, Magic)
    assert "PNG" in result.message
    assert result.mime_type == "image/png"


def test_all_magics_buffer(magic_db):
    png_buffer = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    results = magic_db.all_magics_buffer(png_buffer)
    assert isinstance(results, list)
    assert all(isinstance(r, Magic) for r in results)
    assert any("PNG" in r.message for r in results)
    assert any("image/png" == r.mime_type for r in results)
    assert any(r.message == "data" for r in results)
    assert any("application/octet-stream" == r.mime_type for r in results)


def test_all_magics_file(magic_db, sample_png_path):
    results = magic_db.all_magics_file(sample_png_path)
    assert isinstance(results, list)
    assert all(isinstance(r, Magic) for r in results)
    assert any("PNG" in r.message for r in results)
    assert any("image/png" == r.mime_type for r in results)
    assert any(r.message == "data" for r in results)
    assert any("application/octet-stream" == r.mime_type for r in results)


def test_to_dict(magic_db):
    png_buffer = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    result = magic_db.first_magic_buffer(png_buffer, None)
    d = result.to_dict()
    assert d["message"] == result.message
    assert d["mime_type"] == result.mime_type
    assert d["extensions"] == result.extensions


def test_file_not_found(magic_db):
    with pytest.raises(IOError):
        magic_db.first_magic_file("/nonexistent/file.png")
