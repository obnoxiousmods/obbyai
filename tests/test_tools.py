"""Basic tests for tool modules."""
import asyncio
import pytest


def test_calculator_basic():
    from tools.calculator import _safe_eval
    assert _safe_eval("2 + 2") == 4
    assert _safe_eval("10 * 10") == 100
    assert _safe_eval("sqrt(144)") == 12.0
    assert _safe_eval("2 ** 10") == 1024


def test_calculator_trig():
    from tools.calculator import _safe_eval
    import math
    assert abs(_safe_eval("sin(pi/2)") - 1.0) < 1e-10
    assert abs(_safe_eval("cos(0)") - 1.0) < 1e-10


def test_calculator_rejects_dangerous():
    from tools.calculator import _safe_eval
    with pytest.raises(Exception):
        _safe_eval("__import__('os').system('id')")
    with pytest.raises(Exception):
        _safe_eval("open('/etc/passwd').read()")


@pytest.mark.asyncio
async def test_calculator_run():
    from tools.calculator import run
    result = await run("2 ** 16")
    assert result["result"] == "65536"
    assert "error" not in result


def test_file_processor_detect_image():
    from tools.file_processor import _is_image
    jpeg_header = b"\xff\xd8\xff\xe0" + b"\x00" * 100
    png_header  = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    assert _is_image(jpeg_header) is True
    assert _is_image(png_header) is True
    assert _is_image(b"PK\x03\x04") is False


def test_file_processor_text_decode():
    from tools.file_processor import _decode_text
    text = _decode_text(b"Hello, world!", "test.txt")
    assert text == "Hello, world!"

    json_data = b'{"key": "value", "num": 42}'
    result = _decode_text(json_data, "data.json")
    assert '"key"' in result


def test_chunk_text():
    from tools.rag import chunk_text
    long_text = "word " * 200  # 1000 chars
    chunks = chunk_text(long_text, "test")
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk["text"]) <= 520  # CHUNK_SIZE + CHUNK_OVERLAP buffer
        assert chunk["source"] == "test"
