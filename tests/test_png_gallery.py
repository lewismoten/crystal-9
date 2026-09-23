from pathlib import Path


def test_gallery_page_uses_the_png_manifest_and_nearest_neighbor_rendering():
    page = Path("png-model-gallery/index.html").read_text()

    assert 'fetch("manifest.json")' in page
    assert 'image-rendering: pixelated' in page
    assert "round-trip verified" in page
    assert 'fetch("inspectors.json")' in page
    assert "Derived tensor inspector" in page
