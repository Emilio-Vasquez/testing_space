"""
Flask application serving the astrophysics data pipeline dashboard.

This version of the dashboard focuses on presenting the end products (calibrated
images) on the home page and streaming raw FITS images line by line on
 dedicated pages. In addition to streaming the image, it also exposes basic
 statistics and a histogram for each raw dataset so the frontend can render
 numeric charts instead of just a grayscale picture. All backend logic lives
 inside this module.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any

from flask import Flask, render_template, jsonify, send_from_directory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mapping of available raw datasets.  Each entry maps a slug to a dict with a
# human‑readable title and the relative path (from the FITS root) to the
# corresponding FITS file.  Modify this mapping if you add new raw data.
RAW_OBJECTS: Dict[str, Dict[str, str]] = {
    'pillars': {
        'title': 'Pillars of Creation (Eagle Nebula)',
        # Level‑2 HST science frame downloaded via get_mast_science_and_product.py.
        'file': 'HST/ice00ap1q/ice00ap1q_raw.fits',
    },
    'stephans': {
        'title': "Stephan’s Quintet",
        # Level‑2 drizzled HLA image; contains real structure.
        'file': 'HLA/HST_8063_p7_NIC_NIC3_F110W_01/HST_8063_p7_NIC_NIC3_F110W_01_drz.fits',
    },
    'smacs': {
        'title': 'Webb’s First Deep Field (SMACS 0723)',
        # JWST level‑2 NIRSpec calibrated frame used as a proxy for raw input.
        'file': 'JWST/jw02736006001_02101_00001_nrs1/jw02736006001_02101_00001_nrs1_cal.fits',
    },
}

def _parse_fits_image(path: Path):
    """Parse a FITS file and return a 2D numpy array of pixel values.

    This parser supports FITS files where the primary HDU has no image
    (``NAXIS == 0``) and the image data resides in an extension.  It
    sequentially reads headers until it finds an extension with
    non‑zero ``NAXIS1`` and ``NAXIS2`` values and a valid ``BITPIX``.
    If more than two dimensions are present (e.g. spectral cubes), the
    additional axes are collapsed by taking the first slice along those
    dimensions.  Returns ``None`` if no suitable image is found.
    """
    import numpy as np

    def read_header(file_obj) -> (bytes, Dict[str, str]):
        """Read a FITS header from the current position and return raw bytes
        and a dictionary of header cards."""
        header_bytes = b''
        while True:
            block = file_obj.read(2880)
            if not block:
                break
            header_bytes += block
            if b'END' in block:
                break
        header_str = header_bytes.decode('ascii', 'ignore')
        header = {}
        # Parse 80‑character cards
        for i in range(0, len(header_str), 80):
            card = header_str[i:i + 80]
            if card.strip().startswith('END'):
                break
            if '=' in card:
                key_part, value_comment = card.split('=', 1)
                key = key_part.strip()
                value = value_comment.split('/')[0].strip()
                if value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                header[key] = value
        return header_bytes, header

    with open(path, 'rb') as f:
        # Read primary header
        header_bytes, header = read_header(f)
        try:
            width = int(header.get('NAXIS1', '0'))
            height = int(header.get('NAXIS2', '0'))
            bitpix = int(header.get('BITPIX', '0'))
        except Exception:
            width = height = bitpix = 0
        # If primary HDU doesn't contain an image, iterate through extensions
        if not (width > 0 and height > 0 and bitpix != 0):
            nextend = int(header.get('NEXTEND', '0')) if 'NEXTEND' in header else 0
            found = False
            for _ in range(nextend):
                ext_bytes, ext_header = read_header(f)
                try:
                    width = int(ext_header.get('NAXIS1', '0'))
                    height = int(ext_header.get('NAXIS2', '0'))
                    bitpix = int(ext_header.get('BITPIX', '0'))
                except Exception:
                    width = height = bitpix = 0
                if width > 0 and height > 0 and bitpix != 0:
                    # Found a valid image header
                    header = ext_header
                    found = True
                    break
                else:
                    # Skip any data associated with this extension
                    naxis = int(ext_header.get('NAXIS', '0')) if 'NAXIS' in ext_header else 0
                    dims = [int(ext_header.get(f'NAXIS{i}', '1')) for i in range(1, naxis + 1)]
                    pix_count = 1
                    for d in dims:
                        pix_count *= d
                    # Determine bytes per pixel from BITPIX
                    bp = bitpix
                    if bp == 0:
                        bytes_per = 0
                    elif bp > 0:
                        bytes_per = bp // 8
                    else:
                        bytes_per = (-bp) // 8
                    data_size = pix_count * bytes_per
                    skip_bytes = (data_size + 2879) // 2880 * 2880
                    if skip_bytes:
                        f.seek(skip_bytes, 1)
            if not found:
                return None
        # Determine numpy dtype for big‑endian FITS data
        if bitpix == 8:
            dtype = np.uint8
        elif bitpix == 16:
            dtype = '>i2'
        elif bitpix == 32:
            dtype = '>i4'
        elif bitpix == 64:
            dtype = '>i8'
        elif bitpix == -32:
            dtype = '>f4'
        elif bitpix == -64:
            dtype = '>f8'
        else:
            raise ValueError(f'Unsupported BITPIX value: {bitpix}')
        # Determine dimensions.  Use NAXISn if present; fallback to width/height.
        naxis = int(header.get('NAXIS', '2')) if 'NAXIS' in header else 2
        dims: List[int] = []
        for j in range(1, naxis + 1):
            key = f'NAXIS{j}'
            if key in header:
                dims.append(int(header.get(key, '1')))
            else:
                if j == naxis - 1:
                    dims.append(width or 1)
                elif j == naxis:
                    dims.append(height or 1)
                else:
                    dims.append(1)
        if len(dims) < 2:
            dims = [width, height]
        # Read the pixel data
        total_pixels = 1
        for d in dims:
            total_pixels *= d
        data = np.fromfile(f, dtype=dtype, count=total_pixels)
        if data.size != total_pixels:
            data = data[:total_pixels]
            if data.size < total_pixels:
                data = np.pad(data, (0, total_pixels - data.size), constant_values=0)
        try:
            arr = data.reshape(tuple(dims))
        except Exception:
            arr = data
        while isinstance(arr, np.ndarray) and arr.ndim > 2:
            arr = arr[0]
        image = arr
        if isinstance(image, np.ndarray) and image.ndim == 2:
            h, w = image.shape
            if w == height and h == width:
                image = image.T
        return image

def _compute_stats_and_hist(image) -> Dict[str, Any]:
    """Compute summary statistics and a histogram for a 2D image array.

    Returns a dictionary with min, max, mean, median, std, and a 10‑bin
    histogram of pixel intensities.  The histogram is returned as a list of
    counts corresponding to each bin.
    """
    import numpy as np
    arr = image.astype(float).ravel()
    if arr.size == 0:
        return {
            'min': None,
            'max': None,
            'mean': None,
            'median': None,
            'std': None,
            'histogram': [0] * 10,
        }
    stats = {
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'std': float(np.std(arr)),
    }
    counts, _ = np.histogram(arr, bins=10)
    stats['histogram'] = counts.astype(int).tolist()
    return stats

def create_app() -> Flask:
    """Factory to create and configure the Flask application."""
    app = Flask(__name__, template_folder='templates', static_folder='static')
    # Data directories reside inside the dashboard package
    # Determine where the data directories live.  The dashboard package
    # includes a ``data`` directory for bundled examples, but when
    # downloading new observations via the get_mast_science_and_product.py
    # script the files are stored in a top‑level ``data`` directory.
    data_dir = Path(__file__).resolve().parent / 'data'
    # Look two levels above dashboard (.. = ADPs, ... = project root) for
    # a sibling ``data`` directory.  This covers both layouts.
    root_data_dir = Path(__file__).resolve().parents[3] / 'data'
    # Candidate locations for FITS and rendered files.  The first existing
    # path will be used; otherwise the first entry is chosen by default.
    potential_fits_roots = [
        data_dir / 'fits' / 'mastDownload',
        root_data_dir / 'fits' / 'mastDownload',
    ]
    fits_root = None
    for candidate in potential_fits_roots:
        if candidate.exists():
            fits_root = candidate
            break
    if fits_root is None:
        # Fallback to the first candidate even if it doesn't exist.
        fits_root = potential_fits_roots[0]
    potential_rendered_dirs = [
        data_dir / 'rendered',
        root_data_dir / 'rendered',
    ]
    rendered_dir = None
    for candidate in potential_rendered_dirs:
        if candidate.exists():
            rendered_dir = candidate
            break
    if rendered_dir is None:
        rendered_dir = potential_rendered_dirs[0]

    @app.route('/')
    def home():
        from datetime import datetime
        current_year = datetime.now().year
        nav_items = [{'slug': k, 'title': v['title']} for k, v in RAW_OBJECTS.items()]
        images = []
        if rendered_dir.exists():
            for p in sorted(rendered_dir.iterdir()):
                if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}:
                    title = p.stem.replace('_', ' ').replace('-', ' ')
                    images.append({'filename': p.name, 'title': title})
        return render_template('index.html', images=images,
                               nav_items=nav_items, current_year=current_year)

    @app.route('/rendered/<path:filename>')
    def serve_rendered(filename: str):
        return send_from_directory(rendered_dir, filename)

    @app.route('/raw/<slug>')
    def raw_page(slug: str):
        info = RAW_OBJECTS.get(slug)
        if not info:
            return render_template('error.html', message='Unknown raw dataset'), 404
        nav_items = [{'slug': k, 'title': v['title']} for k, v in RAW_OBJECTS.items()]
        from datetime import datetime
        current_year = datetime.now().year
        return render_template('raw.html', slug=slug, title=info['title'],
                               nav_items=nav_items, current_year=current_year)

    @app.route('/api/raw_data/<slug>')
    def api_raw_data(slug: str):
        import numpy as np
        info = RAW_OBJECTS.get(slug)
        if not info:
            return jsonify({'error': 'Unknown dataset'}), 404
        file_path = fits_root / info['file']
        if not file_path.exists():
            return jsonify({'error': f'FITS file not found for {slug}'}), 404
        try:
            image = _parse_fits_image(file_path)
        except Exception as exc:
            logger.exception("Failed to parse FITS file %s", file_path)
            return jsonify({'error': str(exc)}), 500
        if image is None or not hasattr(image, 'size') or image.size == 0:
            return jsonify({'error': 'No image data found in FITS file'}), 500
        arr = image.astype(float)
        if arr.size == 0:
            return jsonify({'error': 'No pixel data in FITS file'}), 500
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        if max_val > min_val:
            arr = (arr - min_val) / (max_val - min_val) * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        height, width = arr.shape
        data = arr.tolist()
        return jsonify({'width': width, 'height': height, 'data': data})

    @app.route('/api/raw_stats/<slug>')
    def api_raw_stats(slug: str):
        info = RAW_OBJECTS.get(slug)
        if not info:
            return jsonify({'error': 'Unknown dataset'}), 404
        file_path = fits_root / info['file']
        if not file_path.exists():
            return jsonify({'error': f'FITS file not found for {slug}'}), 404
        try:
            image = _parse_fits_image(file_path)
        except Exception as exc:
            logger.exception("Failed to parse FITS file %s", file_path)
            return jsonify({'error': str(exc)}), 500
        if image is None or not hasattr(image, 'size') or image.size == 0:
            return jsonify({'error': 'No image data found in FITS file'}), 500
        stats = _compute_stats_and_hist(image)
        return jsonify(stats)

    return app


if __name__ == '__main__':
    app = create_app()
    host = '0.0.0.0'
    port = 5000
    logger.info("Starting dashboard on %s:%s", host, port)
    app.run(host=host, port=port, debug=True)
