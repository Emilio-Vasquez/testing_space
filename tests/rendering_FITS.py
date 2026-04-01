from pathlib import Path
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

fits_dir = Path("data/fits")
render_dir = Path("data/rendered")
render_dir.mkdir(parents=True, exist_ok=True)

fits_files = list(fits_dir.glob("*.fits"))

for fits_path in fits_files:
    try:
        with fits.open(fits_path) as hdul:
            data = None

            for hdu in hdul:
                if getattr(hdu, "data", None) is not None:
                    arr = np.array(hdu.data)
                    if arr.ndim >= 2:
                        data = arr
                        break

            if data is None:
                print(f"Skipping {fits_path.name}: no image-like data found")
                continue

        data = np.array(data, dtype=float)
        finite = data[np.isfinite(data)]

        if finite.size == 0:
            print(f"Skipping {fits_path.name}: no finite values")
            continue

        vmin = np.percentile(finite, 5)
        vmax = np.percentile(finite, 99)

        out_path = render_dir / (fits_path.stem + ".png")

        plt.figure(figsize=(8, 8))
        plt.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0)
        plt.close()

        print(f"Rendered {fits_path.name} -> {out_path.name}")

    except Exception as e:
        print(f"Failed to render {fits_path.name}: {e}")