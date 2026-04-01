from pathlib import Path
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

spectra_dir = Path("data/spectra")
output_dir = Path("data/spectra_output")
output_dir.mkdir(parents=True, exist_ok=True)

fits_files = sorted(spectra_dir.glob("*.fits"))

if not fits_files:
    print("No FITS files found.")
    raise SystemExit

print(f"Found {len(fits_files)} FITS files")

for fits_path in fits_files:
    print(f"\nProcessing: {fits_path.name}")

    try:
        with fits.open(fits_path) as hdul:
            image_found = False

            for i, hdu in enumerate(hdul):
                data = getattr(hdu, "data", None)

                if data is None:
                    continue

                arr = np.array(data)

                # Look for 2D image-like data
                if arr.ndim == 2:
                    arr = arr.astype(float)
                    finite = arr[np.isfinite(arr)]

                    if finite.size == 0:
                        continue

                    vmin = np.percentile(finite, 5)
                    vmax = np.percentile(finite, 99)

                    stem = fits_path.stem

                    # Original image
                    plt.figure(figsize=(8, 8))
                    plt.imshow(arr, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
                    plt.title(f"{stem} - HDU {i}")
                    plt.axis("off")
                    plt.tight_layout()
                    plt.savefig(output_dir / f"{stem}_hdu{i}_original.png", dpi=150)
                    plt.close()

                    # Tampered image
                    tampered = arr.copy()
                    h, w = tampered.shape
                    r1, r2 = h // 3, h // 3 + max(10, h // 12)
                    c1, c2 = w // 3, w // 3 + max(10, w // 12)
                    tampered[r1:r2, c1:c2] += np.nanmax(finite) * 0.25

                    plt.figure(figsize=(8, 8))
                    plt.imshow(tampered, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
                    plt.title(f"{stem} - Tampered")
                    plt.axis("off")
                    plt.tight_layout()
                    plt.savefig(output_dir / f"{stem}_hdu{i}_tampered.png", dpi=150)
                    plt.close()

                    # Difference image
                    diff = tampered - arr
                    diff_finite = diff[np.isfinite(diff)]
                    dv = np.max(np.abs(diff_finite)) if diff_finite.size else 1

                    plt.figure(figsize=(8, 8))
                    plt.imshow(diff, origin="lower", cmap="gray", vmin=-dv, vmax=dv)
                    plt.title(f"{stem} - Difference")
                    plt.axis("off")
                    plt.tight_layout()
                    plt.savefig(output_dir / f"{stem}_hdu{i}_difference.png", dpi=150)
                    plt.close()

                    threshold = np.nanstd(diff) * 5
                    alert = np.nanmax(np.abs(diff)) > threshold

                    print(f"Saved outputs for HDU {i}")
                    print("ALERT:" if alert else "OK:", "Possible tampering detected" if alert else "No major anomaly detected")

                    image_found = True
                    break

            if not image_found:
                print("No usable 2D image data found")

    except Exception as e:
        print(f"Failed on {fits_path.name}: {e}")