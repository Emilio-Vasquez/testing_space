from astroquery.mast import Observations
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Make sure the download directory exists
download_dir = Path("mast_data")
download_dir.mkdir(parents=True, exist_ok=True)

count = Observations.query_criteria_count(
    obs_collection="HST",
    calib_level=3,
    dataproduct_type="image",
    instrument_name="ACS/WFC",
    filters="F606W",
    proposal_id="12062",
    dataRights="PUBLIC"
)

print("Matching observations:", count)

obs = Observations.query_criteria(
    obs_collection="HST",
    calib_level=3,
    dataproduct_type="image",
    instrument_name="ACS/WFC",
    filters="F606W",
    proposal_id="12062",
    dataRights="PUBLIC"
)

print(f"Found {len(obs)} observations")
print(obs[:3])

one_obs = obs[0]

products = Observations.get_product_list(one_obs)
print(f"Found {len(products)} products for this observation")
print(products[:10])

filtered = Observations.filter_products(
    products,
    productType="SCIENCE",
    extension="fits"
)

print(f"Filtered to {len(filtered)} science FITS files")
print(filtered[:10])

manifest = Observations.download_products(
    filtered[:1],
    download_dir=str(download_dir),
    flat=True
)

print(manifest)

fits_path = manifest["Local Path"][0]
print("Downloaded file:", fits_path)

with fits.open(fits_path) as hdul:
    hdul.info()
    data = hdul[1].data if len(hdul) > 1 and hdul[1].data is not None else hdul[0].data

data = np.array(data, dtype=float)
finite = data[np.isfinite(data)]

vmin = np.percentile(finite, 5)
vmax = np.percentile(finite, 99)

plt.figure(figsize=(8, 8))
plt.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
plt.colorbar()
plt.title("HST Science Product")
plt.show()