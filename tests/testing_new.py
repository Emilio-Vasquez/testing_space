from astroquery.mast import Observations
from astropy.table import vstack
from pathlib import Path
from urllib.parse import unquote
import re

download_dir = Path("data/spectra")
download_dir.mkdir(parents=True, exist_ok=True)

obs = Observations.query_object("M8", radius=".02 deg")

mask = (
    (obs["obs_collection"] == "HST") &
    (obs["dataproduct_type"] == "spectrum")
)
obs = obs[mask]

print(f"HST spectrum observations near M8: {len(obs)}")
print(obs[:5])

product_tables = []
for row in obs[:5]:
    try:
        p = Observations.get_product_list(row)
        if len(p) > 0:
            product_tables.append(p)
    except Exception as e:
        print("Could not get products for one observation:", e)

if not product_tables:
    print("No product tables found.")
else:
    all_products = vstack(product_tables)

    filtered = Observations.filter_products(
        all_products,
        productType="SCIENCE",
        extension="fits"
    )

    print(f"Spectrum FITS products: {len(filtered)}")
    print(filtered[:10])

    def safe_filename_from_uri(uri, fallback_prefix="file"):
        uri = unquote(str(uri))

        # Case 1: HLA-style URI with ?dataset=actual_name.fits
        match = re.search(r"dataset=([^&]+)", uri)
        if match:
            return match.group(1)

        # Case 2: normal mast URI ending in a filename
        name = Path(uri).name
        if name and "?" not in name and name.strip():
            return name

        # Case 3: last resort
        cleaned = re.sub(r'[^A-Za-z0-9._-]', '_', name)
        if cleaned:
            return cleaned

        return f"{fallback_prefix}.fits"

    for i, prod in enumerate(filtered[:10]):
        try:
            uri = prod["dataURI"]
            fname = safe_filename_from_uri(uri, fallback_prefix=f"spectrum_{i}")
            filename = download_dir / fname

            result = Observations.download_file(uri, local_path=str(filename))
            print(i, result, filename)
        except Exception as e:
            print(f"Failed on item {i}: {e}")