from astroquery.mast import Observations
from astropy.table import vstack
from pathlib import Path

download_dir = Path("data/fits")
download_dir.mkdir(parents=True, exist_ok=True)

obs = Observations.query_criteria(
    obs_collection="HST",
    calib_level=3,
    dataproduct_type="image",
    instrument_name="ACS/WFC",
    filters="F606W",
    proposal_id="12062",
    dataRights="PUBLIC"
)

obs_subset = obs[:5]

product_tables = []
for row in obs_subset:
    p = Observations.get_product_list(row)
    if len(p) > 0:
        product_tables.append(p)

all_products = vstack(product_tables)

fits_products = Observations.filter_products(
    all_products,
    productType="SCIENCE",
    extension="fits"
)

print(f"FITS products found: {len(fits_products)}")

for i, prod in enumerate(fits_products[:10]):
    try:
        uri = prod["dataURI"]
        filename = download_dir / Path(uri).name
        result = Observations.download_file(uri, local_path=str(filename))
        print(i, result, filename)
    except Exception as e:
        print(f"Failed on item {i}: {e}")