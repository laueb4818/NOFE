import ee
import geedim  # enables .gd interface
import os

if not os.path.exists('data/'):
    os.makedirs('data/')
os.chdir('data/')

# Authenticate & initialize
ee.Initialize()  # or ee.Authenticate(); ee.Initialize(...)
  
# Parameters
years = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
for year in years:

# bbox for a small region: lat/lon rectangle covering small area
    region = ee.Geometry.Rectangle([10.0, 59.0, 11.0, 60.5]) # [10.0, 59.0, 11.0, 60.5])  # example over part of Europe
    output_path = f"sat_emb_{year}.tif"

# Load the embedding collection and filter for year + region
    emb = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    start = ee.Date.fromYMD(year, 1, 1)
    end   = start.advance(1, 'year')
    filtered = emb \
    .filterDate(start, end) \
    .filterBounds(region)

# Merge tiles if needed
    mosaic = filtered.mosaic()

# Prepare and export
    prep = mosaic.gd.prepareForExport(
    region=region,
    scale=10,         # 10 m resolution as doc says :contentReference[oaicite:6]{index=6}
    crs="EPSG:4326",  # or use the UTM projection of each tile
    dtype="float32"   # embeddings go from -1..1
    )

# Download to GeoTIFF
    prep.gd.toGeoTIFF(output_path)

    print("Saved embedding to", output_path)

