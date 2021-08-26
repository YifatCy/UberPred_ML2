import geopandas as gpd
nyc = gpd.read_file(gpd.datasets.get_path('nybb'))
print(nyc.head(100))
print(nyc['BoroName'])