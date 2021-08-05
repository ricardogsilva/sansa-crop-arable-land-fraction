from setuptools import setup

setup(
    name='calf',
    url='',
    license='',
    author='Ricardo Garcia Silva',
    author_email='ricardo@kartoza.com',
    description='SANSA Crop Arable Land Fraction algorithm',
    version='0.1.0',
    packages=['calf'],
    install_requires=[
        "datacube",
        "geopandas",
        "numpy",
        "pandas",
        "rasterio",
        "rioxarray",
        "tabulate",
        "typer",
        "xarray",
    ],
    entry_points={
        "console_scripts": ["sansa-calf=calf.cli:app"]
    },
)
