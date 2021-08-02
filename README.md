# Crop Arable Land Fraction Algorithm

This repository implements the _Crop Arable Land Fraction_ algorithm.


## Testing

The `test-data` directory contains some data that may be used for testing purposes

```
conda env create --file spec-file.txt
conda activate

cd test-data/docker
docker-compose -p calf-test up -d
cd -

# initialize the datacube DB
datacube --config test-data/docker/datacube/datacube.conf --env sandbox system init

# add the products
datacube --config test-data/docker/datacube/datacube.conf --env sandbox product add data/product-definitions/*

# index datasets
datacube --config test-data/docker/datacube/datacube.conf --env sandbox dataset add data/dataset-documents/*

# python -m calf.calf --help

# or launch a jupyter notebook server and use the provided `calf-interactive-testing` notebook
```