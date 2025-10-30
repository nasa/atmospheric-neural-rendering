# Atmospheric Neural Rendering (AtmoNR)

![](assets/atmonr_instant_ngp.webp)

This is a library and set of scripts to support the usage of neural rendering for atmospheric / cloud tomography in multi-angle satellite data. In other words, this allows the unsupervised 3D reconstruction of clouds in flat, multi-angle satellite imagery.

Current support includes two neural rendering methods: Neural Radiance Fields ([NeRF](https://github.com/bmild/nerf)) and Instant Neural Graphics Primitives ([Instant-NGP](https://github.com/NVlabs/instant-ngp)), as well as one data from one sensor: the Hyper Angular Rainbow Polarimeter ([HARP2](https://pace.oceansciences.org/harp2.htm)) aboard the Plankton, Aerosol, Cloud-ocean Ecosystem ([PACE](https://pace.gsfc.nasa.gov/)) mission.

## Getting Started
Note that this project assumes access to a GPU with at least 32GB of VRAM, because all data is moved to the GPU before training begins to avoid CPU to GPU communication bottlenecks.

1. Clone this repository, then navigate to its top-level directory.
2. Using mamba, create and activate the environment with:
    ```console
    mamba env create -f env.yml && mamba activate atmonr
    ```
3. Follow the [Pytorch instructions](https://pytorch.org/get-started/locally/) to install torch and torchvision with CUDA support.

4. Next, install torchmetrics:
    ```console
    mamba install torchmetrics
    ```

5. Follow the [tiny-cuda-nn instructions](https://github.com/NVlabs/tiny-cuda-nn).

6. To extract from HARP2 data, you'll need a digital elevation map (DEM), which you can download with:
    ```console
    wget https://oceandata.sci.gsfc.nasa.gov/ocssw/static_ancillary/ETOPO1_ocssw.nc -O data/ETOPO1_ocssw.nc
    ```

7. (_Optional_) If you'd like to render videos, you'll need to install the OpenVDB python bindings and the `vdb_render` binary. Follow the [OpenVDB instructions](https://github.com/AcademySoftwareFoundation/openvdb) to build from source, making sure to set `OPENVDB_BUILD_VDB_RENDER` and `OPENVDB_BUILD_PYTHON_MODULE` to `ON`.

    You will also need to install [ffmpeg](https://ffmpeg.org/), which is used to stitch frames into a video.

8. Finally, install AtmoNR:
    ```console
    pip install .
    ```

## Basic Usage

### Getting data

At present, this repository supports the use of HARP2 Level 1B data. You should place these files in `data/HARP2`. If you have an [Earthdata](https://www.earthdata.nasa.gov/) login and have set up an [authentication method](https://earthaccess.readthedocs.io/en/latest/howto/authenticate/), HARP2 files will automatically download using [earthaccess](https://earthaccess.readthedocs.io/en/latest/), as long as you provide a filename that exists.

### Scripts

With all scripts, you can use the `--help` option to see a list of arguments.

#### Train a neural rendering model

The training script will apply a configurable neural rendering pipeline to the provided data. At a minimum, you need a name for the experiment, the path to a configuration file, and the filename of a scene. For example, the following will train Instant-NGP on a HARP2 granule from June 5, 2025, saving the arguments, config, and checkpoints in `data/output/getting_started`:

```console
python scripts/train.py --exp-name getting_started --config-path configs/instant_ngp.json --scene-filename PACE_HARP2.20250605T161356.L1B.V3.nc
```

You can use [TensorBoard](https://www.tensorflow.org/tensorboard) to check on the progress of your experiment. To start a TensorBoard server, run the commands below in a new shell, then navigate to [http://localhost:6006](http://localhost:6006).
```console
mamba activate atmonr
tensorboard --logdir data/tensorboard
```

#### Extract volumetric data

Once you've trained a model on a scene, you can then extract a 3D representation of the volume. At a minimum, you need the name of an experiment that has been trained, the coordinate mode, and the filename for the extracted volume, in netCDF format. The coordinate mode can be either "voxelgrid" or "native". The "voxelgrid" mode will construct a pseudo-grid on the ellipsoid surface with a nearly-constant horizontal distance and constant vertical distance between the centers of adjacent cells. The horizontal distance and vertical bins are configurable. This allows maximal usage of the spatial resolution of the data. The "native" mode for HARP2 will use (downloading if not locally available) the matching level 1C file, using the coarser level 1C grid. This mode is useful when comparing with other PACE products which use this grid.

```console
python scripts/extract.py --exp-name getting_started --coord-mode voxelgrid --extract-filename my_first_extract.nc
```

The netCDF file with the extracted data will be found in `data/output/getting_started/my_first_extract.nc`. The format is self-describing, with similar dimensions and naming conventions to HARP2 data, although depending on the `coord-mode` the horizontal grid may be different.

#### Render a video preview of the extracted data

After extracting volumetric data from a scene, if you've installed OpenVDB with the python bindings and rendering support (see [Getting Started](#getting-started)), you can export your extract to the VDB format and render a video of the extracted volume. At a minimum, you need the path to the extract file, a path for the exported VDB file, a path for the video, and an index of which band you'd like to render the volume for. For HARP2 data, this should probably be 2, since the red band has the most view angles. This should take a few minutes at the default resolution and frame rate.

```console
python scripts/make_video.py --extract-filepath data/output/getting_started/extract.nc --vdb-filepath data/output/getting_started/my_first_export.vdb --video-filepath data/output/getting_started/my_first_render.mp4
```


## Advanced Usage

You might want to use AtmoNR as a library. It solves many challenges with applying neural rendering to large-scale satellite datasets, including the fundamentally different representations of sensor geometry and vast differences in scale. Here are some of the challenges this repository addresses:
- Prioritizing the quality of the extracted density field over the reconstruction quality / rendering quality
- Treating density (the extinction coefficient) as wavelength-dependent to be more physically realistic
- Ingesting multi-band, multi-angle image cubes rather than individual, registered, calibrated RGB images with pinhole camera geometry
- Reconciling geospatial coordinates with the Cartesian coordinates typical in computer graphics
- Performing coordinate transformations and geodesic distance computations in GPU memory to greatly improve speed
- The relative thinness of the atmosphere at large spatial scales causes numerical and optimization problems, which is addressed by allowing the use horizontal coordinates as inputs to the neural rendering method

Check out the scripts for examples of how to interact with the API.

## Contributing

AtmoNR was designed with generalizability in mind, and we welcome pull requests that respect our contribution guidelines.

### General principles

- Avoid the addition of unnecessary dependencies.
- Machine learning code should be implemented in Pytorch if possible.
- Performance is important. You can use the Pytorch profiler with `train.py`'s `--profile` option to diagnose bottlenecks.
- We use [Ruff](https://docs.astral.sh/ruff/) to check our code; run `ruff format` and `ruff check` before making a pull request.
- Keep things modular: new datasets should work for all pipelines, new pipelines should work for all dataset, etc.
- Document thoroughly and update this README if needed.
- Before adding a feature, check that it isn't already covered by the various utilities.

### Adding a new dataset

Datasets are responsible for preprocessing the satellite data, moving it to GPU memory, and getting items and batches of data during training. ExtractDatasets are responsible for defining the sampling scheme during volume extraction, getting items and batches of data, and writing the data to file. You should implement both a Dataset and an ExtractDataset. Both are wrapped by BatchLoader, which quickly iterates over batches of indices, leveraging the ability to fit an entire scene onto a GPU at once.

Make sure to update the Dataset and ExtractDataset type definitions and dictionaries at the top of [src/atmonr/datasets/factory.py](src/atmonr/datasets/factory.py). This will allow Ruff to complain if any methods required by the scripts are unimplemented.

### Adding a new neural rendering method

Pipelines represent neural rendering methods. They are responsible for applying a method (including any point processing or encoding) to a collection of viewing rays, including the rendering step. Pipeline is responsible for maintaining the state of the rendering method (e.g. the networks in NeRF) and for defining the optimizer.

To add a new Pipeline, inherit `atmonr.pipelines.pipeline.Pipeline` and implement all functions. Make sure to update the dictionary at the top of [src/atmonr/pipelines/factory.py](src/atmonr/datasets/factory.py).

### Unit Tests

Work in progress, come back soon!

## Citation

Work in progress, come back soon!

# Notices
The copyright notice below, to be included in the software, has also been provided in the license. 
 
“Copyright © 2023 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.”
 
No other release is authorized at this time.