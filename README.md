# Chroma-LXE

<p align="center">
<img src="assets/tpcs.png" alt="drawing"/>
</p>

Chroma-LXE provides a set of tools for modeling and analyzing the behavior of photons in liquid xenon detectors using [chroma](https://github.com/benland100/chroma) simulation framework. It includes:

- Scaffolding for defining custom geometries and materials using CAD-derived STL files
- Tools to create, save, and use light maps in the detector
- Utilities for visualizing the detector geometry and photon trajectories.
- Training a [SIREN](https://www.vincentsitzmann.com/siren/)-based neural network to learn the lightmap of the detector.
- A couple of demonstration Jupyter Notebooks.

`chroma-lxe` allows for the simulation of complex geometries with arbitrary detector configurations, materials, and surfaces using the [chroma](https://github.com/benland100/chroma) simulation framework, a CUDA-based fast optical propagator with relevant physics processes. From the original repo,

>Chroma is a high performance optical photon simulation for particle physics detectors originally written by A. LaTorre and S. Seibert. It tracks individual photons passing through a triangle-mesh detector geometry, simulating standard physics processes like diffuse and specular reflections, refraction, Rayleigh scattering and absorption.
>
>With the assistance of a CUDA-enabled GPU, Chroma can propagate 2.5 million photons per second in a detector with 29,000 photomultiplier tubes. This is 200x faster than the same simulation with GEANT4.

`chroma` requires a CUDA-enabled GPU to work. To check if your GPU is CUDA-enabled, you can use the [CUDA GPU Checker](https://developer.nvidia.com/cuda-gpus).

# Table of Contents

1. [Chroma-LXE](#chroma-lxe)
2. [Installation](#installation)
   - [Installing chroma](#installing-chroma)
   - [Setting up `chroma-lxe` for use](#setting-up-chroma-lxe-for-use)
3. [Repository Structure](#repository-structure)
4. [Geometries](#geometries)
   - [Material considerations](#material-considerations)
   - [From CAD to Chroma](#from-cad-to-chroma)
   - [Defining the detector](#defining-the-detector)
5. [Running Simulations and Analyses](#running-simulations-and-analyses)
   - [Macros](#macros)
   - [Input/output](#inputoutput)
   - [Databases (`db`)](#databases-db)
6. [Example usage](#example-usage)
   - [Creating a light map](#creating-a-light-map)
     - [PhotonLib](#photonlib)
     - [Learning the lightmap with a neural network (SIREN)](#learning-the-lightmap-with-a-neural-network-siren)
7. [Contact](#contact)

## Installation

### Installing chroma

`chroma` is easiest to run in a container, where all nontrivial dependencies are already installed and ready to use. You can use [Docker](https://www.docker.com/resources/what-container/) or [Apptainer (formerly Singularity)](https://apptainer.org/docs/user/latest/). Either works, but the Apptainer/Singularity container is the recommended way to run chroma because of its ease of binding directories and GPU synchronization. Either way, you will need to install Docker as the Singularity container is built from a Docker image.

<details><summary>Running with Docker</summary>

Before running, ensure that you have installed the NVIDIA Container Toolkit. You can find instructions [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Ensure everything's working by running `sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi` and checking that the output is as expected.

You can run the pre-built Docker container with the following command:

```bash
sudo docker run --runtime=nvidia --gpus all -v $HOME:$HOME -it youngsm/chroma3:lxe-plib
```

The `-v $HOME:$HOME` flag mounts your home directory inside the container, allowing you to access files on your host machine. The `youngsm/chroma3:lxe-plib` image is the latest version of the container with the necessary dependencies for TPC studies found in chroma-lxe.
</details>

<details><summary><b>(Recommended)</b> Running with Apptainer (Singularity)</summary>

To run the container with Singularity, you will need to install Singularity on your machine and build an image to run. To build the image, run the following command:

```bash
sudo singularity build chroma3.simg docker://youngsm/chroma3:lxe-plib
```
(For Gratta lab members, the container is already built and available on the PC and is found in `/proj/common/sw/chroma3.simg`)

To run the container, use the following command:

```bash
singularity run -B /run/user --nv /path/to/chroma3.simg
```

</details>

### Setting up `chroma-lxe` for use.

Install chroma-lxe by following the instructions below.

```bash
# Clone the repository
git clone https://github.com/youngsm/chroma-lxe.git
# Set up the environment
source chroma-lxe/env.sh
# Add the environment setup to your bashrc (optional)
echo "source $PWD/chroma-lxe/env.sh" >> ~/.bashrc
```

Test that the installation was successful by running the following command and ensuring that it runs without errors:

```bash
python -m geometry.fiber
```

Note that `env.sh` adds the `chroma-lxe` directory to your `PYTHONPATH`. This allows you to import the modules from the repository in your scripts.

## Repository Structure

- [`bin/download_short_tpc_stl.sh`](bin/download_short_tpc_stl.sh): Script to download the STL files for the short TPC from the Gratta lab.
- [`data/`](data/): Stores input data for simulations. Think STL files, lightmap coordinates, light spectra, etc.
- [`geometry/`](geometry/): Defines geometrical configurations for the simulations. 
    - [`builder.py`](geometry/builder.py): Code for building a detector geometry from YAML-based configuration files. `build_detector_from_yaml` is the main function here.
    - [`fiber.py`](geometry/fiber.py): Classes for defining optical fiber sources. 
    - [`materials.py`](geometry/materials.py), [`surfaces.py`](geometry/surfaces.py): Definitions for materials and surfaces. Materials and surfaces used in detector definitions must be defined here.
- [`installation/`](installation/): Contains container definitions for Docker and Singularity that can be used to build the chroma container from scratch.
- [`macros/`](macros/): Python scripts that can do little tasks or run full simulations
    - [`config_from_stl.py`](macros/config_from_stl.py): Macro that generates a template detector definition YAML file from a list of STL files. 
    - [`nphoton_scan.py`](macros/nphoton_scan.py): Macro that creates a light map for a detector configuration by scanning over many positions and simulating photon bombs.
    - [`h5_to_plib.py`](macros/h5_to_plib.py): Macro that converts a HDF5 file outputted by `nphoton_scan.py` to a photonlib file for ease of use. See [`notebooks/hv_lightmap.ipynb`](notebooks/hv_lightmap.ipynb) for an example of how to use photonlib files.
    - [`hv.py`](macros/hv.py): Macro showing fiber optic light source simulation in the high voltage setup at the Gratta lab.
    - [`sample_sim.py`](macros/sample_sim.py): Sample barebones simulation file for you to modify.
    - [`materials_checker.py`](macros/materials_checker.py): Macro for visualizing what chroma will think is `material1` (inner material, yellow) and `material2` (outer material, green) if you were to use a specific STL.
- [`notebooks/`](notebooks/): Jupyter Notebooks demonstrating usage and examples.
    - [`generate_positions.ipynb`](notebooks/generate_positions.ipynb): Demonstrates how to use `trimesh` to generate lightmap positions within a detector.
    - [`hv_lightmap.ipynb`](notebooks/hv_lightmap.ipynb): Demonstrates how to interact with a photonlib file and visualize the lightmap.
    - [`hv_events.ipynb`](notebooks/hv_events.ipynb): Demonstrates how to use the `chroma`-generated ROOT output files to visualize PTE on a 3D mesh.
    - [`segment_electrode.ipynb`](notebooks/segment_electrode.ipynb): Notebook used to segment a single STL file containing an electrode into many smaller STL files to see the position-based response of the detector.
    - [`materials_checker.ipynb`](notebooks/materials_checker.ipynb): Same functionality as `materials_checker.py`, but in a notebook. Uses `trimesh` instead of Chroma to visualize the inside and outside materials.

## Geometries

Chroma uses a geometry defined using double-sided triangles. A triangle's physical properties is fully defined by it's _inside_ material, _outside_ material, and a surface material. 

### Material considerations

The **inside (`material1`) and outside (`material2`) materials** identify the bulk properties of the two media the boundary separates.
* E.g., index of refraction and absorption lengths. See [`geometry/materials.py`](geometry/materials.py).

The **surface material** describes the optical properties of the surface
* E.g., diffuse and specular reflectivity, detection efficiency. See [`geometry/surfaces.py`](geometry/surfaces.py).

<p align="center">
<img src="assets/material_example.jpg" alt="drawing" width="500"/>
</p>

Above is an example of the sort of materials you'd want to use for a spherical pmt + reflector submerged in water (a la SNO), taken from the Chroma whitepaper.

The orientation of a triangle is found by using the right hand rule on the triangle vertices in the order in which they are defined. This normal is defined as the direction of the inside material. 

<p align="center">
<img src="assets/rhr.jpg" alt="drawing" width="200"/>
</p>

This is extremely important when defining the geometry of your detector, as the orientation of the triangles will determine the direction of the inside material. If you switch your inside and outside materials, Chroma might think that your detector volume is solid stainless steel and not liquid xenon.

To check which material Chroma thinks is the inside and outside material, you can use the [`materials_checker.ipynb`](notebooks/materials_checker.ipynb) notebook or [`materials_checker.py`](macros/materials_checker.py) macro. This notebook will show you a visualization of the inside and outside materials based on the orientation of the triangles in a STL file by plotting two copies of the detector, one unchanged in yellow (the inner material) and one "exploded" view in green (the outer material). In most cases, the STL is correctly defined such that the inside material (yellow) is the the solid material (like SS) and the outside material (green) is the detector medium (like LXe). See example from the notebook below:

<p align="center">
<img src="assets/inside.jpg" alt="drawing" height="250"/>
<img src="assets/outside.jpg" alt="drawing" height="250"/>
</p>

From this STL we see that the inside material (yellow) is stainless steel and the outside material (green) would be liquid xenon. So in the detector definition for the part using this STL we'd write

```yaml
...
material:
  material1: steel
  material2: lxe
...
```

### From CAD to Chroma

Chroma-lxe uses STL files to define the geometry of the detector. Unfortunately there's no easy way to automate this, so there's a bit of manual work involved. The basic idea is that we will need to categorize our detector into different parts based on the material and surface properties we want to assign to them.

For example, for a LXeTPC with SiPMs, you'd want to do something like:
- Save the TPC (including flanges, conflats, spools, screws, etc.) as `tpc.stl` with the idea that we'll just set this whole object to be stainless steel with a liquid xenon inside.
- Save the individual SiPM tiles as `sipms_#.stl` with the idea that each stl will be a single channel with ceramic interior, lxe exterior, and a detecting surface.
- Save the ceramic boad that the SiPMs are glued on as `ceramic_board.stl` with the idea that this will be a ceramic material with a lxe exterior and ceramic surface.
- ... and so on.

Below is the process for creating a detector definition a SolidWorks:

1. Open the CAD file in SolidWorks
2. Select the parts you want to save
3. Right click and select `Invert Selection`
4. Right click on any of the inverted selection and select `Hide Components`
5. Click File > Save as > Save as type: STL > Options > Unit: Millimeters > Save all components of an assembly in a single file > OK > Save
6. Repeat steps 2-5 for each part you want to save.

After you have all the STLs you need, you can use the `config_from_stl.py` macro to generate a template detector definition YAML file:

```bash
python -m macros.config_from_stl /path/to/stl_files/*.stl
```

> Note that this macro treats a single STL as a single part. If you want to use multiple STLs as a single part, you will need to manually create a part in the YAML definition and use '*' in the `stl` field to specify multiple STLs. See the `sipm_tiles` part in the [`detector.yaml`](geometry/config/detector.yaml) file for an example of this.

Then, you can finally edit the YAML file to assign the correct materials and surfaces to each part.

### Defining the detector

```yaml
# Example detector definition file

target: vacuum
log: true

parts:
  - name: sipm_tiles
    is_detector: true

    # Note: each file in the path will be loaded as a separate part with its own channel!
    path: "/home/sam/sw/chroma-lxe/data/stl/sipm_channels/*.STL"

    # The scale factor to apply to the STL file
    scale: 1.0

    # The rotation to apply to the STL file
    rotation:
      dir: [0.0, 0.0, 0.0]
      angle: 0.0  # in degrees

    # The translation to apply to the STL file
    translation: [-34.103, -62.519, -34.15]

    # material properties
    material:
      surface: perfect_detector
      material1: ceramic
      material2: lxe
      color: orangered
    ...
```

A YAML file (an easy to use json-like filetype) is used to define the geometry of the detector. The geometry is defined by a list of _parts_, each of which is a separate STL file(s). Each part can has its own material and surface properties, and can be marked as a detector or not. 

Each part can also be translated and rotated in 3D space. The translation is defined by a 3D vector, and the rotation is defined by an axis of rotation and an angle in degrees. If you used the method above to create your STL files, they should be oriented correctly and you shouldn't need to translate or rotate them.

Each part is assigned a material1 (inner material), material2 (outer material), and surface. The material is defined by the `material1` and `material2` fields, and the surface is defined by the `surface` field. The materials and surfaces must be defined in the `geometry/materials.py` and `geometry/surfaces.py` files, respectively. 

Optionally, you can also color the part by specifying a color and alpha in the `color` and `alpha` fields. The color can be specified as a hex code (e.g., #ffffff), a string (e.g., 'red', 'blue', etc.). All matplotlib colors are available. The alpha is a float between 0 and 1, where 0 is fully transparent and 1 is fully opaque. If color and alpha are not specified, the part will be colored grey with an alpha of 0.25.

At the top of the file, you will need to define a target medium. This is the medium that the detector is submerged in. As a precaution, all parts are encapsulated by a bounding box that is filled with the target medium. This is to ensure that all photons are absorbed by the target medium and not lost to the void. If your detector volume's edge is fully opaque (e.g., steel), this won't matter so you can set this to any material (i.e., `vacuum`).

The main function that constructs a Chroma geometry from a definition file is `load_geometry_from_yaml` in `geometry/builder.py`. You can visualize your detector by using `geometry/builder.py`:

```bash
python -m geometry.builder /path/to/detector.yaml
```

This will create a 3D visualization of your detector in a window using Chroma. You can rotate the detector by clicking and dragging, zoom in and out with the scroll wheel, and pan by holding the right mouse button and dragging.


## Running Simulations and Analyses

> These notes are adapted from notes provided by Ben Land.

This repository uses a python-based simulation and analysis framework, [`pyrat`](pyrat), first developed by Ben Land at UPenn. The `pyrat` file is ran in conjunction with a _macro_, which is a python script that defines methods that pyrat will run at different stages of the simulation. Example usage:

```bash
pyrat macros/sample_sim.py --output test_output.root --evalset num_photons 1000
```

This command will run the `sample_sim.py` macro, outputting the results to `test_output.root`, and setting `db.num_photons`, the number of photons in each photon bomb, to 1000 instead of the default value used in `__configure__`.

### Macros

The macro is responsible for setting up the simulation, running the simulation, and analyzing the output. An example macro is found in `macros/sample_sim.py`.

* `__configure__(db)` is called once when the macro is loaded to add or modify fields in the database. This happens after any `--db` packages specified at runtime are loaded, but before any `--set` or `--evalset` options are evaluated. Returns nothing. Optional.

* `__define_geometry__(db)` is called once after `__configure__` and should return a Chroma geometry (pyrat will flatten and build the BVH) if a simulation is to be performed. If the result is None or this method does not exist, pyrat will not run a Chroma simulation, and will assume you are running an analysis over existing data. Optional.

* `__event_generator__(db)` should be a [python generator](https://wiki.python.org/moin/Generators) that yields something Chroma can simulate (`chroma.event.Event`, `chroma.event.Vertex`, or `chroma.event.Photons`) if running a simulation, or anything you want passed to `__process_event__` during an analysis.

* `__simulation_start__(db)` and `__simulation_end__(db)` are called before and after the event loop, which iterates over the event generator and calls `__process_event__` for each event.

* `__process_event__(db,ev)` receives the events from the event generator as they are generated. If a simulation is being performed, these will be Chroma `chroma.event.Event` objects post-simulation.

### Input/output

Macros are allowed to define any form of input/output they desire. It is suggested to use simple datastructures to store analysis results. Chroma defines a ROOT datastructure that stores all relevant Chroma event properties, and should be used for that purpose. Reading is done similarly. To save each event using this datastructure, add the `--output` flag to the pyrat command line arugments.

```bash
pyrat /path/to/macro.py --output /path/to/output.root
```

Similarly, if you already have a ROOT file with Chroma events that you want to re-analyze, you can use the `--input` flag.

```bash
pyrat /path/to/macro.py --input /path/to/input.root
```

### Databases (`db`)

The `database` module contains code to allow a python package (or module) to 
define a database that maps string keys to arbitrary values like a python 
dictionary. 

Each module in the package can define a property `__exports__` which should be
a list of variable names in the module to add to the database. 

Any module can define an `__opt_exports__` function which will be passed a 
dictionary of run-dependent options and can return a dictionary of keys and 
values to add to a database.

A database can be used like a standard python dictionary: `value = db[key]`.
It can also access string keys that are valid python variable names as fields
of the database object: `value = db.key`.

The `data` package contains default pyrat paramters and is self-documenting. 
For instance, see `data.chroma` for parameters that control the `Chroma` 
simulation. Macros can `__configure__` the database to add or modify fields, 
and load additional properties. The pyrat executable defines `--set` and 
`--evalset` options which set strings or python values (i.e., evaluated) to database keys. These are done in the order they are described, so runtime sets take precedence.

## Example usage

### Creating a light map

To create a light map, you need to define a set of positions where you want to simulate photon bombs. You can use the [`generate_positions.ipynb`](notebooks/generate_positions.ipynb) notebook to generate a set of positions within a detector. This notebook uses the `trimesh` package to generate random positions within a detector volume. The notebook will output a numpy array of positions that you can use in the `lightmap.py` macro.

Given a detector definition and a set of positions saved as a numpy array, you can create a light map with 1m isotropic 175 nm photons at each position via:

```bash
pyrat macros/lightmap.py \
    -s positions_path /path/to/positions.npy \
    -s config_file /path/to/detector.yaml \
    -s output_file /path/to/lightmap.h5 \
    -es num_photons 1_000_000 \
    -es wavelength 175
```

> Note that -s (equiv to --set) is used to set a string and -es (equiv to --evalset) is used to set an evaluated string. The evaluated string is evaluated as a python expression.

Your lightmap will be saved as a HDF5 file with the following keys:

- `posX`, `posY`, `posZ`: The x, y, and z positions of the photon bomb.
- `detected`: The total number of photons detected at each position.
- `n`: The total number of photons simulated at each position.
- `pte`: The photon transport efficiency (# det/# sim) at each position.
- `ch_##_detected`: The number of photons detected at each position for channel `##`.
- `ch_##_pte`: The PTE at each position for channel `##`.
- `time_spent`: The time spent at each position in seconds.

#### PhotonLib

[PhotonLib](https://github.com/cider-ml/photonlib) is a nice python package that provides some class structure for handling lightmaps. It was originally used for DUNE, but can be used for any lightmap. You can convert the HDF5 file to a PhotonLib file (just another H5 file) using the `h5_to_plib.py` macro:

```bash
python -m macros.h5_to_plib /path/to/lightmap.h5 /path/to/lightmap.plib
```

With this new file we can now easily access and visualize the lightmap data in python. See the [hv_lightmap.ipynb](notebooks/hv_lightmap.ipynb) notebook for an example of how to use these files.


#### Learning the lightmap with a neural network (SIREN)

Sinusoidal representation networks ([SIREN](https://www.vincentsitzmann.com/siren/)) are neural networks that can be used to learn the lightmap. It's a regular fully connected neural network that maps coordinate positions in $\mathbb{R}^3$ to PTE values for each channel in $\mathbb{R}^{N_\text{detector}}$, but instead of using ReLU or Sigmoid activations, it uses a sinusoid activation. I.e., the model $\Phi$ is constructed as

```math
\Phi(\mathbf{x}) = \textbf{W}_n(\phi_{n-1}\circ\phi_{n-2}\circ\cdots\circ\phi_0)(\textbf{x}) + \textbf{b}_n, \\
\textbf{x}_i \mapsto \phi_i(\textbf{x}_i) = \sin(\textbf{W}_i\textbf{x}_i + \textbf{b}_i),
```

where at the $i^{th}$ layer of the network, an affine transform defined by the weight matrix $\textbf{W}_i$ and bias $\textbf{b}_i$ is applied on the input (coordinates in our case) $\textbf{x}_i$.

This allows the network to learn complex, high frequency functions with fewer parameters. There are some nice properties to this network, like being able to generalize past stochasticity inherent in the light map, not scaling with the number of voxels in a volume (a large pitfall of lookup tables like lightmaps) and being continuous and differentiable.

The [`slar`](https://github.com/cider-ml/siren-lartpc) package (`siren-lartpc`) was created by folks in the ICARUS experiment to learn optical transport of a TPC by training on a voxelized lookup table. The input to the network is a PhotonLib file. All of the code for training and evaluating is in the package, so you can just plug and play. Edit the training configuration file in [`siren/config/train.yaml`](siren/config/train.yaml) to include the path to your PhotonLib file, the number of channels (in `model/network/out_features`), and the number of epochs, batch size, etc. I recommend the batch size being an integer multiple of the voxels, and large enough to fit in the memory of your GPU. Training should be quite quick.

Then run the training script, which will already be in your path after you install the package:

```bash
train-siren.py siren/config/train.yaml --logdir /path/to/logs
```

The PyTorch checkpoints as well as pertinent data (loss, PTE/visibility bias) will be saved in the `logdir` you specify. For more information on this whole process, see the [original paper](https://arxiv.org/pdf/2211.01505) that used SIREN for the ICARUS detector.

## Contact

For any questions, please open an issue in this repository or email me at [youngsam@stanford.edu](mailto:youngsam@stanford.edu). I'm very happy to help.