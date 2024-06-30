# Chroma-LXE

Chroma-LXE provides a set of tools for modeling and analyzing the behavior of photons in liquid xenon detectors using [chroma](https://github.com/benland100/chroma) simulation framework. It includes:

- A set of predefined geometries for common liquid xenon detector configurations.
- Tools for defining custom geometries and materials.
- Utilities for visualizing the detector geometry and photon trajectories.
- A collection of Jupyter Notebooks demonstrating how to use the toolkit.

`chroma-lxe` requires a CUDA-enabled GPU to work. To check if your GPU is CUDA-enabled, you can use the [CUDA GPU Checker](https://developer.nvidia.com/cuda-gpus).

It allows for the simulation of complex geometries with arbitrary detector configurations, materials, and surfaces using the [chroma](https://github.com/benland100/chroma) simulation framework, a CUDA-based fast optical propagator with relevant physics processes. From the original repo,

>Chroma is a high performance optical photon simulation for particle physics detectors originally written by A. LaTorre and S. Seibert. It tracks individual photons passing through a triangle-mesh detector geometry, simulating standard physics processes like diffuse and specular reflections, refraction, Rayleigh scattering and absorption.
>
>With the assistance of a CUDA-enabled GPU, Chroma can propagate 2.5 million photons per second in a detector with 29,000 photomultiplier tubes. This is 200x faster than the same simulation with GEANT4.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Repository Structure](#repository-structure)

## Installation

To install Chroma-LXE, clone the repository to wherever you'd like on your machine. For ease of use, it's convenient to clone the repository to somwhere in your home directory.

```bash
git clone https://github.com/youngsm/chroma-lxe.git
```

### Chroma Containers

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

## Setting up `chroma-lxe` for use.



## Repository Structure

- `bin/`: Contains executable scripts.
- `data/`: Stores input data for simulations.
- `geometry/`: Defines geometrical configurations for the simulations.
- `installation/`: Instructions and scripts for setting up the environment.
- `macros/`: Macro files for batch processing.
- `notebooks/`: Jupyter Notebooks demonstrating usage and examples.
- `.gitignore`: Specifies files to be ignored by git.
- `README.md`: This file.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

Ensure you follow the coding standards and write tests for new features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, please open an issue in this repository.

