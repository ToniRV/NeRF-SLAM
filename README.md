<p align="center">
  <h1 align="center">NeRF-SLAM</h1>
  <h1 align="center">
  Real-Time Dense Monocular SLAM with Neural Radiance Fields</h1>
  <p align="center">
    <a href="https://www.mit.edu/~arosinol/"><strong>Antoni Rosinol</strong></a>
    ·
    <a href="https://marinerobotics.mit.edu/"><strong>John J. Leonard</strong></a>
    ·
    <a href="https://web.mit.edu/sparklab/"><strong>Luca Carlone</strong></a>
  </p>
  <!-- <h2 align="center">In Review</h2> -->
  <h3 align="center">
    <a href="https://arxiv.org/abs/2210.13641">Paper</a> |
    <a href="https://www.youtube.com/watch?v=-6ufRJugcEU">Video</a> | 
    <!-- <a href="">Project Page</a>-->
  </h3>
  <div align="center"></div>
</p>
<p align="center">
  <a href="#">
    <img src="./media/intro.gif" alt="" width="90%">
  </a>
</p>
<br>

<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#install">Install</a>
    </li>
    <li>
      <a href="#download-sample-data">Download Datasets</a>
    </li>
    <li>
      <a href="#run">Run</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
    <li>
      <a href="#license">License</a>
    </li>
    <li>
      <a href="#acknowledgments">Acknowledgments</a>
    </li>
    <li>
      <a href="#contact">Contact</a>
    </li>
  </ol>
</details>
<br>

## Install

Clone repo with submodules:
```
git clone https://github.com/ToniRV/NeRF-SLAM.git --recurse-submodules
git submodule update --init --recursive
```

Install torch (see [here](https://pytorch.org/get-started/previous-versions) for other versions):
```
# CUDA 11.3
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

Pip install requirements:
```
pip install -r requirements.txt
```

Compile ngp (you need cmake>3.22):
```
cmake ./thirdparty/instant-ngp -B build
cmake --build build --config RelWithDebInfo -j
```

Install:
```
python setup.py install
```

## Download Sample Data

This will just download one of the replica scenes:
```
./scripts/download_replica_sample.bash
```

## Run

```
python ./examples/slam_demo.py --dataset_dir=./datasets/Replica/office0 --dataset_name=nerf --buffer=100 --slam --parallel_run --img_stride=2 --fusion='nerf' --multi_gpu --gui
```

## Citation

```bibtex
@article{rosinol2022nerf,
  title={NeRF-SLAM: Real-Time Dense Monocular SLAM with Neural Radiance Fields},
  author={Rosinol, Antoni and Leonard, John J and Carlone, Luca},
  journal={arXiv preprint arXiv:2210.13641},
  year={2022}
}
```

## License

This repo is BSD Licensed.
It reimplements parts of Droid-SLAM (BSD Licensed).
Our changes to instant-NGP (Nvidia License) are released in our [fork of instant-ngp](https://github.com/ToniRV/instant-ngp) (branch `feature/nerf_slam`) and 
added here as a thirdparty dependency using git submodules.

## Acknowledgments

This work has been possible thanks to the open-source code from [Droid-SLAM](https://github.com/princeton-vl/DROID-SLAM) and 
[Instant-NGP](https://github.com/NVlabs/instant-ngp), as well as the open-source datasets [Replica](https://github.com/facebookresearch/Replica-Dataset) and [Cube-Diorama](https://github.com/jc211/nerf-cube-diorama-dataset).

## Contact

Feel free to reach out :) [arosinol@mit.edu](arosinol@mit.edu)
