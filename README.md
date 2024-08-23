# benchmarking-unsupervised-online-IDS-masquerade-attacks

This repository provides the Python code to reproduce the results of our paper [Benchmarking Unsupervised Online IDS for Masquerade Attacks in CAN (arXiv:2406.13778 [cs.CR])](https://arxiv.org/abs/2406.13778). In this paper, we introduce a benchmark study of four different non-deep learning (DL)-based unsupervised online intrusion detection systems (IDS) for masquerade attacks in CAN. Our approach differs from existing benchmarks in that we analyze the effect of controlling streaming data conditions in a sliding window setting. We show that although benchmarked IDS are not effective at detecting every attack type, the method that relies on detecting changes at the hierarchical structure of clusters of time series produces the best results at the expense of higher computational overhead.

![Benchmark Workflow](figs/benchmark-workflow.pdf)

## Install Miniconda

```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```

## Clone Repo

```
git clone https://github.com/pmoriano/benchmarking-unsupervised-online-IDS-masquerade-attacks.git
cd ids-benchmark
```

## Setup Environment

```
conda env create -f environment.yaml
```

## Download ROAD Dataset

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10462796.svg)](https://doi.org/10.5281/zenodo.10462796)

## Running Code

```
python streaming-detector-distribution-ROAD.py distribution ROAD
python streaming-detector-correlation-ROAD.py correlation ROAD
python streaming-detector-DBSCAN-ROAD.py DBSCAN ROAD
python streaming-detector-AHC-ROAD.py AHC ROAD
```

## Figure Generation

Run the jupyter notebooks in `notebooks` folder

## Citation

```bibtex
@misc{Moriano:2024:Benchmark:Masquerade:Online:IDS,
      title={Benchmarking Unsupervised Online IDS for Masquerade Attacks in CAN}, 
      author={Pablo Moriano and Steven C. Hespeler and Mingyan Li and Robert A. Bridges},
      year={2024},
      eprint={2406.13778},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2406.13778}, 
}
```

```bibtex
@article{Verma:2024:ROAD,
    doi = {10.1371/journal.pone.0296879},
    author = {Verma, Miki E. AND Bridges, Robert A. AND Iannacone, Michael D. AND Hollifield, Samuel C. AND Moriano, Pablo AND Hespeler, Steven C. AND Kay, Bill AND Combs, Frank L.},
    journal = {PLOS ONE},
    publisher = {Public Library of Science},
    title = {A comprehensive guide to CAN IDS data and introduction of the ROAD dataset},
    year = {2024},
    month = {01},
    volume = {19},
    url = {https://doi.org/10.1371/journal.pone.0296879},
    pages = {1-32},
    number = {1},
}
```


