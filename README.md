## [Tightening Classification Boundaries in Open Set Domain Adaptation through Unknown Exploitation (SIBGRAPI'2023)](http://sibgrapi.sid.inpe.br/sid.inpe.br/sibgrapi/2023/08.18.21.55)

The source code of honor mentioned work of the SIBGRAPI'2023 conference entitled "Tightening Classification Boundaries in Open Set Domain Adaptation through Unknown Exploitation". In this work, we proposed a deep learning approach that addresses the simultaneous challenges of domain-shift and category-shift in unsupervised datasets, *i.e.*, the Open Set Domain Adaptation problem (OSDA). For this, three different methods of utilizing unknown examples to refine the classification boundaries of the deep learning classifier were investigated, leading to absolute gains of up to **5.8% in accuracy**.

All implementations were made upon the original code of [OVANet](https://github.com/VisionLearningGroup/OVANet). Notably, we (1) limited the span of their UNDA method to OSDA; (2) updated the source code to use a newer version of PyTorch; and (3) modified their source code with as minimum changes as possible to implement all of our unknown directed investigations.

### Reproducibility

The following software and library versions were used during our experimental procedure.

```
  Python=3.10.6
  PyTorch=1.13.1
  TorchVision=0.14.1
  Scikit-Learn=1.2.2
  Matplotlib=3.8.2
  pyYAML=6.0
  Pillow=9.4.0
  easydict=1.10
```

  We made available a Dockerfile that can be built and promptly used to reproduce the environment. For this, follow the next instructions:

1. Download the datasets [Office-31](https://faculty.cc.gatech.edu/~judy/domainadapt/#datasets_code) and [Office-Home](http://hemanthdv.org/OfficeHome-Dataset/);
2. Create a parent folder named `data` and displace each of the internal domain folders as following:
    ```
    data/-|
          |- Art
          |- Clipart
          |- Product
          |- Real
          |- amazon/images/...
          |- dslr/images/...
          |- webcam/images/...
    ```
3. Download the [dataset partitioning files](https://drive.google.com/file/d/1j_PT-gRWQQNkbwcWBuNc01D7QtCBYomN/view?usp=sharing) and unzip them;
4. Go to the docker folder and build the docker image with the following command:
    ```
    docker build --build-arg DATASET_PATH=/path/to/data/ --build-arg TXT_PATH=/path/to/txt/ -t unke_ovanet:22.04 .
    ```
5. Begin the container image with the following command:
    ```
    docker run --gpus all --name unke-ovanet --rm -it -w /home/UnkE-OVANet/ unke_ovanet:22.04
    ```
6. You can choose between one of our three different approaches to deal with the unknown by modifying the 90th line of `train.py` code (**"original"**, **"augment"** or **"generate"** options).
7. Run the desired scripts: `./scripts/run_office_obda.sh train.py <GPU-ID>` for Office-31 experiments or `./scripts/run_officehome_obda.sh train.py <GPU-ID>` for Office-Home experimentation.

> NOTE 1: If you do not want to use containers, manually create the environment, displace the `data` folder and `txt` folder inside the cloned repository folder UnkE-OVANet, and follow the 7th step to run the experiments.

>NOTE 2: In our case, running the script inside the container gave an `OSError: No space left on device`. if you come across this, simply add the following command line option in step (5) `--shm-size=512m`. 
### Contact

If you have any doubts, feel free to contact me through the email lucas.silva@ic.unicamp.br

### Citation

```
@InProceedings{SIBGRAPI 2023 Silva,
    author = {L. F. A. {Silva} and N. {Sebe}  and J. {Almeida}},
    title = {Tightening Classification Boundaries in Open Set Domain Adaptation through Unknown Exploitation},
    pages = {1–6},
    booktitle = {2023 36th SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI)},
    address = {Rio Grande, RS, Brazil},
    month = {November 6–9},
    year = {2023},
    publisher = {{IEEE}}
}
```


