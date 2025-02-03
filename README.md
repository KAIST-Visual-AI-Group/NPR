# Neural Pose Representation (NPR)

![teaser](media/teaser.png)

[Project Page](https://neural-pose.github.io) | [arXiv](https://arxiv.org/abs/2406.09728)

[Seungwoo Yoo](https://dvelopery0115.github.io), [Juil Koo](https://63days.github.io), [Kyeongmin Yeo](https://github.com/32V/), [Minhyuk Sung](https://mhsung.github.io)

[KAIST](https://www.kaist.ac.kr/en/)

This is the reference implementation of **Neural Pose Representation Learning for Generating and Transferring Non-Rigid Object Poses (NeurIPS 2024)**.

## Get Started

Clone the repository and create a Python environment:
```
git clone https://github.com/KAIST-Visual-AI-Group/NPR
cd NPR
conda create --name npr python=3.10
conda activate npr
```

Install the required packages by running:
```
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
pip install cholespy fpsample imageio[ffmpeg] libigl PyYAML tensorboard trimesh tqdm
pip install jaxtyping==0.2.24 typeguard==2.13.3 tyro
pip install scipy==1.11.4  # If libigl fails to compute gradient correctly
```
You may need to adjust the PyTorch version along with the corresponding CUDA version.
Our code has also been tested and verified to work with PyTorch 2.0.1 compiled with CUDA 11.7.

Our model also requires the CUDA implementation of farthest point sampling (FPS) by [Erik Wijmans](https://github.com/erikwijmans/Pointnet2_PyTorch). You can install it by running:
```
pip install pointnet2_ops_lib/.
```

> [!NOTE]
> If you are using CUDA 12 or later, update the CUDA architecture list in `pointnet2_ops_lib/setup.py`. Specifically, change the line`os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"` to `os.environ["TORCH_CUDA_ARCH_LIST"] = "5.0;6.0;6.1;7.0;7.5;8.0;8.6;8.7;9.0"`.

We provide sample data to test our codes via [Google Drive](https://drive.google.com/drive/folders/1W3PTL1Ts0jAV31mzCib6gPHFOz9NTqhw?usp=drive_link). Specifically, you can download
- [SMPL Pose Examples](https://drive.google.com/file/d/1Bw09JSxkkHihUOI-n40Pev1M6-9KJ5ZU/view?usp=drive_link)
- [Mixamo Template Meshes](https://drive.google.com/file/d/13FVoiOCpxDmCoFUQA51tNNfl6G6XCpqv/view?usp=drive_link)

The provided pretrained weights are the following:
| Dataset Name | Template Name | Link |
|------|------|------|
| SMPL | Default Human | [Download](https://drive.google.com/file/d/1VHJkKj5LCefDYlVLFhNjEYN3ufg7tdSN/view?usp=drive_link) |

Download the files from the link and place them under the directory `../data`. The config files under `configs` directory assumes that the data files are arranged in this manner. You may need to modify the paths within config files if you have placed the data at different locations. After that, the directory structure should look like:
```
data
├── mixamo_models  # Mixamo template meshes
├── smpl_default  # SMPL pose examples
└── ...
NPR
├── configs
├── media
├── scripts
├── src
├── environment.yaml
└── README.md  # This file
````

## Running Demo

Our demo script `scripts/test.py` requires an experiment directory containing `checkpoints` subdirectory. To begin, download the checkpoint file from the link above and set up the experiment directory as follows:
```
mkdir demo_exp
mkdir demo_exp/checkpoints
mv ${PATH TO CHECKPOINT FILE} demo_exp/checkpoints/
```

Then, run the demo script:
```
python scripts/test.py \
    --train-dset-cfg-file configs/human_body/smpl_default.json \
    --test-dset-cfg-file configs/human_body/Ch24_nonPBR_normalized.json \
    --exp-dir demo_exp
```
Ensure that you provide `configs/human_body/smpl_default.json` to the `--train-dset-cfg-file` argument, as the provided checkpoint was trained using pose examples from this configuration file. The file specified in the `--test-dset-cfg-file` argument defines the template mesh onto which the poses will be transferred.

## Training

Use the script `scripts/train.py` to train the model. To train the model using the provided SMPL pose examples, run the following command:
```
python scripts/train.py \
    --train-dset-cfg-file configs/human_body/smpl_default.json \
    --test-dset-cfg-file configs/human_body/smpl_default.json \
    --out-dir train_exp
```

## Citation
Please consider citing our work if you find this codebase useful:
```
@inproceedings{yoo2024neuralpose,
    title = {{Neural Pose Representation Learning for Generating and Transferring Non-Rigid Object Poses}},
    author = {Yoo, Seungwoo and Koo, Juil and Yeo, Kyeongmin and Sung, Minhyuk},
    booktitle = {NeurIPS},
    year = {2024},
    }
```
