**MASt3R-SLAM Complete Local Setup Guide (Ubuntu/Linux Workstation or Normal GPU Server)**

**Tested Requirements**

-   Ubuntu 22.04 (recommended)

-   Python 3.11

-   CUDA 12.x

-   NVIDIA GPU (≥12 GB VRAM recommended)

-   Git

-   CMake

-   GCC/G++

**Step 1. Install System Packages**

sudo apt update

sudo apt install -y \\

git \\

cmake \\

build-essential \\

gcc \\

g++ \\

python3.11 \\

python3.11-venv \\

python3.11-dev \\

ffmpeg \\

libgl1 \\

libglib2.0-0 \\

libegl1-mesa-dev \\

libglfw3 \\

libglfw3-dev \\

libopencv-dev

**Step 2. Install CUDA (Skip if already installed)**

Verify

nvidia-smi

nvcc \--version

**Step 3. Clone Repository**

mkdir \~/master_slam

cd \~/master_slam

git clone \--recursive https://github.com/rmurai0610/MASt3R-SLAM.git official_mast3r

cd official_mast3r

Verify

git rev-parse HEAD

git submodule status

**Step 4. Create Python Environment**

python3.11 -m venv master_env

source master_env/bin/activate

python -m pip install \--upgrade pip setuptools wheel

**Step 5. Install PyTorch**

CUDA 12.1

pip install torch torchvision torchaudio \\

\--index-url https://download.pytorch.org/whl/cu121

Verify

python -c \"import torch;print(torch.cuda.is_available())\"

**Step 6. Install Requirements**

pip install -r requirements.txt

**Step 7. Install MASt3R**

cd thirdparty/mast3r

pip install \--no-build-isolation -e .

cd ../..

**Step 8. Build curope**

cd thirdparty/mast3r/dust3r/croco/models/curope

python setup.py build_ext \--inplace

cd ../../../../../

**Step 9. Install in3d**

cd thirdparty/in3d

pip install \--no-build-isolation -e .

cd ../..

**Step 10. Install Remaining Packages**

pip install \\

moderngl \\

moderngl-window \\

glfw \\

pyglm \\

imgui \\

pyglet \\

open3d \\

trimesh \\

einops \\

roma \\

tensorboard \\

huggingface-hub \\

opencv-python \\

scikit-learn \\

natsort \\

pyyaml

**Step 11. Install MASt3R-SLAM**

pip install \--no-build-isolation -e .

**Step 12. Export PYTHONPATH**

export PYTHONPATH=\$PWD/thirdparty/mast3r:\$PYTHONPATH

(Optional)

echo \'export PYTHONPATH=\$HOME/master_slam/official_mast3r/thirdparty/mast3r:\$PYTHONPATH\' \>\> \~/.bashrc

source \~/.bashrc

**Step 13. Download Checkpoints**

Create folder

mkdir checkpoints

Download

MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth

MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth

MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl

Place all three files inside

official_mast3r/checkpoints/

**Step 14. Verify Installation**

python - \<\<\'PY\'

import torch

import mast3r

import dust3r

import mast3r_slam

print(torch.cuda.is_available())

print(\"Everything OK\")

PY

**Step 15. (Optional) Fix pyrealsense2**

If **not using Intel RealSense**, edit

mast3r_slam/dataloader.py

Replace

import pyrealsense2 as rs

with

try:

import pyrealsense2 as rs

except ImportError:

rs=None

Inside

class RealsenseDataset

before

super().\_\_init\_\_()

add

if rs is None:

raise ImportError(

\"pyrealsense2 is required only when using RealsenseDataset.\"

)

**Step 16. Prepare Dataset**

Example

\~/master_slam/test.mp4

**Step 17. Run MASt3R-SLAM**

**With GUI**

python main.py \\

\--dataset \~/master_slam/test.mp4 \\

\--config config/base.yaml \\

\--save-as first_run

**Headless**

python main.py \\

\--dataset \~/master_slam/test.mp4 \\

\--config config/base.yaml \\

\--save-as first_run \\

\--no-viz

**Step 18. Output**

Results are saved in

logs/first_run/

Find outputs

find logs/first_run

Point cloud

find logs/first_run -name \"\*.ply\"

Trajectory

find logs/first_run -name \"\*.txt\"

Images

find logs/first_run -name \"\*.png\"

**Step 19. View Point Cloud**

**CloudCompare**

File → Open → output.ply

**MeshLab**

File → Import Mesh

**Open3D**

import open3d as o3d

pc=o3d.io.read_point_cloud(\"output.ply\")

o3d.visualization.draw_geometries(\[pc\])

**Step 20. Daily Startup**

cd \~/master_slam/official_mast3r

source master_env/bin/activate

export PYTHONPATH=\$PWD/thirdparty/mast3r:\$PYTHONPATH

Run

python main.py \\

\--dataset \~/master_slam/test.mp4 \\

\--config config/base.yaml \\

\--save-as first_run

**Verification Commands**

GPU

nvidia-smi

CUDA

python -c \"import torch;print(torch.cuda.is_available())\"

MASt3R

python -c \"import mast3r\"

DUSt3R

python -c \"import dust3r\"

MASt3R-SLAM

python -c \"import mast3r_slam\"

**Common Errors**

**ModuleNotFoundError: dust3r**

export PYTHONPATH=\$PWD/thirdparty/mast3r:\$PYTHONPATH

**AssertionError: retrieval_codebook.pkl**

Ensure these three files exist in checkpoints/:

MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth

MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth

MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl

**torch.cuda.is_available() == False**

-   Check NVIDIA driver (nvidia-smi)

-   Verify CUDA installation

-   Install the correct CUDA-enabled PyTorch build

**No .ply generated**

find logs/first_run

Verify that the run completed successfully and inspect the output directory for generated artifacts.

**Recommended Directory Structure**

master_slam/

│

├── official_mast3r/

│ ├── checkpoints/

│ ├── config/

│ ├── logs/

│ ├── mast3r_slam/

│ ├── thirdparty/

│ ├── master_env/

│ └── main.py

│

├── datasets/

│ ├── room1.mp4

│ ├── room2.mp4

│ └── \...

│

└── outputs/

This provides a concise, end-to-end setup for a local Linux workstation or a standard GPU server, without the HPC-specific steps (Slurm, modules, scratch storage, etc.).
