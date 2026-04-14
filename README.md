<div align="center">

# ✨SimRecon: SimReady Compositional Scene Reconstruction from Real Videos✨

<p align="center">
<a href="https://xiac20.github.io/">Chong Xia</a><sup>1,2,*</sup>,
Kai Zhu<sup>1,*</sup>,
Zizhuo Wang<sup>1</sup>,
<a href="https://liuff19.github.io/">Fangfu Liu</a><sup>1</sup>,
<a href="https://scholar.google.com/citations?user=X7M0I8kAAAAJ&hl=en">Zhizheng Zhang</a><sup>2</sup>,
<a href="https://duanyueqi.github.io/">Yueqi Duan</a><sup>1,†</sup>
<br>
<sup>1</sup>Tsinghua University &nbsp;
<sup>2</sup>Galbot
</p>

<h3 align="center">CVPR 2026 (Highlight)🔥</h3>

<a href="https://arxiv.org/abs/2603.02133"><img src='https://img.shields.io/badge/arXiv-2603.02133-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://xiac20.github.io/SimRecon"><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a><img src='https://img.shields.io/badge/License-MIT-blue'></a> &nbsp;&nbsp;&nbsp;&nbsp;

![Teaser Visualization](assets/teaser.png)
</div>

**SimRecon:** We propose SimRecon,a novel compositional scene reconstruction framework that implements a "Perception-Generation-Simulation" pipeline with specialized bridging modules to ensure high visual fidelity and physical plausibility.

## 📢 News
- 🔥 [04/14/2026] ReplicateAnyScene: Our robust, zero-shot follow-up work is now available. See the [project page](https://xiac20.github.io/ReplicateAnyScene/).
- 🔥 [03/18/2026] We release the code of our Scene Graph Synthesizer (SGS) module, making our entire codebase fully open-sourced.
- 🔥 [03/18/2026] To accommodate diverse choices in 3D asset generation models and physical simulators, we leave their specific implementations open for user customization.
- 🔥 [03/03/2026] We release the code of our Active Viewpoint Optimization (AVO) module. 
- 🔥 [03/03/2026] We release "SimRecon: SimReady Compositional Scene Reconstruction from Real Videos". Check our [project page](https://xiac20.github.io/SimRecon) and [arXiv paper](https://arxiv.org/abs/2603.02133).


## 🌟 Pipeline

![Pipeline Visualization](assets/pipeline.png)


<strong>The overall framework of our approach SimRecon.</strong> We propose a “Perception-Generation-Simulation” pipeline with object-centric scene representations towards compositional 3D scene reconstruction from cluttered video input. In this figure, we provide illustrative visualizations using the backpack as the example to introduce our two core modules: Active Viewpoint Optimization (AVO) and Scene Graph Synthesizer (SGS). There, we visualize a semantic-level graph for clarity, while our framework operates at the instance-level.

<!-- ## 🎨 Video Demos

<video width="100%" controls autoplay loop muted>
  <source src="assets/demo.mp4" type="video/mp4">
</video> -->

## ⚙️ Setup

### 1. Clone Repository
```bash
git clone https://github.com/xiac20/SimRecon.git
cd SimRecon
```
### 2. Environment Setup

1. **Create conda environment**

```bash
conda create -n simrecon python=3.9 -y
conda activate simrecon 
```
2. **Install dependencies**
```bash

pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

pip install --extra-index-url=https://pypi.nvidia.com "cudf-cu11==24.2.*" "cuml-cu11==24.2.*"

pip install -r requirements.txt
```

3. **Additional Setup**

Install [`CropFormer`](https://github.com/qqlu/Entity/tree/main/Entityv2/CropFormer) for instance-level segmentation.

```bash
cd semantic_modules/CropFormer
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
cd ../../../../
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git
cd ..
pip install -r requirements.txt
pip install -U openmim
mim install mmcv
pip install transformers
mkdir ckpts
```
Manually
download [CropFormer checkpoint](https://huggingface.co/datasets/qqlu1992/Adobe_EntitySeg/blob/main/CropFormer_model/Entity_Segmentation/CropFormer_hornet_3x/CropFormer_hornet_3x_03823a.pth)
into `semantic_modules/CropFormer/ckpts`

## 💻Data Preprocessing
Please follow the steps below to process your custom dataset, or directly download [our preprocessed datasets](https://drive.google.com/file/d/1SGLEuadiUjyyZLCZKWPEmIbdNTbp68t4/view?usp=sharing).

### 1. Get sparse reconstruction from video input
Follow the original repository from [COLMAP](https://github.com/colmap/colmap) or [HLOC](https://github.com/cvg/Hierarchical-Localization) to get sparse reconstruction results. For scenes with poor image quality and severe occlusion, we strongly recommend using HLOC or other state-of-the-art methods to complete sparse reconstruction.

### 2. Run instance-level segmentation.
```bash
cd semantic_modules/CropFormer
bash run_segmentation.sh "$DATA_DIR"
cd ../..
```

### 3. Training 2DGS.
```bash
python train_2dgs.py -s data/scene0000_00 -m output/scene0000_00
```

Put the trained `point_cloud.ply` file into the `$DATA_DIR` directory. After successfully executing the above steps, the
data directory should be structured as follows:

   ```
   data
      |——————scene0000_00
         |——————point_cloud.ply
         |——————images
            |——————0.jpg
            ...
         |——————sam
            |——————mask
               |——————0.png
               ...
         |——————sparse
            |——————0
               |——————cameras.bin
               ...
   ```

## 💻Run Examples

We provide three example scenes to help you get started.

### 1. Generate 3D semantic segmentation

```bash
python train_semantic.py -s data/scene0000_00 \
                         -m train_semanticgs \
                         --use_seg_feature --iterations 2500 \
                         --load_filter_segmap --consider_negative_labels
```
- It's normal to get stuck at the `DBScan Filter Stage`, since the backgrount gaussian points may be divided into multi-regions.
- Use `--consider_negative_labels` to suppress floaters during background segmentation.
- The generated instance labels are not accurate and just for reference, which would be further determined by the VLM in the SGS module.

### 2. Optimize best view by AVO

```bash
python optimize_by_avo.py --source_path data/scene0000_00 --label_dir output/data/scene0000_00/train_semanticgs/point_cloud/iteration_2500 --max_iterations 100
```
- Use `--instance_id` to specify the optimized object number. If not specified all objects will be optimized.
- Before optimization, you can view the point cloud of each object under output/data/scene0000_00/train_semanticgs/point_cloud/iteration_2500/label_pointclouds.
- For situations where AVO results are not ideal, please adjust some hyperparameters, such as learning rates for rotation and translation, depth constraint coefficients, etc. The number of optimization rounds can also be adjusted up to 5000 rounds.
- For cases where optimization fails due to artifact drift in the 2dgs output point cloud, check out some recent works dedicated to solving such problems.
- The generated instance labels are not accurate and just for reference, which would be further determined by the VLM in the SGS module.

### 3. Generate 3D assets
Users can generate 3D assets by leveraging the optimized best views. For our pipeline we utilized the Rodin model. Users can proceed to their [official website](https://hyper3d.ai/) to generate the assets directly. We also highly recommend using available open-source Image-to-3D models such as [SAM3D](https://github.com/facebookresearch/sam-3d-objects) or [Trellis](https://github.com/microsoft/TRELLIS) to achieve similar results and reproduce the pipeline.

### 4. Generate instance projected frames
```bash
python coverage_sampling.py --scene_path data/scene0000_00 --num_frames 20 --copy_images
```
- This step leverages VGGT to rapidly sample symbol views exhibiting the maximum voxel coverage.
```bash
python project_instances_to_frames.py --source_path data/scene0000_00 --label_dir output/data/scene0000_00/train_semanticgs/point_cloud/iteration_2500 --sampled_images_dir data/scene0000_00/sampled_images
```
- Users can adjust specific hyperparameters within the script such as min_visible_pixels and min_num_points to ensure an appropriate number of instances.

### 5. Infer final scene graphs
```bash
python infer_scene_graph.py  --instance_project_dir output/data/scene0000_00/train_semanticgs/point_cloud/iteration_2500/instance_project  --output_dir output/data/scene0000_00/train_semanticgs/point_cloud/iteration_2500/scene_graphs
```
```bash
python merge_scene_graphs.py --scene_graphs_dir output/data/scene0000_00/train_semanticgs/point_cloud/iteration_2500/scene_graphs
```
- Users can subsequently leverage the inferred scene graphs to ascertain relative object relationships.
- For spatial placement and coordinate alignment users can compute a basic bounding box from the instance 3DGS to estimate the center position and size. Users can then apply [FoundationPose](https://github.com/NVlabs/FoundationPose) to estimate the orientation by aligning the 3D asset with its best view. This algorithmic step is relatively straightforward so users can easily implement this part themselves based on their chosen assets.
- Regarding the final placement within the simulator users can input the assets manually based on their related parameters. The placement follows the layer-by-layer physical simulation activation procedure detailed in our paper to obtain the final physically plausible scenes.


## 🔗Acknowledgement

We are thankful for the following great works when implementing SimRecon:

- [2DGS](https://github.com/hbb1/2d-gaussian-splatting), [InstaScene](https://github.com/zju3dv/InstaScene), [Rodin](https://github.com/CLAY-3D/OpenCLAY?tab=readme-ov-file), [SAM](https://github.com/facebookresearch/segment-anything), [VGGT](https://github.com/facebookresearch/vggt), [Spatial-MLLM](https://github.com/THU-SI/Spatial-MLLM)

## 📚Citation

```bibtex
@misc{xia2026simreconsimreadycompositionalscene,
  title={SimRecon: SimReady Compositional Scene Reconstruction from Real Videos}, 
  author={Chong Xia and Kai Zhu and Zizhuo Wang and Fangfu Liu and Zhizheng Zhang and Yueqi Duan},
  year={2026},
  eprint={2603.02133},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2603.02133}, 
}
```
