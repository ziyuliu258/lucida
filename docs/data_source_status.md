# Data source and asset status

This repository contains the code and experiment description, not the data used by the local run. The frozen local manifest had five contexts and 50 trajectories, comprising 207 supervised expert-action turns and 45 injected-error turns whose output labels were masked.

| Source route | Contexts | Trajectories | Local preparation status |
|---|---:|---:|---|
| MesaTask / 3D-FRONT | 2 | 20 | Prepared |
| FoundationPose | 2 | 20 | Prepared |
| CA-1M / SAM 3D Objects | 1 | 10 | Prepared |

The CA-1M context used four views and masks with a generated SAM 3D mesh. The FoundationPose contexts used released RGB, depth, camera, mask, pose and mesh assets. MesaTask does not provide depth/camera records for these scenes; the local preparation used layout-derived point clouds and a documented fixed display projection. These are implementation details of this experiment, not claims that the sources provide those annotations.

The local source checkouts and generated data are excluded from this repository. The following source versions were present locally:

- [MesaTask](https://github.com/InternRobotics/MesaTask), commit `0e9f0bb8e4f9790b55aac6dfd482dc51c90365b2`
- [ml-cubifyanything](https://github.com/apple/ml-cubifyanything), commit `00e9cb1f9c1b478bf49e08fea4d88e486794cfc1`
- [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects), commit `f91db411c50efee93d8db7aeb323885650f6f722`
- Additional local preparation sources included MoGe, PyTorch3D and utils3d under `data/vendor/`.

Recreating the dataset requires obtaining each upstream dataset, codebase and model checkpoint independently, following its access and license terms, and preparing the expected local directory layout. No data download or redistribution rights are implied by this repository.
