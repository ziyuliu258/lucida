# Local data and assets

No dataset, raw source archive, generated observation, frozen manifest or model asset is tracked in this repository. The experiment used five contexts and 50 trajectories; source routes and protocol details are in [the provenance notes](../docs/data_source_status.md).

Place locally prepared assets under this directory using the layout expected by the selected config and script. The existing experiment configs refer to paths such as `data/overfit50_surface/manifest.json`. Validate the manifest and confirm every referenced file exists before training.

The source datasets and derived assets have separate redistribution terms. Check those terms before publishing any files. `/data/*` is ignored by Git so raw or generated material cannot be added accidentally; this README is the only tracked file under `data/` by default.
