# Model checkpoints

The supplied archive did not contain the actual model binaries. The files named `model.pth` and `efficientnet.onnx` were Git LFS pointer files, not usable model weights, so they are intentionally not included in this V1 package.

Some legacy inference paths require valid checkpoint files. Obtain the matching model weights from the project owner or an authorized source before using those paths. Do not treat predictions as validated until inference has been tested with the correct weights.
