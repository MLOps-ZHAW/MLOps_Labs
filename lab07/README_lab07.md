# ML Deployment - The Last Mile and beyond

As more enterprises and startups alike develop their AI capabilities, we’re seeing a common roadblock emerge — known as AI’s “last mile” problem. When machine learning engineers and data scientists refer to the
"last mile", they usually mean the steps required to make an AI application available for widespread use.

> The last mile describes the short geographical segment of delivery of communication and media services or the delivery of products to customers located in dense areas. Last mile logistics tend to be complex and costly to providers of goods and services who deliver to these areas. (Investopedia).

In this lab, we look at how to bridge the last gap - because your job as a data scientist is not over once a model is trained. Concretely, this means, we seek to answer the questions:

- How to deploy a model?
- How to scale and monitor a deployed model?
- How to protect a deployed model from inputs it was not made for, such as outliers and adversarial attacks?

In the first half, we will be using NVIDIA's open-source [Triton Inference Server](https://github.com/triton-inference-server/server), one of the industry standards for serving models, and in the second half, we will be building our own solutions for detecting outliers and adversarial attacks.

## What you will learn

- How to export a model to ONNX and serve it with Triton, without writing a web service.
- How to talk to an inference server via the standardized Open Inference Protocol.
- How to move pre- and postprocessing to the server with the Python backend and ensembles, and how to package a server as a Docker image.
- How to scale a service with model instances and dynamic batching - and how to measure whether it actually helps.
- How to monitor a deployed model with Prometheus metrics.
- How to detect outliers with a variational autoencoder.
- How to correct and detect adversarial attacks by matching the prediction distributions of a classifier.

## Setup

**Before the lab**, install [Docker](https://docs.docker.com/get-started/get-docker/) (or Podman) and pull the Triton image. It is large (about 9 GB download, 16 GB on disk):

```shell
docker pull nvcr.io/nvidia/tritonserver:26.09-py3
```

Then create the Python environment. It contains the Triton client, the tools to export the model (`torch`, `timm`, `onnx`, `onnxruntime`), `foolbox`, `alibi-detect` and JupyterLab:

```shell
conda env create -f lab07/env.yaml
conda activate mlops-lab-07
```

- **Ports:** Triton uses the ports 8000 (HTTP), 8001 (gRPC) and 8002 (metrics). Only one server can run at a time; stop it with `Ctrl+C` before starting the next one.
- **Downloads:** besides the Triton image, the TinyViT weights are downloaded from the Hugging Face Hub (about 50 MB) and the notebooks download MNIST (about 60 MB).
- **Hardware:** everything runs on a laptop CPU; a GPU is optional. Training the models in the notebooks takes a few minutes on a GPU and up to about half an hour on a CPU.
- **Apple Silicon Macs:** Docker runs the `arm64` variant of the Triton image (CPU only). If it does not work on your machine, use your lab VM.

## Lab parts

Work through the parts in this order. Each notebook comes with a `_solution` version.

|Topic|Link|
|:----|:---|
|Deployment with Triton Inference Server| [`triton.md`](./triton.md) |
|Detecting outliers and adversarial attacks| [`outliers_attacks_drifts.md`](./outliers_attacks_drifts.md) |
|Outlier detection with VAEs| [`notebooks/outlier_detection.ipynb`](./notebooks/outlier_detection.ipynb) |
|Adversarial attack detection| [`notebooks/adversarial_attack_detection.ipynb`](./notebooks/adversarial_attack_detection.ipynb) |

## Additional Resources

### Inference solutions

Triton is of course not the only solution for deploying models. Here are a few others:

- [KServe](https://kserve.github.io/): model serving on Kubernetes (can use Triton as its runtime)
- [BentoML](https://www.bentoml.com/): Python-first serving and packaging
- [LitServe](https://lightning.ai/docs/litserve): a lightweight, Python-first inference server
- [Ray Serve](https://docs.ray.io/en/latest/serve/index.html): scalable serving on Ray clusters
- [vLLM](https://docs.vllm.ai/): serving engine for large language models
- [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)
- [TorchServe](https://pytorch.org/serve/) (archived, no longer maintained!)
- ... and many more!

### Monitoring solutions

You don't always have to build your own post-deployment monitoring solutions. Here are a few tools that can help you:

- [NannyML](https://nannyml.readthedocs.io/)
- [Alibi Detect](https://docs.seldon.ai/alibi-detect)
- [Evidently AI](https://github.com/evidentlyai/evidently)
- [Deepchecks Monitoring](https://docs.deepchecks.com/monitoring/stable/getting-started/welcome.html)
- ... and, again, many more!
