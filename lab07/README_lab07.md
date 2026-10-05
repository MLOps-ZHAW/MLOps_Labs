# ML Deployment - The Last Mile and beyond

As more enterprises and startups alike develop their AI capabilities, we’re seeing a common roadblock emerge — known as AI’s “last mile” problem. When machine learning engineers and data scientists refer to the
"last mile", they usually mean the steps required to make an AI application available for widespread use.

> The last mile describes the short geographical segment of delivery of communication and media services or the delivery of products to customers located in dense areas. Last mile logistics tend to be complex and costly to providers of goods and services who deliver to these areas. (Investopedia).

In this lab, we look at how to bridge the last gap - because your job as a data scientist is not over once a model is trained. Concretely, this means, we seek to answer the questions:

- How to deploy a model?
- How to scale and monitor a deployed model?
- How to protect a deployed model from inputs it was not made for, such as outliers and adversarial attacks?

In the first half, we will be using Seldon's open-source `MLServer`, and in the second half, we will be building our own solutions for detecting outliers and adversarial attacks.

## What you will learn

- How to serve a model with `MLServer`, both with your own Python code and with a built-in inference runtime.
- How to talk to an inference server via the standardized Open Inference Protocol.
- Which knobs `MLServer` offers to scale a service (parallel workers, adaptive batching) and how to package it as a Docker image.
- How to detect outliers with a variational autoencoder.
- How to correct and detect adversarial attacks by matching the prediction distributions of a classifier.

## Setup

```shell
conda env create -f lab07/env.yaml
conda activate mlops-lab-07
```

The environment contains `mlserver` (with the Hugging Face runtime), `timm`, `torch`, `foolbox`, `alibi-detect` and JupyterLab.

- **Older library versions on purpose.** `mlserver-huggingface` 1.7.1 only works with `torch<2.9` and `transformers<4.42`, and `mlserver` itself breaks with `uvloop>=0.22`. That's why the environment pins older versions than the other labs.
- **Ports:** `MLServer` uses the ports 8080 (REST), 8081 (gRPC) and 8082 (metrics). Only one server can run at a time; stop it with `Ctrl+C` before starting the next one.
- **Downloads:** the models are downloaded from the Hugging Face Hub on first use (TinyViT about 50 MB, DistilGPT2 about 350 MB). The notebooks download MNIST (about 60 MB).
- **Hardware:** everything runs on a laptop. Training the models in the notebooks takes a few minutes on a GPU and up to about half an hour on a CPU.
- **Docker** is only needed for the optional packaging step in the `MLServer` part.

## Lab parts

Work through the parts in this order. Each notebook comes with a `_solution` version.

|Topic|Link|
|:----|:---|
|Deployment with `MLServer`| [`mlserver.md`](./mlserver.md) |
|Detecting outliers and adversarial attacks| [`outliers_attacks_drifts.md`](./outliers_attacks_drifts.md) |
|Outlier detection with VAEs| [`notebooks/outlier_detection.ipynb`](./notebooks/outlier_detection.ipynb) |
|Adversarial attack detection| [`notebooks/adversarial_attack_detection.ipynb`](./notebooks/adversarial_attack_detection.ipynb) |

## Additional Resources

### Inference solutions

`MLServer` is of course not the only solution for deploying models. Here are a few other solutions:

- [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)
- [TorchServe](https://pytorch.org/serve/) (no longer actively maintained!)
- [KServe](https://kserve.github.io/)
- [NVIDIA Triton Inference Server](https://www.nvidia.com/en-us/ai-data-science/products/triton-inference-server/)
- ... and many more!

### Monitoring solutions

You don't always have to build your own post-deployment monitoring solutions. Here are a few tools that can help you:

- [NannyML](https://nannyml.readthedocs.io/)
- [Alibi Detect](https://docs.seldon.ai/alibi-detect)
- [Evidently AI](https://github.com/evidentlyai/evidently)
- [Deepchecks Monitoring](https://docs.deepchecks.com/monitoring/stable/getting-started/welcome.html)
- ... and, again, many more!
