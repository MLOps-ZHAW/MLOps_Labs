# ML Deployment - The Last Mile and beyond

As more enterprises and startups alike develop their AI capabilities, we’re seeing a common roadblock emerge — known as AI’s “last mile” problem. When machine learning engineers and data scientists refer to the
"last mile", they usually mean the steps required to make an AI application available for widespread use.

> The last mile describes the short geographical segment of delivery of communication and media services or the delivery of products to customers located in dense areas. Last mile logistics tend to be complex and costly to providers of goods and services who deliver to these areas. (Investopedia).

In this lab, we look at how to bridge the last gap - because your job as a data scientist is not over once a model trained. Concretely, this means, we seek to answer the questions:

- How to deploy a model?
- How to monitor models?
- How to explain model output to your users?

In the first half, we will be using Seldon's open-source `MLServer`, and in the second half, we will be building our own solution for detecting outliers, attacks, and drift.

|Topic|Link|
|:----|:---|
|Deployment with `MLServer`| [`mlserver.md`](./mlserver.md) |
|Detecting data drift| [`outliers_attacks_drifts.md`](./outliers_attacks_drifts.md) |

## Additional Resources

### Inference solutions

`MLServer` is of course not the only solution for deploying models. Here are a few other solutions:

- [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving)
- [TorchServe](https://pytorch.org/serve/) (deprecated!)
- [KFServer](https://kserve.github.io/)
- [NVIDIA Triton Inference Server](https://www.nvidia.com/en-us/ai-data-science/products/triton-inference-server/)
- ... and many more!

### Monitoring solutions

You do always have to build your own post-deployment monitoring solutions. Here are a few tools that can help you:

- [NannyML](https://nannyml.readthedocs.io/)
- [Alibi Detect](https://docs.seldon.io/projects/alibi-detect/en/stable/)
- [Evidently AI](https://github.com/evidentlyai/evidently)
- [Deepchecks Monitoring](https://docs.deepchecks.com/monitoring/stable/getting-started/welcome.html)
- ... and, again, many more!
