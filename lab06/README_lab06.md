# It's the data, silly

Data science teams face data management questions around versions of data and machine learning models. How do we keep track of changes in data, source code, and ML models together? What's the best way to organize and store variations of these files and directories?
Another problem in the field has to do with bookkeeping: being able to identify past data inputs and processes to understand their results, for knowledge sharing, or for debugging.

This week, we continue our exploration of data-centric MLOps. After covering data pipelines in the previous lab,
we will now look at the data itself: Storing, versioning - and generating more of it!

## What you will learn

- What data-centric MLOps is and how it's different from model-centric MLOps.
- `git lfs` and why it alone is not enough to handle your datasets.
- Data version control with `dvc`, a tool built on top of `git lfs`.
- What synthetic data is, how you can synthesize your own datasets, and where its limits are.

For each of the topics we have prepared separate sections, which you can find below.

|Topic|Link|
|:----|:----|
|Data-centric MLOps| [Click me](./data_centric_mlops.md) |
|Git LFS| [Click me](./git_lfs.md)|
|DVC| [Click me](./dvc.md)|
|More data! (Augmentation by Albumentation)| [Click me](./notebooks/albumentations.ipynb)|
|Even more data! (Synthetic data by HF Diffusers)| [Click me](./notebooks/synthetic_data.ipynb) |

## Recommended Reading

Today's lab might be over, but there's still a lot that we haven't covered.

### Data Versioning

Data versioning and the associated practices are an active field of research. Here are some reading suggestions:

- [Versioning Data Is About More than Revisions: A Conceptual Framework and Proposed Principles](https://datascience.codata.org/articles/10.5334/dsj-2021-012): This paper is about what constitutes good data versioning practices.
- [Understanding Data Storage and Ingestion for Large-Scale Deep
Recommendation Model Training](https://arxiv.org/abs/2108.09373): This paper describes the data storage and ingestion system that Meta uses for their content recommendation algorithms.
- [NeSSA: Near-Storage Data Selection for Accelerated Machine Learning Training](https://dl.acm.org/doi/10.1145/3599691.3603404): This paper proposes a novel training architecture that selects an "optimal" subset from a big dataset. It also shows the impact of (very) large dataset on training times.

### But I **really** love `git`

There are alternatives to `LFS` that scale better. One such alternative is `git-annex`.
In their own words,

> git-annex allows managing large files with git, without storing the file contents in git. It can sync, backup, and archive your data, offline and online. Checksums and encryption keep your data safe and secure. Bring the power and distributed nature of git to bear on your large files with git-annex.

You can learn more about `git-annex` on [its website](https://git-annex.branchable.com/).

### DVC alternatives

`DVC` is of course not the only data versioning tool on the market. Here are two alternatives that might be of interested to you.

- [Oxen](https://www.oxen.ai/): a lightning fast alternative to DVC, but less stable. Ask the lecturer for anecdotes. ;)
- [LakeFS](https://lakefs.io/): a heavier alternative to DVC that scales to Petabytes of data. For when you are working with _a lot_ of data.

### Synthetic data

Synthetic data is a hot topic right now. There are many libraries out there, which can be used to generate data, most of which are enterprise solutions
with rather cumbersome open source versions. Below are a two excellent open source alternatives that also come with great documentation and links to papers.

- [synthcity](https://github.com/vanderschaarlab/synthcity), a library for generating and evaluating synthetic tabular data for privacy, fairness and data augmentation.
- [nbsynthetic](https://github.com/NextBrain-ai/nbsynthetic), a simple and robust tabular synthetic data generation library for small and medium size datasets.

Generally, such libraries are limited when it comes to domain specific applications. Then it is often best to train your own generative models and come up with you own benchmarks. After all, you should know your data best!
