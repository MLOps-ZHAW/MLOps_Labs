# Experiment and Model Management with MLflow

## What is MLFLow?

In their own words:

> MLflow is an open-source platform, purpose-built to assist machine learning practitioners and teams in handling the complexities of the machine learning process. MLflow focuses on the full lifecycle for machine learning projects, ensuring that each phase is manageable, traceable, and reproducible.

MLFLow is an end-to-end ML platform with several components that can be used for much more than experiment management:

- [Tracking][1]: An API and UI for logging parameters, code versions, metrics, and artifacts.
- [Evaluate][2]: `mlflow.evaluate` provides an API to evaluate the performance of a model on one or more datasets. It supports classification, regression, and a range of language modeling tasks. Evaluations are logged to the tracking API.
- [Model Registry][3]: An API and UI for storing and managing different versions of models.
- [Projects][4]: MLflow Projects are a standardized format for packaging ML code, workflows, and artifacts.
- [MLflow deployments for LLMs][5]: A server that provides a set of standardized APIs for streamlining access to both closed- and open-source LLMs.
- [Prompt Engineering UI][6]: A dedicated UI for experimenting, refining, and evaluating prompts.
- [Recipes][7]: Blueprints for structuring (real-world) ML projects and deployments.

In this lab, we will primarily be using the tracking. Other components follow in future labs.
Note that you have to use your own computer for the MLflow parts. Google Colab has dropped support for background processes, which we need for this lab.

## The MLflow UI

Let's start a local instance of MLFLow (run this in a terminal):

```shell
mlflow server --host 127.0.0.1 --port 8080
```

This will start an MLFlow tracking server, its UI and all the other necessary components. Use your browser to navigate to `localhost:8080`. You should be presented with a page that looks like the one in the screenshot below:

![MLFlow UI](imgs/image.png)

As you can see, there are two tabs: one for experiments and one for models. In the subsequent sections, we will introduce the various components, so keep the MLflow instance running.

## What you will learn

| Topic                 | Notebook |
|-----------------------|----------|
| Experiment tracking with MLflow | [Tracking](mlflow_tracking.ipynb) |
| Hyperparameter tuning with Ray Tune | [Tuning](ray_tune.ipynb) ([Solution](ray_tune_solution.ipynb))|
| Hyperparameter tuning a teeny tiny diffusion model| [Diffusing](diffusion.ipynb)|

## Further reading

As mentioned in the respective notebooks, both MLflow and Ray can do much more than we have shown you here that could be of interest to you - not only because they might act as inspiration for your MLOps project later in the semester. We highly recommend you take a look at the other features of both tools. For your convenience, we repeat the MLFlow documentation links below.

### MLflow

- [Tracking][1]: An API and UI for logging parameters, code versions, metrics, and artifacts.
- [Evaluate][2]: `mlflow.evaluate` provides an API to evaluate the performance of a model on one or more datasets. It supports classification, regression, and a range of language modeling tasks. Evaluations are logged to the tracking API.
- [Model Registry][3]: An API and UI for storing and managing different versions of models.
- [Projects][4]: MLflow Projects are a standardized format for packaging ML code, workflows, and artifacts.
- [MLflow deployments for LLMs][5]: A server that provides a set of standardized APIs for streamlining access to both closed- and open-source LLMs.
- [Prompt Engineering UI][6]: A dedicated UI for experimenting, refining, and evaluating prompts.
- [Recipes][7]: Blueprints for structuring (real-world) ML projects and deployments.

### Ray

Ray comes with a similar wealth of features!

- [`Ray Core`](https://docs.ray.io/en/latest/ray-core/walkthrough.html), the foundation that everything in the Ray ecosystem is built-on. This library contains core primitives (i.e., tasks, actors, objects) for building and scaling distributed applications.
- [`Ray Data`](https://docs.ray.io/en/latest/data/data.html), a scalable distributed data processing library for ML workloads.
- [`Ray Train`](https://docs.ray.io/en/latest/train/train.html), a scalable machine learning library for distributed training and fine-tuning.
- [`Ray Serve`](https://docs.ray.io/en/latest/serve/index.html), a scalable, framework-agnostic model serving library for building online inference APIs

If you like distributed computing but for some reason don't like Ray, maybe [`Dask`](https://docs.dask.org/en/stable/) is for you! Dask is a Python library for parallel and distributed computing. You can think of it as distributed versions of `pandas` and `numpy`.

### Hyperparameter tuning

As you have likely realized when solving the hyperparameter tuning Jupyter notebook, there is much more to it than simple grid search.
Here is a non-exhaustive list of resources that could be of interest if you want to learn more.

- **Random Search for Hyper-Parameter Optimization** by James Bergstra and Yoshua Bengio (2012)
  - This paper introduced the concept of using random search for hyperparameter optimization, showing its effectiveness compared to grid search.
- **Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization** by Lisha Li, Kevin Jamieson, Giulia DeSalvo, Afshin Rostamizadeh, and Ameet Talwalkar (2018)
  - Hyperband introduces a novel bandit-based approach for hyperparameter optimization, showing significant improvements in efficiency compared to traditional methods.
- **Bayesian Optimization with Robust Bayesian Neural Networks** by Jost Tobias Springenberg, Aaron Klein, Stefan Falkner, and Frank Hutter (2016)
  - This paper explores the use of Bayesian optimization with robust Bayesian neural networks for hyperparameter optimization, demonstrating its effectiveness in handling noise and uncertainty.
- **Practical Bayesian Optimization of Machine Learning Algorithms** by Jasper Snoek, Hugo Larochelle, and Ryan P. Adams (2012)
  - The paper presents a practical approach to Bayesian optimization for hyperparameter tuning, with applications in optimizing machine learning algorithms.
- **Population Based Training of Neural Networks** by Max Jaderberg, Valentin Dalibard, Simon Osindero, Wojciech M. Czarnecki, Jeff Donahue, Ali Razavi, Oriol Vinyals, Tim Geen, Iain Dunning, Karen Simonyan, Chrisantha Fernando, and Koray Kavukcuoglu (2017)
  - This paper introduces Population Based Training (PBT), a method for evolving the hyperparameters of a population of models over time, leading to efficient exploration of the hyperparameter space.
- **Efficient and Robust Automated Machine Learning** by Matthias Feurer, Aaron Klein, Katharina Eggensperger, Jost Tobias Springenberg, Manuel Blum, and Frank Hutter (2015)
  - The paper presents Auto-sklearn, an automated machine learning toolkit built on top of scikit-learn, which incorporates Bayesian optimization for hyperparameter tuning.
- **Learning Curve Prediction with Bayesian Neural Networks** by James Lloyd and Roger Grosse (2014)
  - This paper proposes the use of Bayesian neural networks for predicting learning curves in hyperparameter optimization, aiding in the selection of optimal hyperparameters.
- **BOHB: Robust and Efficient Hyperparameter Optimization at Scale** by Stefan Falkner, Aaron Klein, and Frank Hutter (2018)
  - BOHB (Bayesian Optimization and Hyperband) integrates Bayesian optimization with the Hyperband algorithm, providing robust and efficient hyperparameter optimization at scale.
- **SMAC: Sequential Model-Based Algorithm Configuration** by Frank Hutter, Holger H. Hoos, and Kevin Leyton-Brown (2011)
  - SMAC is a sequential model-based algorithm configuration approach that combines Bayesian optimization with model-based parameter configuration techniques, demonstrating its effectiveness in tuning algorithm parameters.
- **Nevergrad – A Gradient-Free Optimization Platform** by The Nevergrad team (2020)
  - Nevergrad is an open-source Python platform for derivative-free optimization, which includes hyperparameter tuning algorithms among its features.

For most methods, open-source implementations are available. If you are looking for other hyperparameter tuning libraries, here are some suggestions:

- [`Ax`](https://ax.dev/), a framework for adaptive experimentation.
- [`BoTorch`](https://botorch.org/), Bayesian optimization in PyTorch.
- [`BayesOpt`](https://github.com/rmcantin/bayesopt), a Bayesian optimization library.
- [`BOHB`](https://www.automl.org/blog_bohb/), robust and efficient hyperparameter optimization at scale.
- [`Optuna`](https://optuna.org/), an open source hyperparameter optimization framework to automate hyperparameter search.

[1]: https://mlflow.org/docs/latest/tracking.html#tracking
[2]: https://mlflow.org/docs/latest/models.html#model-evaluation
[3]: https://mlflow.org/docs/latest/model-registry.html#registry
[4]: https://mlflow.org/docs/latest/projects.html#projects
[5]: https://mlflow.org/docs/latest/llms/deployments/index.html#deployments
[6]: https://mlflow.org/docs/latest/llms/prompt-engineering/index.html#prompt-engineering
[7]: https://mlflow.org/docs/latest/recipes.html#recipes
