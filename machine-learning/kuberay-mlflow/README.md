# Machine Learning Tools You Should Know About

## Developing and Service Models With KubeRay + MLFlow

Developing machine-learning models locally can be cumbersome.
A typical workflow might be:

- train a model in a notebook
- save it to disk (or deploy it to a test environment)
- manually test it with a script
- verify results or return to the top

Somewhere along the way, something changes
(dependencies upgrade or differ, the environment changes, etc.),
and the model you tested locally behaves differently once it's deployed.

There are a lot of modern tools which can help shorten this feedback loop,
but they can be complicated for people who aren't experts with them.
Using a lot of tools also means that there can be a lot of variation in how
they're deployed which adds to the complexity we mentioned earlier.

## Standardized Local Deployment

In this project, we build and run an end-to-end
machine learning workflow entirely on a local machine using some really
neat tools (some that you could use in production):

- Skaffold + Minikube to deploy everything with one command
- Ray to handle distributed training and model serving
- FastAPI + Ray Serve to expose a production-style inference API
- MLflow to track experiments, log models, and manage artifacts
- Redis as Ray’s Global Control Store (GCS) for coordination and state

Individually, none of these tools are new.
What is powerful is how they fit together.
This setup demonstrates how we can simplify our model development
and tracking process, both locally or in a hosted environment.

As mentioned above, our goal is to standardize the development process so that
your local setup looks similar to a production setup.
This is the idea behind "shift-left" methodology.
The more robustly you can run these workflows locally, the more quickly and
reliably you can ship software, models, and other pieces of tech.

### What Is Ray, and Why Use It?

At its core,
[Ray](https://www.ray.io/)
is a distributed execution engine.
Ray lets you write Python code that looks local but
can run across multiple processes, cores, or machines.
You define work using:

- Ray tasks (`@ray.remote`) for stateless distributed functions
- Ray actors for stateful, long-running workers

Behind the scenes, Ray handles everything related to distributed execution:

- Scheduling tasks across workers
- Passing data between tasks efficiently
- Tracking task execution and failures
- Monitoring, metrics, and log aggregation
- Coordinating distributed state via a control plane (GCS)

#### What Does "Distributed Training" Mean?

Distributed training doesn’t necessarily mean training a massive neural
network across hundreds of GPUs.
More generally, it is the process of running tasks on different threads,
processes, or even machines to parallelize work.
We can split up the work to train a model amongst multiple workers which means
the training can happen much faster than if everything was a single, serial batch.

In this project, Ray `remotes` are used to train a model using this distributed
pattern.
Once the model is trained, we will log
its results directly to a tool called MLflow (discussed below).

#### What Does It Mean to "Serve" a Model?

Model serving is about turning a trained model into a query-able API.
In most cases, it means running your model and exposing a `/predict` or similar
endpoint.
You can send some data to your model and it will a computation on your inputs.

With
[Ray Serve](https://docs.ray.io/en/latest/serve/index.html),
we will take our trained model and turn it into a scalable deployment.
Ray is really cool, but Serve is super cool.
I recommend taking a spin through their docs.
Serve is also framework agnostic, so we will create an
HTTP application with
[FastAPI](https://fastapi.tiangolo.com/)
and then host it with Ray, across workers, using Ray Serve.

FastAPI will give us type-safety, amazing request handling, and
custom python logic packaged into routes
while Serve optimizes and distributes across workers.

### What Is MLflow, and Why Does It Matter?

[MLflow](https://mlflow.org/)
is an experiment tracking and model management platform.
It solves a pretty hard ML problem in ML, model tracking and versioning.
For example, we will use it to:

- keep track of what we ran
- what parameters we used
- experiment results (MSE/Loss)
- version our models

MLflow provides all of the above and a lot more and makes the features
available with an intuitive UI or an SDK.
Instead of scattering models across local directories or relying on
naming conventions, MLFlow can become our system of record.

In this project, MLflow is deployed locally in Kubernetes and
integrated directly into Ray training jobs.
When a worker finishes training, it logs the model, metrics, and parameters
before uploading the trained model.
