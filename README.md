# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices.

This project is using the dog breed classication data set and applying a ResNet50 model with transfer learning. At first, we're doing hyperparameter tuning using a smaller fraction of the dataset. Afterwards, we're training the model with the best hyperparameters on the full dataset. This is monitored by debugger and profiling hooks using the Sagemaker tools. At the end, the final model is deployed to an endpoint and queried using a sample image.

## Project Set Up and Installation

First, the Sagemaker instance is created. In my case, I'm using my own AWS account and not the Udacity classroom.

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
It contains training, test and validation images for various dog breeds - you can see more details in the train_and_deploy notebook.

### Access
Next, the extracted data is uploaded to S3; either aws s3 sync command line tool.

## Hyperparameter Tuning
I used the resnet50 pretrained model provided by PyTorch, as it is a powerful model that is well suited to distinguish objects like in this task. The last layer is replaced with an own layer to make it suitable for dog breed classification.

For the hyperparameter tuning, I tested batch size and learning rates in various ranges. The best hyperparameters are the ones that minimize the loss.

* A screenshot of completed training jobs:

![hyperparameter tuning job results](screenshots/hyperparameter-tuning-job.png)

* Logs metrics during the training process

|	|batch-size	|lr	|TrainingJobName	|TrainingJobStatus	|FinalObjectiveValue	|TrainingStartTime	|TrainingEndTime	|TrainingElapsedTimeSeconds|
|---|-------|-----------|-------------------------------------------|-----------|-----------|---------------------------|---------------------------|-----|
|0	|"32"	|0.003251	|pytorch-training-230221-2133-004-5f79cb85	|Completed	|0.065263	|2023-02-21 21:57:49+00:00	|2023-02-21 22:02:17+00:00	|268.0|
|1	|"32"	|0.001084	|pytorch-training-230221-2133-003-c320b9e0	|Completed	|0.066843	|2023-02-21 21:52:33+00:00	|2023-02-21 21:57:05+00:00	|272.0|
|2	|"512"	|0.096445	|pytorch-training-230221-2133-002-70dd903a	|Completed	|77.830025	|2023-02-21 21:44:01+00:00	|2023-02-21 21:51:04+00:00	|423.0|
|3	|"512"	|0.089827	|pytorch-training-230221-2133-001-1143ce62	|Completed	|61.521866	|2023-02-21 21:34:46+00:00	|2023-02-21 21:42:14+00:00	|448.0|

* Tune at least two hyperparameters

Tuned both batch-size and learning rate.

* Retrieve the best best hyperparameters from all your training jobs

batch-size: 0.0032505765136462395
learning rate: 0.0032505765136462395

## Debugging and Profiling

**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker

I performed training and debugging through the Amazon SageMaker library SMDebug. I added the corresponding calls to the hpo.py (which I extended so that it can also be used for training by configuring a parameter, instead of running the extra train_model.py script). Both the test and train functions set hooks to measure performance during training.

Profiling gives insights for example about memory and CPU usage of the instances. This can be used for example to identify bottlenecks.

### Results

I performed training using an ml.m5.2xlarge instance; overall, it took 22 minutes.
**TODO**: What are the results/insights did you get by profiling/debugging your model?

**TODO** Remember to provide the profiler html/pdf file in your submission.


## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.
![Screenshot of deployed endpoint](screenshots/endpoint.png)

