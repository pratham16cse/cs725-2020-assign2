# CS 725 (Autumn 2020): Assignment 2

## General Instructions
Please read the following important instructions before getting started on the assignment.
1. This assignment should be completed individually.
2. This assignment is entirely programming-based. A correponding Kaggle task is also hosted [here](https://www.kaggle.com/c/cs725-autumn-2020-assignment-2/overview). Instructions to join the Kaggle competition will be shared on Moodle.
3. Your final submission should be a .tar.gz bundle of a directory organized exactly as described in the [Submission Instructions](#submission-instructions). Submissions that do not strictly adhere to this structure will be penalized.
4. Successful completion of the assignment would include: (A) Submitting <your-roll-number>.tar.gz on Moodle and (B) Having your roll number appear on the Kaggle leaderboard.

## Implement a Feedforward Neural Network using NumPy
This assignment will familiarize you with training and evaluating feedforward neural networks. You will work on a regression task where you will have to predict the release year of a song from a set of timbre-based audio features extracted from the song. This consists of a year range between 1922 to 2011. (More details about this corpus are available [here](https://archive.ics.uci.edu/ml/datasets/yearpredictionmsd).) Click [here](https://www.kaggle.com/c/cs725-autumn-2020-assignment-2/data) to download the training, development and test sets from Kaggle.

## Dataset Information
- The dataset contains three files `train.csv`, `dev.csv`, and `test.csv`
- `train.csv` and `dev.csv` contains following columns:
```
1. label - Year of release of the song in the range [1922, 2011]
2. TimberAvg1
3. TimberAvg2
.
.
13. TimberAvg12
14. TimbreCovariance1
15. TimbreCovariance2
.
.
91. TimbreCovariance78
```
<!---| Label       | TimberAvg1 | TimberAvg2 | ... | TimberAvg12 | TimbreCovariance1 | TimbreCovariance2 | ... | TimbreCovariance78 | --->
<!---|---|---|---|---|---|---|---|---|---|--->
- `test.csv` contains same features except `label`

## Part 1
In Part 1, you will implement the neural network, train it using train data and report its performance on dev data.

### Part 1.A (25 Points)
Implement the functions definitions given in `nn.py` to crate and train a neural network. Run stochastic gradient descent on Mean Squared Error (MSE) loss function.

For both Part 1.A and Part 1.B, use fixed values of following hyper-parameters:
```
- Batch size for training: 256
- Seed for numpy: 42
```
#### What to submit in Part 1.A?
For Part 1.A, there is no report submission. Only code needs to be submitted.

### Part 1.B (15 Points)
Report Root Mean Squared Error (RMSE) on training and dev data using following hyper-parameter configurations.


|Learning Rate | No. of hidden layers | Size of each hidden layer | λ(regulariser) | RMSE(train) | RMSE(dev) |
|---|---|---|---|---|---|
|0.001 | 1 | 64 | 0.1 |
|0.001 | 1 | 64 | 10 |
|0.001 | 1 | 128 | 0.1 |
|0.001 | 1 | 128 | 10 |
|0.001 | 2 | 64 | 0.1 |
|0.001 | 2 | 64 | 10 |
|0.001 | 2 | 128 | 0.1 |
|0.001 | 2 | 128 | 10 |
|0.01 | 1 | 64 | 0.1 |
|0.01 | 1 | 64 | 10 |
|0.01 | 1 | 128 | 0.1 |
|0.01 | 1 | 128 | 10 |
|0.01 | 2 | 64 | 0.1 |
|0.01 | 2 | 64 | 10 |
|0.01 | 2 | 128 | 0.1 |
|0.01 | 2 | 128 | 10 |

#### What to submit in Part 1.B?
Create a section `Part 1.B` in the `Report.pdf` and fill the above table in this section.

## Part 2 (10 points)
In Part 2, you will evaluate your network's performance on test data given in `test.csv`. Submit your predictions on test data on [Kaggle competition](https://www.kaggle.com/c/cs725-autumn-2020-assignment-2/overview) in a `<roll_number>.csv` file in the following format:
```
Id,Predictions
1.0,2000.0
2.0,2000.0
3.0,2000.0
.
.
10000.0,2000.0
```

You are free to use any hyper-parameters in this task.
Report the hyper-parameter configurations you used and score obtained on test data in the leaderboard. Clearly specify the hyper-parameters you used to and the score obtained on leaderboard using those hyper-parameter configurations.

#### What to submit in Part 2?
Create a section `Part 2` in the `Report.pdf` and write the hyper-parameters and scores obtained on test data.

#### Tips to improve your rank on leaderboard
You can explore following techniques to get better generalization performance
- [Feature Scaling](https://en.wikipedia.org/wiki/Feature_scaling)
- [Feature Selection](https://en.wikipedia.org/wiki/Feature_scaling)
- [Dropout](https://youtu.be/qfsacbIe9AI?list=PLyqSpQzTE6M9gCgajvQbc68Hk_JKGBAYT)
- [Batch Normalization](https://youtu.be/1XMjfhEFbFA?list=PLyqSpQzTE6M9gCgajvQbc68Hk_JKGBAYT)
- [Early Stopping](https://youtu.be/zm5cqvfKO-o?list=PLyqSpQzTE6M9gCgajvQbc68Hk_JKGBAYT)

## Submission Instructions
- Your submission should contain three files: (i) `nn.py`, (ii)`Report.pdf`, and (iii)`Readme.txt`.
- Use `Readme.txt` to describe any other information needed to run your code successfully.
- Add these files to directory `<your_roll_number>`.
- Compress the directory `<your_roll_number>` in .tgz format using following command:
 
  ```tar -czf <your_roll_number>.tar.gz <your_roll_number>```
  
- Submit the `<your_roll_number>.tar.gz` file.




