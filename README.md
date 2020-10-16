# CS 725 (Autumn 2020): Assignment 2


This assignment is due by **11.55 pm on November 1, 2020**. For each additional day after Nov 1st until Nov 3rd, there will be a 10% reduction in marks. The submission portal on Moodle will close at 11.55 pm on Nov 3rd.

## General Instructions
Please read the following important instructions before getting started on the assignment.
1. This assignment should be completed individually.
2. This assignment is entirely programming-based. A correponding Kaggle task is also hosted [here](https://www.kaggle.com/c/cs725-autumn-2020-assignment-2/overview). Please signup on Kaggle using your IITB LDAP email accounts, with Kaggle `Display Name = <your_roll_number>` . Instructions to join the Kaggle competition will be shared on Moodle.
3. Your final submission should be a .tar.gz bundle of a directory organized exactly as described in the [Submission Instructions](#submission-instructions). Submissions that do not strictly adhere to this structure will be penalized.
4. Successful completion of the assignment would include: (A) Submitting <your-roll-number>.tar.gz on Moodle and (B) Having your roll number appear on the Kaggle leaderboard.

## Implement a Feedforward Neural Network using NumPy
This assignment will familiarize you with training and evaluating feedforward neural networks. You will work on a regression task where you will have to predict the release year of a song from a set of timbre-based audio features extracted from the song. This consists of a year range between 1922 to 2011. (More details about this corpus are available [here](https://archive.ics.uci.edu/ml/datasets/yearpredictionmsd).) Click [here](https://www.kaggle.com/c/cs725-autumn-2020-assignment-2/data) to download the training, development and test sets from Kaggle.

## Dataset Information
- The dataset contains three files `train.csv`, `dev.csv`, and `test.csv`
- Each row in the `__.csv` file contains timbre-based audio features extracted from a song.
- The dataset has 90 features: 12 timbre average values and 78 timbre covariance values. Each column denotes a feature.
- `train.csv` and `dev.csv` contains following columns:
```
1. label - Year of release of the song in the range [1922, 2011]
2. TimbreAvg1
3. TimbreAvg2
.
.
13. TimbreAvg12
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
Implement the functions definitions given in [nn.py](nn.py) to create and train a neural network. Run stochastic gradient descent on Mean Squared Error (MSE) loss function.

For both Part 1.A and Part 1.B, use fixed values of following hyper-parameters:
```
- Number of epochs: 50
- Batch size for training: 128
- Seed for numpy: 42
- Use ReLU activation function for all HIDDEN layers.
```

#### Initialization of Weights and Biases (for both Part 1 and Part 2)
Initialize Weights and Biases using uniform distribution in the range \[-1, 1\].


#### What to submit in Part 1.A?
For Part 1.A, only code needs to be submitted.

### Part 1.B (15 Points)
Report Root Mean Squared Error (RMSE) on training and dev data using following hyper-parameter configurations.


|Learning Rate | No. of hidden layers | Size of each hidden layer | λ(regulariser) | RMSE(train) | RMSE(dev) |
|---|---|---|---|---|---|
|0.001 | 1 | 64 | 0 |
|0.001 | 1 | 64 | 5 |
|0.001 | 1 | 128 | 0 |
|0.001 | 1 | 128 | 5 |
|0.001 | 2 | 64 | 0 |
|0.001 | 2 | 64 | 5 |
|0.001 | 2 | 128 | 0 |
|0.001 | 2 | 128 | 5 |
|0.01 | 1 | 64 | 0 |
|0.01 | 1 | 64 | 5 |
|0.01 | 1 | 128 | 0 |
|0.01 | 1 | 128 | 5 |
|0.01 | 2 | 64 | 0 |
|0.01 | 2 | 64 | 5 |
|0.01 | 2 | 128 | 0 |
|0.01 | 2 | 128 | 5 |

#### What to submit in Part 1.B?
Fill the table in `part_1b.csv` file given in [this](https://github.com/pratham16cse/cs725-2020-assign2) repository.
<!---2. Create a section `Part 1.B` in the `Report.pdf` and write your observations from the results in `Results.csv` file.--->

## Part 2 (10 points)
In Part 2, you will evaluate your network's performance on test data given in `test.csv`.

In this part, there is no restriction on any hyper-parameter values. You are also allowed to explore various hyper-parameter tuning and cross-validation techniques.

You are also free to create any wrapper functions over given functinos in [nn.py](nn.py)

Submit your predictions on test data on [Kaggle competition](https://www.kaggle.com/c/cs725-autumn-2020-assignment-2/overview) in a `<roll_number>.csv` file in the following format:
```
Id,Predictions
1.0,2000.0
2.0,2000.0
3.0,2000.0
.
.
10000.0,2000.0
```

<!---Report the hyper-parameter configurations you used and score obtained on test data in the leaderboard.--->
<!---Clearly specify the hyper-parameters you used and the score obtained on leaderboard using those hyper-parameter configurations.--->
In a CSV file, write the name of the hyper-parameter and the value you used.

#### What to submit in Part 2?
Create a two-column csv file `part_2.csv` and write the name of hyper-parameter in first column and value in the second column.

For example:
| Name | Value |
|---|---|
| `learning_rate` | 0.003 |
| `batch_size` | 48 |
| `dropout` | 0.15 |


#### Tips to improve your rank on leaderboard
You can explore following techniques to get better generalization performance
- [Feature Scaling](https://en.wikipedia.org/wiki/Feature_scaling)
- [Feature Selection](https://en.wikipedia.org/wiki/Feature_scaling)
- [Dropout](https://youtu.be/qfsacbIe9AI?list=PLyqSpQzTE6M9gCgajvQbc68Hk_JKGBAYT)
- [Batch Normalization](https://youtu.be/1XMjfhEFbFA?list=PLyqSpQzTE6M9gCgajvQbc68Hk_JKGBAYT)
- [Early Stopping](https://youtu.be/zm5cqvfKO-o?list=PLyqSpQzTE6M9gCgajvQbc68Hk_JKGBAYT)

## Submission Instructions
- Your submission should contain four files: (i) `nn.py`, (ii)`part_1b.pdf`, (iii)`part_2.csv`, and (iii)`Readme.txt`.
- Use `Readme.txt` to describe any other information needed to run your code successfully.
- Add these files to directory `<your_roll_number>`.
- Compress the directory `<your_roll_number>` in .tgz format using following command:
 
  ```tar -czf <your_roll_number>.tar.gz <your_roll_number>```
  
- Submit the `<your_roll_number>.tar.gz` file.




