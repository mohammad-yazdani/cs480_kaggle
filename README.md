# CS480 Kaggle Competition

# What things I used?
- `keras`
- `sklearn`
- `tensorflow`  (duh)
- `pandas`
- `matplotlib`  (for debugging)
- `numpy`       (duh)

# How to run
1. First down the kaggle data and put the `kaggle` folder at the root of the project.
2. Make sure you have all the dependencies figured (see `requirements.txt`).
3. Run `main.py`

- Without any caching it takes about 15 minutes to run on my system:
    - Ryzen 3700X 8 core
    - 32GB RAM
    - GTX 1660 Super 6GB VRAM

# How does the algorithm work?
So I am using the following data from `train.csv`:
```
        "id",
        "competition-num",
        "category",
        "num-comments",
        "feedback-karma",
        "ratings-given",
        "ratings-received",
        "description",
        "link-tags",
        "num-authors",
        "prev-games",
        "fun-average",
        "innovation-average",
        "theme-average",
        "graphics-average",
        "audio-average",
        "humor-average",
        "mood-average",
        "fun-rank",
        "innovation-rank",
        "theme-rank",
        "graphics-rank",
        "audio-rank",
        "humor-rank",
        "mood-rank",
        "label"
```

- Some of these columns are used to score/cluster other columns and are later dropped.
- I also use the thumbnails

## The DNN:
- Input shape: (26, 1)
- Output shape: (None, 6)
- Loss: Categorical Cross-Entropy
- Optimizer: adam
```
Layer:          Size:       Kernel:     Activation:
Conv1D          128         1           relu
Conv1D          128         1           relu
Conv1D          128         1           relu
Conv1D          128         1           relu
MaxPool1D(2)
Conv1D          96          1           relu
Conv1D          96          1           relu
Conv1D          96          1           relu
MaxPool1D(2)
Conv1D          64          1           relu
Conv1D          64          1           relu
MaxPool1D(2)
Conv1D          32          1           relu
Conv1D          32          1           relu
Flatten()
Dropout(0.2)
Dense           60                      relu
Dense           30                      relu
Dense           16                      relu
Dense           6                       softmax
```

## How do I use everything besides description and thumbnails?
I basically pass all of the numerical values straight as features
### How about `link-tags`:
For link-tags, I score link tags based on what they have.
- I realized there is a correlation between games having a certain link-tag (`macos` for example)
and having a higher average score. So link tags are basically scored based on how likely they are to achieve a high rating.
- The link-tag score replaced the actual link tags before passing data to the model.

## How do I use descriptions? [NLP 😎]
Descriptions are scored in two ways.
- One way is through clustering and correlating the cluster label to the average rating of the cluster.
    - After studying the data, I noticed that running KNNClustering on the description texts based on size buckets gives a decent prediction of the ratings.
    - The way it works is that descriptions are divided into buckets based on increasing size.
    - Buckets are [0], (0, 50], (50, 100], (100, 250], (250,500], (500, max]
    - The training data is divided into these buckets and every bucket gets local cluster labels by running KNNClustering.
    - Later for evaluation, the description is evaluated based on length -> label score of sub-cluster.

- The other way is through how many links are in the description:
    - After studying the correlation between the number of links in description and rating, I noticed number of urls are pretty decent in guiding the model to good predictions.
    
## How do I use thumbnails? [CV 😭]
(This was the hardest part, and I'm still debating myself on how impactful it was.)
Before I explain how I used the thumbnails I'd like to share some experiments and observations.
- I ran multiple Keras models (VGG16/19, ResNet, Inception, Xception)
and pretty much all of the did the same sort of clustering the thumbnails. What you will get is roughly 6 labels and 4 labels have a 22 ish distribution with the other two being the rest and zero.
- I also ran a CNN model on all the thumbnails with ratings as labels (I'm gonna bet a lot of students did this too). For my experiments, the model would not achieve more than 40 percent accuracy when running on the test split of the train data (80/20 ratio).
- I also randomly selected thumbnails from different rating groups and visually inspected myself (I was really frustrated).
- Observations:
    - Thumbnails can be used as good features but ONLY when the problem is: Is the game better than rating or not.
    - Basically they are useful to predict if a game is very bad or mediocre.

**Finally, How I used thumbnail**
- I run a VGG19 model to extract features.
- Then I flatten the features and pass to KMean clustering.
- I assign an image score based on how good the predicted label of the image has done before (averaging the rating of the labels).
- This will basically give you almost the same score for mediocre to top and lower scores for bottom ratings.

# How can this be improved?
- I think I could have spent some of the time I worked on CV and put it on NLP. I believe the descriptions of each game can be a much better feature than thumbnails.
# cs480_kaggle
