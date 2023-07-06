# Machine Learning Project: Recommendation System

## Members
- Nguyen Hoang Anh - 20214946 - anh.nh214946@sis.hust.edu.vn
- Tran Hoang Anh - 20210023 - anh.th210023@sis.hust.edu.vn
- Nguyen Tuan Long - 20214963 - long.nt214963@sis.hust.edu.vn
- Bui Hai Duong - 20214953 - duong.bh214953@sis.hust.edu.vn

## Topic
A recommendation system (recommender system) is a system designed to provide suggestions for items that are relevant to a specific user.

## Dataset
- We use the first 100,000 lines of the file `UserBehaviour.csv` published by Taobao on https://tianchi.aliyun.com/dataset/649.
- The dataset consists of five attributes: `user_id`, `item_id`, `category_id`, `behavior`, `timestamp`. The column `behavior` can have  4 values `{'pv', 'fav', 'cart', 'buy'}` corresponding to 4 actions: visit, favor, add to cart and buy.

## Proposed algorithms
- We propose three algorithms to solve the problem:
    - Collaborative filtering, user-based
    - Collaborative filtering, item-based
    - Collaborative filtering model-based with matrix factorization
- In our case, matrix factorization seems to be the best choice because the dataset is sparse and the number of users and items is large. Collaborative filtering is not very effective due to the sparsity of the dataset.

## Implementation

### Collaborative filtering, user-based
- In this algorithm, we assume that an user may also want items that are similar to his/her favorites. So there are 2 main steps in this algorithm:
    - Determine the k-nearest neighbors (KNNs) for each user,
    - Calculate the predicted scores for all unknown scores for each user.

### Collaborative filtering, item-based
- The idea in this algorithm is similar to the approach of user-based collaborative filtering. The idea is very simple: An user may temp to like similar items to what he liked in the past.
- Instead of finding KNNs for each user, we find the KNN for each item, then use known scores of these neighbors to fill in unknown values for each item.

### Collaborative filtering model-based with matrix factorization
- In this algorithm, we use matrix factorization to factorize the user-item matrix into two matrices: user matrix and item matrix. The user matrix is a matrix of size (number of users, number of latent factors) and the item matrix is a matrix of size (number of latent factors, number of items).
- We provide two models for this algorithm: one with bias term and one without bias term.
- We will find 2 embedding vectors $U$ and $V$ of $R$ such that $R \sim U \times V$ ($\times$ is a matrix multiplication operation). Then, unknown scores that user $i$ give to item $j$ can be calculated based on the result of $U \times V$.

## Results
Results can be found in the report.

## How to run?
### Dependencies
- TensorFlow: `pip install tensorflow`
- Matplotlib: `pip install matplotlib`
- scikit-learn: `pip install -U scikit-learn`

### How to run
Simply run the code in `RecommendationEngine.ipynb`. The code is well-documented and easy to understand.