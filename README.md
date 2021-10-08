# Predicting MNIST Digits using K-Nearest Neighbors Algorithm

## Michael Robertson

---



## Introduction

### Data

We use the MNIST (Modified National Institute of Standards and Technology) database of handwritten digits downloaded through Scikit-Learn from
[OpenML](https://www.openml.org/d/554). The dataset contains 70,000 28×28 pixel images of centered digits.

### Model

To classified unlabeled samples, we implemented the K-nearest neighbors algorithm. To classify a given example based off a feature set, this model examines
the distance between the feature vector and every other example in the train set,
and collects those <i>k</i> “neighbors” who have the least distance. Because our 784
pixel-features were numbers, we treated our images as vectors, and use the Euclidean distance as our metric between our training examples and the example
we wish to classify. Having collected these <i>k</i> nearest neighbors, we have several ways to classify the given example. Most simply, we could take a tally of the
categorizations of each of the neighbors, and classify the example as the category with the greatest tally, and tie-break randomly. However, this approach
gives no consideration to distances within the neighbor set. For intermediate
complexity, we could use the distances from the example to each neighbor as
a tie-breaking mechanism, favoring the classificaiton of the neighbor with the
shorter distance. Lastly, we could use the distances from the example to the
neighbor as weights in a linear combination of the classification vector (which
in our case in binary, because a number is only ever appropriately classified as
one number), and examine the magnitude of that vector when projected along
each of the classification axes.
The K-nearest neighbors model uses “Lazy learning,” or instance-based
learning, because it requires no training period prior to prediction. Put another way, when no computation can be front loaded before the classifcation attempt, KNN runs in O(<i>nmo</i>) time, where <i>m</i>, <i>n</i> and <i>o</i> are the number of training examples, the number of unclassified examples we attempt to predict, and the size of the data, respectively. For our case, we have a training set of size 6650,
a test set of size 350, and data of size 28^2 = 784. We thus have a running
time on the order of (6650)(350)(784)≈ 10^8 for our own naive implementation.
However, for a single example, we can predict it in O(<i>nm</i>) time, and we believe
advance techniques allow the professional implementation of this algorithm by
SciKit-Learn to improve the time complexity.

## Results

We use the F<sub>1</sub> score to measure our model’s success. The F<sub>1</sub> combines precision
and recall via the harmonic mean to give a reliable overall measure of our model’s
performance.


<p align="center">
    
  <img src="https://raw.githubusercontent.com/robertson809/MNIST-KNN/main/fig/sklearn.png" alt="drawing" width="600"/>
</p>



We compare these results to those that we naively implemented


<p align="center">
    
  <img src="https://raw.githubusercontent.com/robertson809/MNIST-KNN/main/fig/KNN.png" alt="drawing" width="600"/>
</p>


Because we would clearly use the professional model, we only demonstrate the
functionality of our model. As such, we only used 500 examples in our training
set to predict 500 examples in our validate set. Our implementation of KNN
is clearly outperformed by the professional implementation, but our rates are
certainly higher than those that could come from guessing alone.
Based on the above results and tuning, we recommend using the SciKit-Learn implementation of KNN, with <i>k</i>= 3.

## Conclusion

Our results show that we can confidently classify handwritten digits in our
test set with near perfect accuracy. Our results rival but do not surpass those
reported by Weinberger, who report a “test error rate of 1.3” (we assume their
“test error rate is similar to our F<i>1</i> metric, but we’re unable to find its definition in their work) on the MNIST dataset [1]. In their 2009 work, they substitute
a Mahanalobis distance metric where we used a Euclidean distance to improve
results. The professional implementation from SciKit-Learn allows for a user
defined metric, so we could potentially implement this change to improve our
performance.
Despite our strong results, care should be taken in expecting these results
to generalize to mail, because the input data will likely be of a very different
form. Although one might expect that an image of a letter that showed digits
in a higher resolution than 28×28 pixels would be easier to predict, results in
machine learning are often counter-intuitive, and nothing should be taken for
granted. Secondly, if we hope to sort letters, we will also need to classify letters.
The increase in the number of classes from which we much choose may also
complicate the problem. So, although our results here are certainly promising,
there are several major challenges to address before we have a functioning mail
sorting system.


## References

[1] Kilian Q Weinberger and Lawrence K Saul. Distance metric learning for
large margin nearest neighbor classification. Journal of Machine Learning
Research, 10(Feb):207–244, 2009.
