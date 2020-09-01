## Semantic Gap

-   What the computer see: giant grid of numbers (pixels)

-   What human see: actual image



## KNN (K nearest neighbour) method

#### 0. Images and High-dimensional vectors

-   idea of high dimensional points in the plane
-   concrete images
-   **pixels of image allow us to think these images as high dimensional vectors**

#### 1. Distance Metric

==How to get the difference of pixels between two images==

<img src="C:\Users\26526\Desktop\9.7开学前\cs231n\image\QQ截图20200901131959.png" style="zoom:44%;" />

-   **L1 distance：** $d_1(I1, I2) = \sum_{p}|I_{1}^p - I_{2}^p|$

    -   ==**L1 distance depends on the choice of coordinate system(坐标系)/the coordinate system of the data**==

        rotating the coordinates system may change L1 distance

    -   **Better used when every individual elements in data-vector has explicit meanings, such as when classifying employees, feature vector of a staff could be [age, sallary, seniority]:**                                  
        $$
        X = [25, $3200, 12]
        $$
        

-   **L2 distance：**$d_2(I_1, I_2) = \sqrt{\sum_{p}(I_1^p - I_2^p)^2}$ 

    -   more commonly used

    -   **Better used when entries in data-vector doesn’t have explicit meaning, such as:**
        $$
        X = [1,0,0,2,3]
        $$
        

<img src="C:\Users\26526\Desktop\9.7开学前\cs231n\image\QQ截图20200901132113.png" alt="50" style="zoom:44%;" />

<img src="C:\Users\26526\Desktop\9.7开学前\cs231n\image\QQ截图20200901145344.png" style="zoom:44%;" />

#### 2. KNN

-   with majority voting, **outline of decision regions (decision boundaries) can be smoothed**
-   1-NN may enormously suffer from **noise points** that **interlude into other decision regions**
-   ==while regions in K=5==: no majority among the K-nearest neighbours
    -   several majorities: **make a random pick among them**

<img src="C:\Users\26526\Desktop\9.7开学前\cs231n\image\QQ截图20200901144403.png" alt="2" style="zoom:44%;" />

-   **KNN is a lot more robust to noise**



#### 3. Problem for KNN used as image classifier

-   **Very slow at test time**

-   **L1/L2 distance(Pixel Distance) aren’t good measures of similarity between images:** Euclidean distance/L1 distance are sort of ==vectorial distance functions==, they do not correspond very well to ==perceptual similarity between images==, and are not very good ways to measure distance between images

    KNN is singly using **distance metric** to measure similarity between images. However, distance metric alone actually cannot capture the full difference between iamges

<img src="C:\Users\26526\Desktop\9.7开学前\cs231n\image\QQ截图20200901174600.png" style="zoom:45%;" />

-   **Curse of dimensionality: **asks for training examples to **cover the data space(maybe high dimensional) quite densely**, because we possibly want at least one training image to be close to each test image wherever the test image is situated. This means that **the number of training examples needed grows exponentially with the space’s dimension**.

    <img src="C:\Users\26526\Desktop\9.7开学前\cs231n\image\QQ截图20200901175230.png" style="zoom:44%;" />



## Spliting the Training Set

#### Three parts: train set; validation set; test set (together with cross-validation)

<img src="C:\Users\26526\Desktop\9.7开学前\cs231n\image\QQ截图20200901173241.png" alt="5" style="zoom:44%;" />

-   train set is used to train our model
-   validation set is used to choose hyperparameters (using model trained upon test set), **we choose the best set of hyperparameters that averagely performed greateset on validation folds**
-   test set is **fixed**, **last touched**, **only once**, to represent real-world data and for testing the model’s ability when handling unseen data that could come from anywhere

#### Make sure the partition is random

-   If you’re collecting data over time, **Do not use the earlier fetched data as train-set, and later fetched data as test-set**, which could lead to a shift(偏差) that cause problems

 #### Distribution of Learning Alg’s performance (on different folds of validation set)

<img src="C:\Users\26526\Desktop\9.7开学前\cs231n\image\QQ截图20200901173620.png" style="zoom:50%;" />

-   line goes through the mean
-   bars indicated standard deviation (方差，标准差)





## Linear Classifiers

**Building blocks for large neural networks and convolutional networks**

**Parametric classifier, knowledge about the training data summarized into matrix W, and prior human knowledge(or redeem to imbalance) as b**

##### 1. Algerbra viewpoint: dot products of templates and pixel vector

<img src="C:\Users\26526\Desktop\9.7开学前\cs231n\image\QQ截图20200901193845.png" style="zoom:43%;" />

-   ==**each row of W corresponds to some template for one class of image**==, such as the first row stands for cat images’ template. ==$[0.2, -0.5, 0.1, 2.0]$ gives out the weight/influence of each pixel of the input image==. e.g.

    -   0.2: first pixel of a cat image contributes “0.2” to become as a cat
    -   -0.5: second pixel of a cat image contributes “-0.5” to become as a cat
    -   0.1: third pixel of a cat image contributes “0.1” to become as a cat
    -   2.0: forth pixel of a cat image contributes “2.0” to become as a cat, **which means that the lower right pixel of cat images are their key feature, that is to say, to become a cat image, the lower right pixel is most highly considered and has the highest weight**

    

-   **$Wx$ is data independence scaling:** computing dot product of the rows of W(each row represent a category) with input pixel vector, gives a similarity between each template for the class and the pixels of our image

-   the output vectors gives out the score for each category, which stands for **the level of similarity between each class and this given image**. We could choose the class that got the hightest score, probably most similar to input image. In CIFAR-10 we would get 10 scores, actually.

##### **2. Template Matching Idea: See what the dog template looks like (Visual Viewpoint)**

![](C:\Users\26526\Desktop\9.7开学前\cs231n\image\QQ截图20200901195038.png)

-   linear classifiers are actually **learning only one template for each class to recognize each category**, meaning that they would average out all the variations that a class could appear (**such as averaging every horse image so that the horse template seems to have two heads**, see above).
-   each visual template stands for **the  1 * 3072 pixel vector of each row in W unravel into 32 * 32 * 3 images**

<img src="C:\Users\26526\Desktop\9.7开学前\cs231n\image\QQ截图20200901195351.png" style="zoom:63%;" />

-   the **W’s**: template color matrix for each category/class.

##### 3. High Dimensional Space idea: Linear classifiers provide hyperplanes as linear separators

<img src="C:\Users\26526\Desktop\9.7开学前\cs231n\image\QQ截图20200901200419.png" style="zoom: 50%;" />

-   each training image is a point in this high dimensional space
-   linaer classifiers put linear decision boundaries to draw linear separation between one category and the rest of the categories

#### 4. Hard cases for linear classifiers (non-separable in low-dimensional space)

<img src="C:\Users\26526\Desktop\9.7开学前\cs231n\image\QQ截图20200901200817.png" style="zoom:60%;" />

-   Three modes: ==multi modal data==, such as horse images with (1) horse looking to the left (2) horse looking to the right. (1)(2) may become isloated islands (as picture-3 shown) in the pixel space.

##### 5. Summary: Tree ViewPoints

<img src="C:\Users\26526\Desktop\9.7开学前\cs231n\image\QQ截图20200901201651.png" style="zoom:48%;" />