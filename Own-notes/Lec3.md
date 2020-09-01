## Loss Functiosn: $l(W)$

-   a loss function takes in W as input, produces a score for W, and then tells us how bad W is quantitatively.
-   searching through the space of all W’s to find the **W that minimizes the loss function on your training data**, this process invlove the procedure of optimization.

#### SVM Loss (For Multi-Classes)

<img src="C:\Users\26526\Desktop\GitHub\CS231n\image\QQ截图20200901221610.png" style="zoom:50%;" />

-   Given a set of data $\{x_i, y_i\}_{i=1}^{m}$, our learned classifier produces a *scores-vector* $s := f(x_i, W)$ for each sample $x_i$, which is giving predicted score for classes.

-   $L_i$ is the individual loss function for $x_i$, which sums over all the incorrect categories. 

    $s_{y_i}$ stands for the true label’s score, and $s_j$ stands for the incorrect label’s score. In our appreciation this should be bigger than the others, so 

    -    when $s_{y_i} \geq s_j + 1$, then no penalty
    -   when $s_{y_i} < s_j + 1$, the more $s_{y_i}$ is smaller than the others, the bigger the penalty will be
    -   here “1” is called safety margin.
        -   Actually, this **1** can be arbitary any positive value; cause we only care that the correct score is much greater than the incorrect score, and the absolute diff between them doesn’t matter.
        -   If rescale the matrix W up and down, then the scores would rescale correspondingly
        -   **1** get washed out and canceled out with the scaling operation of W

-   ==Sum of hinge losses==

    <img src="C:\Users\26526\Desktop\GitHub\CS231n\image\QQ截图20200901222949.png" style="zoom:50%;" />

-   Loss over full dataset is average:

    <img src="C:\Users\26526\Desktop\GitHub\CS231n\image\QQ截图20200901223643.png" style="zoom:50%;" />

    L is a quantitative measure that reveals how bad our classifier perform on one data-set.