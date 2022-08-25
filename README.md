<h1 align="center">CS224n: Assignment Solutions</h1>
<p align="center"><b>Natural Language Processing with Deep Learning</b></p>
<p align="center"><i>Stanford - Winter 2022</i></p>

## About

### Overview

These are my solutions for the **CS224n** course assignments offered by _Stanford University_ (Winter 2022). Written questions are explained in detail, the code is brief and commented (see examples below). From what I investigated, these should be the most explained solutions.

> Check out my solutions for **[CS231n](https://github.com/mantasu/cs231n)**. From what I've checked, they should be the shortest.

### Main sources (official)
* [**Course page**](http://web.stanford.edu/class/cs224n/index.html)
* [**Assignments**](http://web.stanford.edu/class/cs224n/index.html#schedule)
* [**Lecture videos** (2021)](https://www.youtube.com/playlist?list=PLoROMvodv4rOSH4v6133s9LFPRHjEmbmJ)

<br>

## Requirements
For **conda** users, the instructions on how to set-up the environment are given in the handouts. For `pip` users, I've gathered all the requirements in one [file](requirements.txt). Please set up the virtual environment and install the dependencies (for _linux_ users):

```shell
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

You can install everything with **conda** too (see [this](https://stackoverflow.com/questions/51042589/conda-version-pip-install-r-requirements-txt-target-lib)). For code that requires **Azure** _Virtual Machines_, I was able to run everything successfully on **Google Colab** with a free account.

> Note: Python 3.8 or newer should be used

<br>

## Solutions

### Structure

For every assignment, i.e., for directories `a1` through `a5`, there is coding and written parts. The `solutions.pdf` files are generated from latex directories where the provided templates were filled while completing the questions in `handout.pdf` files and the code.

### Assignments

* [A1](a1): Exploring Word Vectors (_Done_)
* [A2](a2): word2vec (_Done_)
* [A3](a3): Dependency Parsing (_Done_)
* [A4](a4): Neural Machine Translation with RNNs and Analyzing NMT Systems (_Done_)
* [A5](a5): Self-Attention, Transformers, and Pretraining (_Done_)

<br>

## Examples

<details><summary><b>Written (Attention Exploration)</b></summary>
<br>

**Question (b) ii.**

<hr>

<sub>
As before, let $v_a$ and $v_b$ be two value vectors corresponding to key vectors $k_a$ and $k_b$, respectively. Assume that <b>(1)</b> all key vectors are orthogonal, so $k_i^\top k_j = 0$ for all $i \neq j$; and <b>(2)</b> all key vectors have norm $1$ (recall that a vector $x$ has norm 1 iff $x^\top x = 1$). <b>Find an expression</b> for a query vector $q$ such that $c \approx \frac{1}{2}(v_a + v_b)$.<br>
<sub>
<b>Hint</b>: while the <i>softmax</i> function will never <i>exactly</i> average the two vectors, you can get close by using a large scalar multiple in the expression.
</sub></sub>

<hr>

<br>

**Answer**

<hr>

<sub>
Assume that $\mathbf{c}$ is approximated as follows:
</sub>

<sub>
$$\mathbf{c}\approx 0.5 \mathbf{v}_a + 0.5 \mathbf{v}_b$$
</sub>

<sub>
This means we want $\alpha_a\approx0.5$ and $\alpha_b\approx0.5$, which can be achieved when (whenever $i\ne a$ and $i\ne b$):
</sub>

<sub>
$$\mathbf{k}_a^{\top}\mathbf{q}\approx\mathbf{k}_b^{\top}\mathbf{q} \gg \mathbf{k}_i^{\top}\mathbf{q}$$
</sub>

<sub>
Like explained in the previous question, if the dot product is big, the probability mass will also be big and we want a balanced mass between $\alpha_a$ and $\alpha_b$. $\mathbf{q}$ will be largest for $\mathbf{k}_a$ and $\mathbf{k}_b$ when it is a large multiplicative of a vector that contains a component in $\mathbf{k}_a$ direction and in $\mathbf{k}_b$ direction:
</sub>

<sub>
$$\mathbf{q}=\beta(\mathbf{k}_a + \mathbf{k}_b),\quad\text{where } \beta \gg 0$$
</sub>

<sub>
Now, since the keys are orthogonal to each other, it is easy to see that:
</sub>

<sub>
$$\mathbf{k}_a^{\top}\mathbf{q}=\beta; \quad \mathbf{k}_b^{\top}\mathbf{q}=\beta; \quad \mathbf{k}_i^{\top}\mathbf{q}=0, \text{ whever }i\ne a\text{ and }i\ne b$$
</sub>

<sub>
Thus when we exponentiate, only $\exp(\beta)$ will matter, because $\exp(0)$ will be insignificant to the probability mass. We get that:
</sub>

<sub>
$$\alpha_a=\alpha_b=\frac{\exp(\beta)}{n-2 + 2\exp(\beta)}\approx\frac{\exp(\beta)}{2\exp(\beta)}\approx\frac{1}{2}, \text{ for }\beta \gg 0$$
</sub>

<hr>


</details>

<details><summary><b>Code (Negative Sampling)</b></summary>
<sub>

```python
def negSamplingLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset,
    K=10
):
    """ Negative sampling loss function for word2vec models

    Implement the negative sampling loss and gradients for a centerWordVec
    and a outsideWordIdx word vector as a building block for word2vec
    models. K is the number of negative samples to take.

    Note: The same word may be negatively sampled multiple times. For
    example if an outside word is sampled twice, you shall have to
    double count the gradient with respect to this word. Thrice if
    it was sampled three times, and so forth.

    Arguments/Return Specifications: same as naiveSoftmaxLossAndGradient
    """

    # Negative sampling of words is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    negSampleWordIndices = getNegativeSamples(outsideWordIdx, dataset, K)
    indices = [outsideWordIdx] + negSampleWordIndices

    ### YOUR CODE HERE (~10 Lines)

    ### Please use your implementation of sigmoid in here.

    # We will multiply where same words are involved, avoiding recalculations
    un, idx, n_reps = np.unique(indices, return_index=True, return_counts=True)
    U_concat = outsideVectors[un]
    
    # For convenience
    n_reps[idx==0] *= -1
    U_concat[idx!=0] *= -1
    S = sigmoid(centerWordVec @ U_concat.T)
    
    # Find loss and derivatives w.r.t. v_c, U
    loss = -(np.abs(n_reps) * np.log(S)).sum()
    gradCenterVec = np.abs(n_reps) * (1 - S) @ -U_concat
    gradOutsideVecs = np.zeros_like(outsideVectors)
    gradOutsideVecs[un] = n_reps[:, None] * np.outer(1 - S, centerWordVec)

    ### END YOUR CODE

    return loss, gradCenterVec, gradOutsideVecs
```

</sub>
</details>
