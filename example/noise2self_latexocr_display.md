# Noise2Self: Blind Denoising by Self-Supervision

*Joshua Batson * 1 Loic Royer * 1*

## Contents
- [Abstract](#abstract)
- [1. Introduction](#1-introduction)
- [2. Related Work](#2-related-work)
- [3. Calibrating Traditional Models](#3-calibrating-traditional-models)
- [3.1. Single-Cell](#31-single-cell)
- [3.2. PCA](#32-pca)
- [4. Theory](#4-theory)
- [4.1. How good is the optimum?](#41-how-good-is-the-optimum)
- [4.2. Doing better](#42-doing-better)
- [5. Deep Learning Denoisers](#5-deep-learning-denoisers)
- [6. Discussion](#6-discussion)
- [Acknowledgements](#acknowledgements)
- [References](#references)
- [1. Notation](#1-notation)
- [2. Gaussian Processes](#2-gaussian-processes)
- [3.1. Uniform Pixel Selection](#31-uniform-pixel-selection)
- [3. Masking](#3-masking)
- [4. Calibrating Traditional Denoising Methods](#4-calibrating-traditional-denoising-methods)
- [3.2. Linear Combinations](#32-linear-combinations)
- [5. Neural Net Examples](#5-neural-net-examples)
- [5.1. Datasets: Ha`nz`ı, CellNet, ImageNet](#51-datasets-hanzı-cellnet-imagenet)
- [5.2. Architecture](#52-architecture)
- [5.3. Training](#53-training)
- [6. Single-Cell Gene Expression](#6-single-cell-gene-expression)
- [5.4. Inference](#54-inference)
- [5.5. Evaluation](#55-evaluation)

## Abstract

We propose a general framework for denoising high-dimensional measurements which requires no prior on the signal, no estimate of the noise, and no clean training data. The only assumption is that the noise exhibits statistical independence across different dimensions of the measurement, while the true signal exhibits some correlation. For a broad class of functions (“J -invariant”), it is then possible to estimate the performance of a denoiser from noisy data alone. This allows us to calibrate J -invariant versions of any parameterised denoising algorithm, from the single hyperparameter of a median ﬁlter to the millions of weights of a deep neural network. We demonstrate this on natural image and microscopy data, where we exploit noise independence between pixels, and on single-cell gene expression data, where we exploit independence between detections of individual molecules. This framework generalizes recent work on training neural nets from noisy images and on cross-validation for matrix factorization.

## 1. Introduction

We would often like to reconstruct a signal from highdimensional measurements that are corrupted, undersampled, or otherwise noisy. Devices like high-resolution cameras, electron microscopes, and DNA sequencers are capable of producing measurements in the thousands to millions of feature dimensions. But when these devices are pushed to their limits, taking videos with ultra-fast frame rates at very low-illumination, probing individual molecules with electron microscopes, or sequencing tens of thousands of cells simultaneously, each individual feature can become quite noisy. Nevertheless, the objects being studied are often very structured and the values of different features are

highly correlated. Speaking loosely, if the “latent dimension” of the space of objects under study is much lower than the dimension of the measurement, it may be possible to implicitly learn that structure, denoise the measurements, and recover the signal without any prior knowledge of the signal or the noise. Traditional denoising methods each exploit a property of the noise, such as Gaussianity, or structure in the signal, such as spatiotemporal smoothness, self-similarity, or having low-rank. The performance of these methods is limited by the accuracy of their assumptions. For example, if the data are genuinely not low rank, then a low rank model will ﬁt it poorly. This requires prior knowledge of the signal structure, which limits application to new domains and modalities. These methods also require calibration, as hyperparameters such as the degree of smoothness, the scale of self-similarity, or the rank of a matrix have dramatic impacts on performance. In contrast, a data-driven prior, such as pairs (xi, yi) of noisy and clean measurements of the same target, can be used to set up a supervised learning problem. A neural net trained to predict yi from xi may be used to denoise new noisy measurements (Weigert et al., 2018). As long as the new data are drawn from the same distribution, one can expect performance similar to that observed during training. Lehtinen et al. demonstrated that clean targets are unnecessary (2018). A neural net trained on pairs (xi, xi) of independent noisy measurements of the same target will, under certain distributional assumptions, learn to predict the clean signal. These supervised approaches extend to image denoising the success of convolutional neural nets, which currently give state-of-the-art performance for a vast range of image-to-image tasks. Both of these methods require an experimental setup in which each target may be measured multiple times, which can be difﬁcult in practice. In this paper, we propose a framework for blind denoising based on self-supervision. We use groups of features whose noise is independent conditional on the true signal to predict one another. This allows us to learn denoising functions from single noisy measurements of each object, with performance close to that of supervised methods. The same approach can also be used to calibrate traditional image denoising methods such as median ﬁlters and non-local means,

a b

independent feature dimensions images

**Figure 1. (a) The box represents the dimensions of the measurement x. J is a subset of the dimensions, and f is a J-invariant function: it**

has the property that the value of f (x) restricted to dimensions in J, f (x)J , does not depend on the value of x restricted to J, xJ . This enables self-supervision when the noise in the data is conditionally independent between sets of dimensions. Here are 3 examples of dimension partitioning: (b) two independent image acquisitions, (c) independent pixels of a single image, (d) independently detected RNA molecules from a single cell.

and, using a different independence structure, denoise highly under-sampled single-cell gene expression data. We model the signal y and its noisy measurement x as a pair of random variables in Rm. If J ⊂ {1, . . . , m} is a subset of the dimensions, we write xJ for x restricted to J. Deﬁnition. Let J be a partition of the dimensions {1, . . . , m} and let J ∈ J . A function f : Rm → Rm is J-invariant if f (x)J does not depend on the value of xJ . It is J -invariant if it is J-invariant for each J ∈ J .

We propose minimizing the self-supervised loss

$$ L(f ) = E f (x) - x 2 \tag{1} $$

over J -invariant functions f . Since f has to use information from outside of each subset of dimensions J to predict the values inside of J, it cannot merely be the identity. Proposition 1. Suppose x is an unbiased estimator of y, i.e. E[x|y] = y, and the noise in each subset J ∈ J is independent from the noise in its complement Jc, conditional on y. Let f be J -invariant. Then

$$ E f (x) - x 2 = E f (x) - y 2 + E x - y 2 . \tag{2} $$

That is, the self-supervised loss is the sum of the ordinary supervised loss and the variance of the noise. By minimizing the self-supervised loss over a class of J -invariant functions, one may ﬁnd the optimal denoiser for a given dataset. For example, if the signal is an image with independent, mean-zero noise in each pixel, we may choose J = {{1}, . . . , {m}} to be the singletons of each coordinate. Then “donut” median ﬁlters, with a hole in the center, form a class of J -invariant functions, and by comparing the value of the self-supervised loss at different ﬁlter radii, we are able to select the optimal radius for denoising the image at hand (See §3). The donut median ﬁlter has just one parameter and therefore limited ability to adapt to the data. At the other extreme,

c d ACCT...TGAG ACTG...TGAC TAGC...CTCA ATAT...CGTC TTAG...GAGC CGCA...ACAC ACCT...GGTT ACCT...TGAC ACCG...TGTA GCGT...CGAC ACCT...GATC TTCG...AGAT CGCT...GTGT ACAT...GAGG pixels molecules

we may search over all J -invariant functions for the global optimum: Proposition 2. The -invariant function f∗ minimizing (1) J J satisﬁes ∗ E[yJ |xJc ] fJ (x)J = for each subset J ∈ J .

That is, the optimal J -invariant predictor for the dimensions of y in some J ∈ J is their expected value conditional on observing the dimensions of x outside of J. In §4, we use analytical examples to illustrate how the optimal J -invariant denoising function approaches the optimal general denoising function as the amount of correlation between features in the data increases. In practice, we may attempt to approximate the optimal denoiser by searching over a very large class of functions, such as deep neural networks with millions of parameters. In §5, we show that a deep convolutional network, modiﬁed to become J -invariant using a masking procedure, can achieve state-of-the-art blind denoising performance on three diverse datasets. Sample code is available on GitHub1 and deferred proofs are contained in the Supplement.

## 2. Related Work

Each approach to blind denoising relies on assumptions about the structure of the signal and/or the noise. We review the major categories of assumption below, and the traditional and modern methods that utilize them. Most of the methods below are described in terms of application to image denoising, which has the richest literature, but some have natural extensions to other spatiotemporal signals and to generic measurements of vectors. Smoothness: Natural images and other spatiotemporal signals are often assumed to vary smoothly (Buades et al.,

1https://github.com/czbiohub/noise2self

2005b). Local averaging, using a Gaussian, median, or some other ﬁlter, is a simple way to smooth out a noisy input. The degree of smoothing to use, e.g., the width of a ﬁlter, is a hyperparameter often tuned by visual inspection. Self-Similarity: Natural images are often self-similar, in that each patch in an image is similar to many other patches from the same image. The classic non-local means algorithm replaces the center pixel of each patch with a weighted average of central pixels from similar patches (Buades et al., 2005a). The more robust BM3D algorithm makes stacks of similar patches, and performs thresholding in frequency space (Dabov et al., 2007). The hyperparameters of these methods have a large effect on performance (Lebrun, 2012), and on a new dataset with an unknown noise distribution it is difﬁcult to evaluate their effects in a principled way. Convolutional neural nets can produce images with another form of self-similarity, as linear combinations of the same small ﬁlters are used to produce each output. The “deep image prior” of (Ulyanov et al., 2017) exploits this by training a generative CNN to produce a single output image and stopping training before the net ﬁts the noise. Generative: Given a differentiable, generative model of the data, e.g. a neural net G trained using a generative adversarial loss, data can be denoised through projection onto the range of the net (Tripathi et al., 2018). Gaussianity: Recent work (Zhussip et al., 2018; Metzler et al., 2018) uses a loss based on Stein’s unbiased risk estimator to train denoising neural nets in the special case that noise is i.i.d. Gaussian. Sparsity: Natural images are often close to sparse in e.g. a wavelet or DCT basis (Chang et al., 2000). Compression algorithms such as JPEG exploit this feature by thresholding small transform coefﬁcients (Pennebaker & Mitchell, 1992). This is also a denoising strategy, but artifacts familiar from poor compression (like the ringing around sharp edges) may occur. Hyperparameters include the choice of basis and the degree of thresholding. Other methods learn an overcomplete dictionary from the data and seek sparsity in that basis (Elad & Aharon, 2006; Papyan et al., 2017). Compressibility: A generic approach to denoising is to lossily compress and then decompress the data. The accuracy of this approach depends on the applicability of the compression scheme used to the signal at hand and its robustness to the form of noise. It also depends on choosing the degree of compression correctly: too much will lose important features of the signal, too little will preserve all of the noise. For the sparsity methods, this “knob” is the degree of sparsity, while for low-rank matrix factorizations, it is the rank of the matrix. Autoencoder architectures for neural nets provide a gen-

eral framework for learnable compression. Each sample is mapped to a low-dimensional representation—the value of the neural net at the bottleneck layer— then back to the original space (Gallinari et al., 1987; Vincent et al., 2010). An autoencoder trained on noisy data may produce cleaner data as its output. The degree of compression is determined by the width of the bottleneck layer. UNet architectures, in which skip connections are added to a typical autoencoder architecture, can capture high-level spatially coarse representations and also reproduce ﬁne detail; they can, in particular, learn the identity function (Ronneberger et al., 2015). Trained directly on noisy data, they will do no denoising. Trained with clean targets, they can learn very accurate denoising functions (Weigert et al., 2018). Statistical Independence: Lehtinen et al. observed that a UNet trained to predict one noisy measurement of a signal from an independent noisy measurement of the same signal will in fact learn to predict the true signal (Lehtinen et al., 2018). We may reformulate the Noise2Noise procedure in terms of J -invariant functions: if x1 = y + n1 and x2 = y + n2 are the two measurements, we consider the composite measurement x = (x1, x2) of a composite signal (y, y) in R2m and set J = {J1, J2} = {{1, . . . , m}, {m + 1, . . . , 2m}}. Then f∗ E[y|x1]. (x)J2 = J An extension to video, in which one frame is used to compute the pullback under optical ﬂow of another, was explored in (Ehret et al., 2018). In concurrent work, Krull et al. train a UNet to predict a collection of held-out pixels of an image from a version of that image with those pixels replaced (2018). A key difference between their approach and our neural net examples in §5 is in that their replacement strategy is not quite J -invariant. (With some probability a given pixel is replaced by itself.) While their method lacks a theoretical guarantee against ﬁtting the noise, it performs well in practice, on natural and microscopy images with synthetic and real noise. Finally, we note that the “fully emphasized denoising autoencoders” in (Vincent et al., 2010) used the MSE between an autoencoder evaluated on masked input data and the true value of the masked pixels, but with the goal of learning robust representations, not denoising.

## 3. Calibrating Traditional Models

Many denoising models have a hyperparameter controlling the degree of the denoising—the size of a ﬁlter, the threshold for sparsity, the number of principal components. If ground truth data were available, the optimal parameter θ for a family of denoisers fθ could be chosen by minimizing fθ(x) − y 2. Without ground truth, we may nevertheless

noisy self-supervised donut classic

donut classic

ground truth donut r classic

Radius of median ﬁlter

r=1 r=2 r=3 r=4 r=5 r=6

more noisy more blurry

**Figure 2. Calibrating a median ﬁlter without ground truth. Different median ﬁlters may be obtained by varying the ﬁlter’s radius. Which is**

optimal for a given image? The optimal parameter for J -invariant functions such as the donut median can be read off (red arrows) from the self-supervised loss.

compute the self-supervised loss fθ(x) − x 2. For general More generally, let gθ be any classical denoiser, and let J be fθ, it is unrelated to the ground truth loss, but if fθ is J any partition of the pixels such that neighboring pixels are invariant, then it is equal to the ground truth loss plus the in different subsets. Let s(x) be the function replacing each noise variance (Eqn. 2), and will have the same minimizer. pixel with the average of its neighbors. Then the function fθ deﬁned by In Figure 2, we compare both losses for the median ﬁlter gr, which replaces each pixel with the median over a disk (3) fθ(x)J := gθ(1J · s(x) + 1Jc · x)J , of radius r surrounding it, and the “donut” median ﬁlter fr, which replaces each pixel with the median over the same for each J ∈ J , is a J -invariant version of gθ. Indeed, disk excluding the center, on an image with i.i.d. Gaussian since the pixels of x in J are replaced before applying gθ, noise. For J = {{1}, . . . , {m}} the partition into single the output cannot depend on xJ . pixels, the donut median is J -invariant. For the donut median, the minimum of the self-supervised loss fr(x) − x 2 In Supp. Figure 1, we show the corresponding loss curves for J -invariant versions of a wavelet ﬁlter, where we tune (solid blue) sits directly above the minimum of the ground truth loss fr(x) − y 2 (dashed blue), and selects the opthe threshold σ, and NL-means, where we tune a cut-off timal radius r = 3. The vertical displacement is equal to distance h (Buades et al., 2005a; Chang et al., 2000; van der the variance of the noise. In contrast, the self-supervised Walt et al., 2014). The partition J used is a 4x4 grid. Note loss gr(x) − x 2 (solid orange) is strictly increasing and that in all these examples, the function fθ is genuinely differtells us nothing about the ground truth loss gr(x) − y 2 ent than gθ, and, because the simple interpolation procedure may itself be helpful, it sometimes performs better. (dashed orange). Note that the median and donut median are genuinely different functions with slightly different perfor- In Table 1, we compare all three J -invariant denoisers on a mance, but while the former can only be tuned by inspecting single image. As expected, the denoiser with the best selfthe output images, the latter can be tuned using a principled supervised loss also has the best performance as measured loss. by Peak Signal to Noise Ratio (PSNR).

**Table 1. Comparison of optimally tuned J -invariant versions of**

classical denoising models. Performance is better than original method at default parameter values, and can be further improved (+) by adding an optimal amount of the noisy input to the J invariant output (§4.2).

METHOD LOSS PSNR J-INVT J-INVT J-INVT+ DEFAULT MEDIAN 0.0107 27.5 28.2 27.1 WAVELET 0.0113 26.0 26.9 24.6 NL-MEANS 0.0098 30.4 30.8 28.9

### 3.1. Single-Cell

In single-cell transcriptomic experiments, thousands of individual cells are isolated, lysed, and their mRNA are extracted, barcoded, and sequenced. Each mRNA molecule is mapped to a gene, and that ∼20,000-dimensional vector of counts is an approximation to the gene expression of that cell. In modern, highly parallel experiments, only a few thousand of the hundreds of thousands of mRNA molecules present in a cell are successfully captured and sequenced (Milo et al., 2010). Thus the expression vectors are very undersampled, and genes expressed at low levels will appear as zeros. This makes simple relationships among genes, such as co-expression or transitions during development, difﬁcult to see. If we think of the measurement as a set of molecules captured from a given cell, then we may partition the molecules at random into two sets J1 and J2. Summing (and normalizing) the gene counts in each set produces expression vectors xJ1 and xJ2 which are independent conditional on the true mRNA content y. We may now attempt to denoise x by training a model to predict xJ2 from xJ1 and vice versa. We demonstrate this on a dataset of 2730 bone marrow cells from Paul et al. using principal component regression (Paul et al., 2015), where we use the self-supervised loss to ﬁnd an optimal number of principal components. The data contain a population of stem cells which differentiate either into erythroid or myeloid lineages. The expression of genes preferentially expressed in each of these cell types is shown in Figure 3 for both the (normalized) noisy data and data denoised with too many, too few, and an optimal number of principal components. In the raw data, it is difﬁcult to discern any population structure. When the data is under-corrected, the stem cell marker Iﬁtm1 is still not visible. When it is over-corrected, the stem population appears to express substantial amounts of Klf1 and Mpo. In the optimally corrected version, Iﬁtm1 expression coincides with low expression of the other markers, identifying the stem population, and its transition to the two more mature states is easy to see.

raw b a Iﬁtm1 4 under-corrected

Mpo 0 Mpo c over-corrected d optimal erythroid cells stem cells myeloid cells 0 Iﬁtm1 Mpo 0 Mpo e

number of principal components

**Figure 3. Self-supervised loss calibrates a linear denoiser for single**

cell data. (a) Raw expression of three genes: a myeloid cell marker (Mpo), an erythroid cell marker (Klf1), and a stem cell marker (Iﬁtm1). Each point corresponds to a cell. (e) Self-supervised loss for principal component regression. In (d) we show the the denoised data for the optimal number of principal components (17, red arrow). In (c) we show the result of using too few components and in (b) that of using too many. X-axes show square-root normalised counts.

### 3.2. PCA

Cross-validation for choosing the rank of a PCA requires some care, since adding more principal components will always produce a better ﬁt, even on held-out samples (Bro et al., 2008). Owen and Perry recommend splitting the feature dimensions into two sets J1 and J2 as well as splitting the samples into train and validation sets (Owen & Perry, 2009). For a given k, they ﬁt a rank k principal component regression fk : Xtrain,J1 → Xtrain,J2 and evaluate its predictions on the validation set, computing fk(Xvalid,J1 ) − Xvalid,J2 2. They repeat this, permuting train and validation sets and J1 and J2. Simulations show that if X is actually a sum of a low-rank matrix plus Gaussian noise, then the k minimizing the total validation loss is often the optimal choice (Owen & Perry, 2009; Owen

& Wang, 2016). This calculation corresponds to using the a b self-supervised loss to train and cross-validate a {J1, J2}invariant principal component regression.

## 4. Theory

In an ideal situation for signal reconstruction, we have a prior p(y) for the signal and a probabilistic model of the noisy measurement process p(x|y). After observing some measurement x, the posterior distribution for y is given by Bayes’ rule:

p(y|x) = p(x|y)p(y) . p(x|y)p(y)dy 1 pixel 2 pixels 3 pixels In practice, one seeks some function f (x) approximating a relevant statistic of y|x, such as its mean or median. The optimal optimal J-invariant mean is provided by the function minimizing the loss: 1.0 length scale (pixels) 3.0 Ex f (x) − y 2

**Figure 4. The optimal J -invariant predictor converges to the optimal predictor. Example images for Gaussian processes of different**

(The L1 norm would produce the median) (Murphy, 2012). length scales. The gap in image quality between the two predictors Fix a partition J of the dimensions {1, . . . , n} of x and tends to zero as the length scale increases. suppose that for each J ∈ J , we have

p(x|y) = p(xJ |y)p(xJc |y),

**Figure 4 illustrates this phenomenon for the example of**

Gaussian Processes, a computationally tractable model of i.e., xJ and xJc are independent conditional on y. We signals with correlated features. We consider a process on consider the loss a 33 × 33 toroidal grid. The value of y at each node is standard normal and the correlation between the values at Ex f (x) − x 2 = Ex,y f (x) − y 2 + x − y 2 p and q depends on the distance between them: Kp,q = − 2 f (x) − y, x − y . exp(− p − q 2 /2 2), where is the length scale. The noisy measurement x = y + n, where n is white Gaussian If f is J -invariant, then for each j the random variables noise with standard deviation 0.5. f (x)j|y and xj|y are independent. The third term reduces to j Ey(Ex|y[f (x)j − yj])(Ex|y[xj − yj]), which vanishes While when E[x|y] = y. This proves Prop. 1. E y − fJ (x) ≥ E y − E[y|x] ∗ Any J -invariant function can be written as a collection of for all , the gap decreases quickly as the length scale inordinary functions fJ : R|Jc| → R|J|, where we separate creases. the output dimensions of f based on which input dimensions The Gaussian process is more than a convenient example; it they depend on. Then actually represents a worst case for the recovery error as a L(f ) = E fJ (xJc ) − xJ 2 . function of correlation. J ∈J Proposition 3. Let x, y be random variables and let xG and yG be Gaussian random variables with the same covariance This is minimized at fJ∗,G matrix. Let f∗ and be the corresponding optimal J- J ∗ E[xJ E[yJ f ( x ) = |xJc ] = |xJc ]. invariant predictors. Then c J J We bundle these functions into f∗ , proving Prop. 2. ∗ ∗,G E y − fJ (x) ≤ E y − fJ (x) . J

### 4.1. How good is the optimum?

Proof. See Supplement. How much information do we lose by giving up xJ when trying to predict yJ ? Roughly speaking, the more the fea- Gaussian processes represent a kind of local texture with no higher structure, and the functions fJ∗,G turn out to be linear tures in J are correlated with those outside of it, the closer f∗ will be to E[y|x] and the better both will estimate y. (Murphy, 2012). (x) J

Alphabet

Gaussian Process of same covariance

noisy clean optimally denoised

**Figure 5. For any dataset, the error of the optimal predictor (blue) is lower than that for a Gaussian Process (red) with the same covariance**

matrix. We show this for a dataset of noisy digits: the quality of the denoising is visibly better for the Alphabet than the Gaussian Process (samples at σ = 0.8).

At the other extreme is data drawn from ﬁnite collection of templates, like symbols in an alphabet. If the alphabet consists of {a1, . . . , ar} ∈ Rm and the noise is i.i.d. mean-zero Gaussian with variance σ2, then the optimal J-invariant prediction independent is a weighted sum of the letters from the alphabet. The weights wi = exp(− (ai − x) · 1Jc 2 /2σ2) are proportional to the posterior probabilities of each letter. When the noise is low, the output concentrates on a copy of the closest letter; when the noise is high, the output averages many letters. In Figure 5, we demonstrate this phenomenon for an alphabet consisting of 30 16x16 handwritten digits drawn from MNIST (LeCun et al., 1998). Note that almost exact recovery is possible at much higher levels of noise than the Gaussian process with covariance matrix given by the empirical covariance matrix of the alphabet. Any real-world dataset will exhibit more structure than a Gaussian process, so nonlinear functions can generate signiﬁcantly better predictions.

### 4.2. Doing better

If f is J -invariant, then by deﬁnition f (x)j contains no information from xj, and the right linear combination λf (x)j + (1 − λ)xj will produce an estimate of yj with lower variance than either. The optimal value of λ is given by the variance of the noise divided by the value of the self-supervised loss. The performance gain depends on the quality of f : for example, if f improves the PSNR by 10 dB, then mixing in the optimal amount of x will yield another 0.4 dB. (See Table 1 for an example and Supplement for proofs.)

## 5. Deep Learning Denoisers

The self-supervised loss can be used to train a deep convolutional neural net with just one noisy sample of each image in

Gaussian Process 0.03 Alphabet

0.01

Noise standard deviation 0.2 0.4 0.6 0.8 1.0 1.2 1.4

a dataset. We show this on three datasets from different domains (see Figure 6) with strong and varied heteroscedastic synthetic noise applied independently to each pixel. For the datasets Ha`nz`ı and ImageNet we use a mixture of Poisson, Gaussian, and Bernoulli noise. For the CellNet microscopy dataset we simulate realistic sCMOS camera noise. We use a random partition of 25 subsets for J , and we make the neural net J -invariant as in Eq. 3, except we replace the masked pixels with random values instead of local averages. We train two neural net architectures, a UNet and a purely convolutional net, DnCNN (Zhang et al., 2017). To accelerate training, we only compute the net outputs and loss for one partition J ∈ J per minibatch. As shown in Table 2, both neural nets trained with selfsupervision (Noise2Self) achieve superior performance to the classic unsupervised denoisers NLM and BM3D (at default parameter values), and comparable performance to the same neural net architectures trained with clean targets (Noise2Truth) and with independently noisy targets (Noise2Noise). The result of training is a neural net gθ, which, when converted into a J -invariant function fθ, has low selfsupervised loss. We found that applying gθ directly to the noisy input gave slightly better (0.5 dB) performance than using fθ. The images in Figure 6 use gθ. Remarkably, it is also possible to train a deep CNN to denoise a single noisy image. The DnCNN architecture, with 560,000 parameters, trained with self-supervision on the noisy camera image from §3, with 260,000 pixels, achieves a PSNR of 31.2.

## 6. Discussion

We have demonstrated a general framework for denoising high-dimensional measurements whose noise exhibits some conditional independence structure. We have shown how

noisy NLM BM3D N2S (UNet) N2S (DnCNN) N2N (UNet) N2T (DnCNN)

**Figure 6. Performance of classic, supervised, and self-supervised denoising methods on natural images, Chinese characters, and ﬂuorescence microscopy images. Blind denoisers are NLM, BM3D, and neural nets (UNet and DnCNN) trained with self-supervision (N2S).**

We compare to neural nets supervised with a second noisy image (N2N) and with the ground truth (N2T).

to use a self-supervised loss to calibrate or train any J invariant class of denoising functions. There remain many open questions about the optimal choice of partition J for a given problem. The structure of J must reﬂect the patterns of dependence in the signal and independence in the noise. The relative sizes of each subset J ∈ J and its complement creates a bias-variance tradeoff in the loss, exchanging information used to make a prediction for information about the quality of that prediction. For example, the measurements of single-cell gene expression could be partitioned by molecule, gene, or even pathway, reﬂecting different assumptions about the kind of stochasticity occurring in transcription. We hope this framework will ﬁnd application to other domains, such as sensor networks in agriculture or geology, time series of whole brain neuronal activity, or telescope observations of distant celestial bodies.

true

**Table 2. Performance of different denoising methods by Peak Signal to Noise Ratio (PSNR) on held-out test data. Error bars for**

CNNs from training ﬁve models.

HA` NZ`I METHOD IMAGENET CELLNET RAW 6.5 9.4 15.1 NLM 8.4 15.7 29.0 BM3D 11.8 17.8 31.4 UNET (N2S) 13.8 ± 0.3 18.6 32.8 ± 0.2 DNCNN (N2S) 13.4 ± 0.3 18.7 33.7 ± 0.2 UNET (N2N) 13.3 ± 0.5 17.8 34.4 ± 0.1 DNCNN (N2N) 13.6 ± 0.2 18.8 34.4 ± 0.1 UNET (N2T) 13.1 ± 0.7 21.1 34.5 ± 0.1 DNCNN (N2T) 13.9 ± 0.6 22.0 34.4 ± 0.4

## Acknowledgements

Thank you to James Webber, Jeremy Freeman, David Dynerman, Nicholas Sofroniew, Jaakko Lehtinen, Jenny Folkesson, Anitha Krishnan, and Vedran Hadziosmanovic for valuable conversations. Thank you to Jack Kamm for discussions on Gaussian Processes and shrinkage estimators. Thank you to Martin Weigert for his help running BM3D. Thank you to the referees for suggesting valuable clariﬁcations. Thank you to the Chan Zuckerberg Biohub for ﬁnancial support.

## References

Bro, R., Kjeldahl, K., Smilde, A. K., and Kiers, H. A. L. Cross-validation of component models: A critical look at current methods. Analytical and Bioanalytical Chemistry, 390(5):1241–1251, March 2008.

Buades, A., Coll, B., and Morel, J.-M. A non-local algorithm for image denoising. In 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR’05), volume 2, pp. 60–65. IEEE, 2005a.

Buades, A., Coll, B., and Morel, J.-M. A review of image denoising algorithms, with a new one. Multiscale Modeling & Simulation, 4(2):490–530, 2005b.

Chang, S. G., Yu, B., and Vetterli, M. Adaptive wavelet thresholding for image denoising and compression. IEEE transactions on image processing, 9(9):1532–1546, 2000.

Dabov, K., Foi, A., Katkovnik, V., and Egiazarian, K. Image denoising by sparse 3-D transform-domain collaborative ﬁltering. IEEE Transactions on Image Processing, 16(8): 2080–2095, August 2007.

Ehret, T., Davy, A., Facciolo, G., Morel, J.-M., and Arias, P. Model-blind video denoising via frame-to-frame training.

Elad, M. and Aharon, M. Image denoising via sparse and redundant representations over learned dictionaries. IEEE Transactions on Image Processing, 15(12):3736–3745, December 2006.

Gallinari, P., Lecun, Y., Thiria, S., and Soulie, F. Memoires associatives distribuees: Une comparaison (Distributed associative memories: A comparison). Proceedings of COGNITIVA 87, Paris, La Villette, May 1987, 1987.

Krull, A., Buchholz, T.-O., and Jug, F. Noise2Void - learning denoising from single noisy images.

Lebrun, M. An analysis and implementation of the BM3D image denoising method. Image Processing On Line, 2: 175–213, August 2012.

LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. Gradientbased learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.

Lehtinen, J., Munkberg, J., Hasselgren, J., Laine, S., Karras, T., Aittala, M., and Aila, T. Noise2Noise: Learning image restoration without clean data. In International Conference on Machine Learning, pp. 2971–2980, 2018.

Ljosa, V., Sokolnicki, K. L., and Carpenter, A. E. Annotated high-throughput microscopy image sets for validation. Nature Methods, 9(7):637–637, July 2012.

Metzler, C. A., Mousavi, A., Heckel, R., and Baraniuk, R. G. Unsupervised learning with Stein’s unbiased risk estimator. arXiv:1805.10531 [cs, stat], May 2018.

Milo, R., Jorgensen, P., Moran, U., Weber, G., and Springer, M. BioNumbers – the database of key numbers in molecular and cell biology. Nucleic Acids Research, 38(suppl 1): D750–D753, January 2010.

Murphy, K. P. Machine Learning: a Probabilistic Perspective. Adaptive computation and machine learning series. MIT Press, Cambridge, MA, 2012. ISBN 978-0-262- 01802-9.

Owen, A. B. and Perry, P. O. Bi-cross-validation of the SVD and the nonnegative matrix factorization. The Annals of Applied Statistics, 3(2):564–594, June 2009.

Owen, A. B. and Wang, J. Bi-cross-validation for factor analysis. Statistical Science, 31(1):119–139, 2016.

Papyan, V., Romano, Y., Sulam, J., and Elad, M. Convolutional dictionary learning via local processing.

Paszke, A., Gross, S., Chintala, S., Chanan, G., Yang, E., DeVito, Z., Lin, Z., Desmaison, A., Antiga, L., and Lerer, A. Automatic differentiation in PyTorch. In NIPS-W, 2017.

Paul, F., Arkin, Y., Giladi, A., Jaitin, D., Kenigsberg, E., Keren-Shaul, H., Winter, D., Lara-Astiaso, D., Gury, M., Weiner, A., David, E., Cohen, N., Lauridsen, F., Haas, S., Schlitzer, A., Mildner, A., Ginhoux, F., Jung, S., Trumpp, A., Porse, B., Tanay, A., and Amit, I. Transcriptional heterogeneity and lineage commitment in myeloid progenitors. Cell, 163(7):1663–1677, December 2015.

Pennebaker, W. B. and Mitchell, J. L. JPEG still image data compression standard. Van Nostrand Reinhold, New York, 1992. ISBN 978-0-442-01272-4.

Ronneberger, O., Fischer, P., and Brox, T. U-Net: Convolutional networks for biomedical image segmentation.

Tripathi, S., Lipton, Z. C., and Nguyen, T. Q. Correction by projection: Denoising images with generative adversarial networks. arXiv:1803.04477 [cs], March 2018.

Ulyanov, D., Vedaldi, A., and Lempitsky, V. Deep image prior. arXiv:1711.10925 [cs, stat], November 2017.

van der Walt, S., Schnberger, J. L., Nunez-Iglesias, J., Boulogne, F., Warner, J. D., Yager, N., Gouillart, E., Yu, T., and contributors, t. s.-i. scikit-image: image processing in Python. PeerJ, 2:e453, 2014.

van Dijk, D., Sharma, R., Nainys, J., Yim, K., Kathail, P., Carr, A. J., Burdziak, C., Moon, K. R., Chaffer, C. L., Pattabiraman, D., Bierie, B., Mazutis, L., Wolf, G., Krishnaswamy, S., and Peer, D. Recovering gene interactions from single-cell data using data diffusion. Cell, 174(3): 716–729.e27, July 2018.

Vincent, P., Larochelle, H., Lajoie, I., Bengio, Y., and Manzagol, P.-A. Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion. Journal of machine learning research, 11(Dec):3371–3408, 2010.

Weigert, M., Schmidt, U., Boothe, T., Mller, A., Dibrov, A., Jain, A., Wilhelm, B., Schmidt, D., Broaddus, C., Culley, S., Rocha-Martins, M., Segovia-Miranda, F., Norden, C., Henriques, R., Zerial, M., Solimena, M., Rink, J., Tomancak, P., Royer, L., Jug, F., and Myers, E. W. Content-Aware image restoration: Pushing the limits of ﬂuorescence microscopy. July 2018.

Zhang, K., Zuo, W., Chen, Y., Meng, D., and Zhang, L. Beyond a Gaussian denoiser: Residual learning of deep CNN for image denoising. IEEE Transactions on Image Processing, 26(7):3142–3155, July 2017.

Zhussip, M., Soltanayev, S., and Chun, S. Y. Training deep learning based image denoisers from undersampled measurements without ground truth and without image prior. arXiv:1806.00961 [cs], June 2018.

Supplement to Noise2Self: Blind Denoising by Self-Supervision

## 1. Notation

For a variables x ∈ Rm and J ⊂ {1, . . . , m}, we write xJ for the restriction of x to the coordinates in J and xJc for the restriction of x to the coordinates in Jc. If f : Rm → Rm is a function, we write f (x)J for the restriction of f (x) to the coordinates in J. A partition J of a set X is a set of disjoint subsets of X whose union is all of X. When J = {j} is a singleton, we write x−j for xJc , the restriction of x to the coordinates not equal to j.

## 2. Gaussian Processes

Let x and y be random variables. Then the estimator of y from x minimizing the expected mean-square error (MSE) is x → E[y|x]. The expected MSE of that estimator is simply the variance of y|x:

Ex y − E[y|x] 2 = Ex Var(y|x).

If x and y are jointly multivariate normal, then the righthand-side depends only on the covariance matrix Σ. If

Σ = Σxx Σyx Σxy Σyy ,

then then right-hand-side is in fact a constant independent of x:

−1 Var(y|x) = Σyy − ΣyxΣxx Σxy.

(See Chapter 4 of (Murphy, 2012).) Lemma 1. Let Σ be a symmetric, positive semi-deﬁnite matrix with block structure

Σ = Σ11 Σ12 Σ21 Σ22 .

Then −1 Σ11 Σ12Σ22 Σ21.

Proof. Since Σ is PSD, we may factorize it as a product XT X for some matrix X. (For example, take the spectral decomposition Σ = V T ΛV , with Λ the diagonal matrix of

eigenvalues, all of which are nonnegative since Σ is PSD and V the matrix of eigenvectors. Set X = Λ1/2V .) Write

X = X1 X2 ,

so that XT . If is the projection operator onto Σij = Xj πX2 i the column-span of X2, then

(X2T −1 T I πX2 = X2 X2 ) X2 .

Multiplying on the left and right by and yields XT X1

)− T T (X2T T X1 X1 X1 X2 X2 X2 X1 −1 Σ11 Σ12Σ22 Σ21,

where the second line follows by grouping terms in the ﬁrst.

Lemma 2. Let x, y be random variables and let xG and yG be Gaussian random variables with the same covariance matrix. Then

Ex y − E[y|x] 2 ≤ ExG yG − E[yG|xG] 2 .

Proof. These are in fact the expected variances of the conditional variables:

E y − Ey|x 2 = ExEy|x y − Ey|x 2 = Ex Var[y|x].

Using the formula above for the Gaussian process MSE, we now need to show that

−1 Ex Var[y|x] ≤ Σyy −

By the law of total variance,

Var(y) = Varx(E[y|x]) + Ex Var(y|x).

So it sufﬁces to show that

We also considered setting each entry of s(x) to a random variable uniform on [0, 1]. This produces a random J - Varx(E[y|x]) −1 ≥ invariant function, ie, a distribution g(x) whose marginal g(x)J does not depend on xJ . Without loss of generality, we set Ex = Ey = 0. We compute the covariance of x with E[y|x]. We have

### 3.1. Uniform Pixel Selection

In Krull et. al., the authors propose masking procedures Cov(x, E[y|x]) = Ex [x · E [y|x]] that estimate a local distribution q(x) in the neighborhood of a pixel and then replace that pixel with a sample from = Ex [E [xy|x]] the distribution. Because the value at that pixel is used to = Ex [E [xy|x]] estimate the distribution, information about it leaks through = E[xy] and the resulting random functions are not genuinely J- = Cov(x, y) invariant. For example, they propose a method called Uniform Pixel The statement follows from an application of Lemma 1 to Selection (UPS) to train a neural net to predict xj from the covariance matrix of x and E[y|x]. UPSj(x), where UPSj is the random function replacing the jth entry of x with the value of at a pixel k chosen Proposition 1. Let x, y be random variables and let xG and uniformly at random from the r × r neighborhood centered yG be Gaussian random variables with the same covariance at j (Krull et al., 2018). fJ∗,G matrix. Let f∗ and be the corresponding optimal J- J invariant predictors. Then Write ιjk(x) is the vector x with the value xj replaced by xk . ∗,G 2 The function f ∗ minimizing the self-supervised loss E y − fJ (x) ≤ E y − fJ (x) . ∗

Ex f (UPSj(x))j − xj 2 Proof. We ﬁrst reduce the statement to unconstrained optimization, noting that satisﬁes ∗ Eyj |xJc . fJ (x)j =

f ∗(x)j = Ex[xj| UPSj(x)] The statement follows from Lemma 2 applied to yj, xJc . = Ex Ek[xj|ιjk(x)] E[xj |ιjk (x)] =2

## 3. Masking

r k = r21 E[xj|ιjj(x)] + r21 E[xj|ιjk(x)] In this section, we discuss approaches to modifying the input to a neural net or other function f to create a J -invariant k=j function. E[xj |x−j ] = 2 xj + 2 The basic idea is to choose some interpolation function s(x) r r k=j and then deﬁne g by 1∗ = 2 xj + 1 − 2 fJ (x)j , r r g(x)J := f (1J · s(x) + 1Jc · x)J , where f∗ E[xj|x−j] is the optimum of the self- (x)j = J where 1J is the indicator function of the set J. supervised loss among J -invariant functions. In Section 3 of the paper, on calibration, s is given by a This means that training using UPS masking can, given suflocal average, not containing the center. Explicitly, it is ﬁcient data and a sufﬁciently expressive network, produce a convolution with the kernel linear combination of the noisy input and the Noise2Self optimum. The smaller the region used for selecting the pixel,   the larger the contribution of the noise will be. In practice, 0 0.25 0  0.25 0 0.25  . however, a convolutional neural net may not be able to learn to recognize when it was handed an interesting pixel xj and 0 0.25 0

when it had been replaced (say by comparing the value at a If we ﬁx y, then xj and E[yj|x−j] are both independent espixel in UPSj(x) to each of its neighbors). timators of yj, so the above reasoning applies. Note that the loss itself is the variance of xj|x−j, whose two components One attractive feature of UPS is that it keeps the same perare the variance of xj|yj and the variance of yj|x−j. pixel data distribution as the input. If, for example, the input is binary, then local averaging and random uniform The optimal value of λ, then, is given by the variance of replacements will both be substantial deviations. This may the noise divided by the value of the self-supervised loss. regularize the behavior of the network, making it more For example the function f reduces the noise by a factor of sensible to pass in an entire copy of x to the trained network 10 (ie, the variance of yj|x−j is a tenth of the variance of later, rather than iteratively masking it. xj|yj), then λ∗ = 1/11 and the linear combination has a PSNR 0.43 higher than that of f alone. We suggest a simple modiﬁcation: exclude the value of xj when estimating the local distribution. For example, replace

## 4. Calibrating Traditional Denoising Methods

it with a random neighbor. The image denoising methods were all demonstrated on

### 3.2. Linear Combinations

the full camera image included in the scikit-image library for python (van der Walt et al., 2014). An inset from In this section we note that if f is J -invariant, then f (x)j that image was displayed in the ﬁgures. and xj give two uncorrelated estimators of yj for any coordinate j. Here we investigate the effect of taking a linear We also used the scikit-image implementations of the combination of them. median ﬁlter, wavelet denoiser, and NL-means. The noise standard deviation was 0.1 on a [0, 1] scale. Given two uncorrelated and unbiased estimators u and v of some quantity y, we may form a linear combination: In addition to the calibration plots for the median ﬁlter in the text, we show the same for the wavelet and NL-means denoisers in Supp. Figure 1. wλ = λu + (1 − λ)v.

## 5. Neural Net Examples

The variance of this estimator is

### 5.1. Datasets: Ha`nz`ı, CellNet, ImageNet

λ2U + (1 − λ)2V, Ha`nz`ı We constructed a dataset of 13029 Chinese characters (ha`nz`ı) rendered as white on black 64x64 images (image where U and V are the variances of u and v respectively. intensity within [0, 1]), and applied to each one substan- This expression is minimized at tial Gaussian (µ = 0, σ = 0.7) and Bernoulli (half pixels blacked out) noise. Each Chinese character appears 6 times in the whole dataset of 78174 images. We then split this λ = V /(U + V ). dataset in a training and test set (90% versus 10%). CellNet We constructed a dataset of 34630 image tiles The variance of the mixed estimator wλ is U V /(U + V ) = (128x128) obtained by random partitioning of a large col- . When the variance of is much lower than that V v 1+V /U lection of single channel ﬂuorescence microscopy images of u, we just get V out, but when they are the same the of cultured cells. These images were downloaded from the variance is exactly halved. Note that this is monotonic in V , Broad Bioimage Benchmark Collection (Ljosa et al., 2012). so if estimators v1, . . . , vn are being compared, their rank Before cropping, we ﬁrst gently denoise the images using will not change after the original signal is mixed in. In terms the non-local means algorithm. We do so in order to reof PSNR, the new value is move a very low and nearly imperceptible amount of noise already present in these images – indeed, the images have an excellent signal-to-noise ratio to start from. Next, we PSNR(wλ, y) = 10 ∗ log10 1 + V /U V use a rich noise model to simulate typical noise on sCMOS scientiﬁc cameras. This noise model consists of: (i) spatially variant gain noise per pixel, (ii) Poisson noise, (iii) Cauchy = PSNR(V ) + 10 ∗ log10(1 + V /U ) distributed additive noise. We choose parameters so as to V 1V2 ≈ PSNR(V ) + 10 obtain a very aggressive noise regime. −2 log10(e) U 2U ImageNet In order to generate a large collection of natural ≈ PSNR(V ) + 4.3 · V image tiles, we downloaded the ImageNet LSVRC 2013 Vali- U

Calibrating Wavelet Denoiser

Calibrating NL-means

**Figure 1. Calibrating a wavelet ﬁlter and Non-local means without ground truth. The optimal parameter for J -invariant (masked) versions**

can be read off (red arrows) from the self-supervised loss.

dation Set consisting of 20121 RGB images – typically photographs. From these images we generated 60000 cropped images of dimension 128x128 with each RGB value within [0, 255]. These images were mistreated by the strong combination of Poisson (λ = 30), Gaussian (σ = 80), and Bernoulli noise (p = 0.2). In the case of Bernoulli noise, each pixel channel (R, G, or B) has probability p of being dark or hot, i.e. set to the value 0 or 255.

masked classic

masked classic

Sigma threshold

masked classic

masked classic

Cut-off distance

### 5.2. Architecture

We use a UNet architecture modelled after (Ronneberger et al., 2015). The network has an hourglass shape with skip connections between layers of the same scale. Each convolutional block consists of two convolutional layers with 3x3 ﬁlters followed by an InstanceNorm. The number of channels is [32, 64, 128, 256]. Downsampling uses strided

convolutions and upsampling uses transposed convolutions. like Noise2Noise, Noise2Self, NL-means, and BM3D, will The network is implemented in PyTorch (Paszke et al., 2017) exhibit this shrinkage. To make up for this difference, we and the code is also included in the supplement. rescale the outputs of all methods to match the mean and variance of the ground truth.

### 5.3. Training

We compute the PSNR for fully reconstructed images on hold-out test sets which were not part of the training or We convert a neural net fθ into a random J -invariant funcvalidation procedure. tion:

## 6. Single-Cell Gene Expression

(1) 1J · fθ(1Jc · xJ + 1J · u) J ∈J The lossy capture and sequencing process producing singlecell gene expression can be expressed as a Poisson distriwhere u is a vector of random numbers distributed uniformly bution 1. A given cell has a density λ = (λ1, . . . , λm) over on [0, 1]. To speed up training, we only compute the coordigenes i ∈ {1, . . . m}, with i λi = 1. If we sample N nates for one J per pass, and that J is chosen randomly for molecules, we get a multinomial distribution which can be each batch with density 1/25. The loss is restricted to those approximated as xi ∼ Poisson(N λi). coordinates. While one would like to model molecular counts directly, We train with a batch size of 64 for Ha`nz`ı and CellNet and the large dynamic range of gene expression (about 5 orders a batch size of 32 for ImageNet. of magnitude) makes linear models difﬁcult to ﬁt directly. We train for 50 epochs for CellNet, 30 epochs for Ha`nz`ı Instead, one typically introduces a normalized variable z, and 1 epoch for ImageNet. for example

### 5.4. Inference

zi = ρ(N0 ∗ xi/N ), We considered two approaches for inference. In the ﬁrst, we consider a partition J containing 25 sets and apply where N = i xi is the total number of molecules in a Equation (1) to produce a genuinely J -invariant function. given cell, N0 is a normalizing constant, and ρ i√s some This requires |J | applications of the network. nonlinearity. Common values for ρ include x → x and x → log(1 + x). In the second, we just apply the trained network to the full noisy data. This will include the information from xj in Our analysis of the Paul et al. dataset (Paul et al., 2015) the prediction fθ(x)j. While the information in this pixel follows one from the tutorial for a diffusion-based denoiser was entirely redundant during training, some regularization called MAGIC, and we use the scprep package to perform induced by the convolutional structure of the net and the normalization (van Dijk et al., 2018). In the language above, training procedure may have caused it to learn a function N0 is the median of the total molecule count per cell and ρ which uses that information in a sensible way. Indeed, on is square root. our three datasets, the direct application was about 0.5 dB Because we work on the normalized variable z, the optimal better than the J -independent version. denoiser would predict

### 5.5. Evaluation

E[zi|λ] ≈ Exi∼Poisson Nλi √xi N0/N . We evaluated each reconstruction method using the Peak Signal-to-Noise Ratio (PSNR). For two images with range This function of λi is positive, monotonic and maps 0 to [0, 1], this is a log-transformation of the mean-squared error: 0, so it is directionally informative. Since expectations do not commute with nonlinear functions, inverting it would PSNR(x, y) = 10 ∗ log10(1/ x − y 2). not produce an unbiased estimate of λi. Nevertheless, it provides a quantitative estimate of gene expression which is Because of clipping, the noise on the image datasets is well-adapted to the large dynamic range. not conditionally mean-zero. (Any noise on a pixel with 1 While the polymerase chain reaction (PCR) used to amplify intensity 1, for example, must be negative.) This induces a the molecules for sequencing would introduce random multiplicabias: E[x|y] is shrunk slightly towards the mean intensity. tive distortions, many modern datasets introduce unique molecular For methods trained with clean targets, like Noise2Truth and indentiﬁers (UMIs), barcodes attached to each molecule before DnCNN, this effect doesn’t matter; the network can learn to ampliﬁcation which can be used to deduplicate reads from the produce the correct value. The outputs of the blind methods same original molecule.

## References

Krull, A., Buchholz, T.-O., and Jug, F. Noise2Void - learning denoising from single noisy images.

Ljosa, V., Sokolnicki, K. L., and Carpenter, A. E. Annotated high-throughput microscopy image sets for validation. Nature Methods, 9(7):637–637, July 2012.

Murphy, K. P. Machine Learning: a Probabilistic Perspective. Adaptive computation and machine learning series. MIT Press, Cambridge, MA, 2012. ISBN 978-0-262- 01802-9.

Paszke, A., Gross, S., Chintala, S., Chanan, G., Yang, E., DeVito, Z., Lin, Z., Desmaison, A., Antiga, L., and Lerer, A. Automatic differentiation in PyTorch. In NIPS-W, 2017.

Paul, F., Arkin, Y., Giladi, A., Jaitin, D., Kenigsberg, E., Keren-Shaul, H., Winter, D., Lara-Astiaso, D., Gury, M., Weiner, A., David, E., Cohen, N., Lauridsen, F., Haas, S., Schlitzer, A., Mildner, A., Ginhoux, F., Jung, S., Trumpp, A., Porse, B., Tanay, A., and Amit, I. Transcriptional heterogeneity and lineage commitment in myeloid progenitors. Cell, 163(7):1663–1677, December 2015.

Ronneberger, O., Fischer, P., and Brox, T. U-Net: Convolutional networks for biomedical image segmentation.

van der Walt, S., Schnberger, J. L., Nunez-Iglesias, J., Boulogne, F., Warner, J. D., Yager, N., Gouillart, E., Yu, T., and contributors, t. s.-i. scikit-image: image processing in Python. PeerJ, 2:e453, 2014.

van Dijk, D., Sharma, R., Nainys, J., Yim, K., Kathail, P., Carr, A. J., Burdziak, C., Moon, K. R., Chaffer, C. L., Pattabiraman, D., Bierie, B., Mazutis, L., Wolf, G., Krishnaswamy, S., and Peer, D. Recovering gene interactions from single-cell data using data diffusion. Cell, 174(3): 716–729.e27, July 2018.
