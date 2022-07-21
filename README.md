
# Using Bayesian methods to find change point in sequential data

This repository is based on a project in a production plant to apply Bayesian methods to find change points in a sequence of production machine data, in order to alert engineers for maintenance. 

The production line uses mounter machines to place components onto a PCB. The mounter is adjusted according to a planned schedule. However, sometimes the mounter's mounting accuracy would drop below a threshold, causing defective products, recalls, and unplanned stoppage for emergency maintenance. 

Experienced engineers know that these "sudden" drops in accuracy are not that sudden after all. There are usually signs before performance crosses the final threshold, in the form of a small, step-wise or continuous deviation to one side of the designated position. These deviations are so small and irregular, that they do not trigger automatic optical inspection (AOI) alerts. But these changes accumulate, until they finally cause trouble.

At the time of the project, engineers rely on frequent manual inspection of the performance data, as well as personal experience to spot these signs, and plan preventive maintenance. They would like to have an algorithm that automatically warns them where there are change points detected in the performance data over time. The challenge is how to detect a pattern change when the base pattern is a random distribution with noise. 

The problem can be formulated into a timeseries change point detection. Bayesian methods are used for its ability to detect change in a statistical distribution. The project is largely an implementation of the original paper from [Schuetz and Holschneider, University of Potsdam](https://arxiv.org/pdf/1104.3448.pdf). Kudos to the authors.

### Bayesian methods in a few words

We demonstrate Bayesian method with a simple case: a school as 60% boys and 40% girls. Boys always wear pants, girls wear pants 50% of the time and dresses 50% of the time. If you see a pupil wearing a pair of pants, what is the probability that it is a boy / girl?

Assuming the school as U pupils in total. There are U \* P(Boy) \* P(Pants|Boy) boys who wear pants, and U \* P(Girl) \* P(Pants|Girl) girls who wear pants. The latter divided by the total is the probability that the person in pants you see is a girl:

P(Girl|Pants) = P(Girl) \* P(Pants|Girl) / [P(Boy) \* P(Pants|Boy) + P(Girl) \* P(Pants|Girl)]

To generalize the case, we have the Bayes' theorem:

P(B|A) = P(A|B) \* P(B) / [P(A|B) \* P(B) + P(A|~B) \* P(~B) ]

or P(B|A) = P(AB) / P(A), or P(B|A) \* P(A) = P(AB)


## Our change point detection case

### Problem formulation

Based on the engineers' requests, three types of changes are identified that need to be detected:

- A step-wise change in the average placement position of the components
- A continuous change in the average
- A step-wise change in the standard deviation

Mounter placment position is average position + a random dispersion from the average: $y(t) = \beta_0  \phi_-^\theta + \beta_1 \zeta_-^\theta + \beta_2 \zeta_+^\theta + \beta_3 \phi_+^\theta + \xi(t)$
  
where $\phi$ and $\zeta$ are step functions, $\phi$ is a constant step in both the positive and negative directions, $\zeta$ a linear step function:
  
  $$\phi_-^\theta = 
  \begin{cases}
    1 &\quad \text{if } t \text{ <= } \theta \\ 
    0 &\quad \text{else}
  \end{cases}$$
  
  $$\phi_+^\theta = 
  \begin{cases}
    1 &\quad \text{if } t \text{ >= } \theta \\ 
    0 &\quad \text{else}
  \end{cases}$$
  
  $$\zeta_-^\theta = 
  \begin{cases} 
    \theta - t  &\quad \text{if } t \text{ <= } \theta \\ 
    0 &\quad \text{else}
  \end{cases}$$
  
  $$\zeta_+^\theta = 
  \begin{cases}
    \theta -t &\quad \text{if } t \text{ >= } \theta \\
    0 &\quad \text{else}
  \end{cases}$$

$\xi(t)$ is a normally distributed random variable around the average. The amplitude of $\xi(t)$ can also change continuously, therefore we define the standard deviation $STD(\xi(t)) = \sigma(1 + s_1 \zeta_-^\theta + s_2 \zeta_+^\theta)$.

The objective is to calculate the probability of a change in any of the parameters $\beta$, $\theta$ and $s$. High probability of change in 1). $\beta$ would mean high probability of step change in the average placement position; in 2). $\theta$ would mean a continuous change in the position; in 3). $s$ would mean a continuous change in the amplitude of the dispersion. 
  
### System modeling

$y = F \beta + \xi$, where
  
  $$F_\theta = 
 \begin{pmatrix}
 (\phi_-^\theta)_1 & (\zeta_-^\theta)_1 & (\zeta_+^\theta)_1 & (\phi_+^\theta)_1 \\
 \vdots  & \vdots  & \ddots & \vdots  \\
 (\phi_-^\theta)_n & (\zeta_-^\theta)_n & (\zeta_+^\theta)_n & (\phi_+^\theta)_n \\
 \end{pmatrix}$$
 
 $t \in [1, n]$, t is time, n is horizon
 
Noise is assumed to be normally distributed and described as:
 
$\xi \sim \mathcal{N} (0, \sigma^2 \Omega)$, where covariance matrix $\Omega$ is
  
$$(\Omega_{\theta, s_1, s_2})_{ij} = \big( \big[ 1 + s_1 (\zeta_-^\theta)_j + s_2 (\zeta_+^\theta)_j \big]^2 \big) \cdot \delta_{ij}$$
  
$\delta_{ij}$ is the dirac delta.

### Likelihood function

Based on the above equations, we get the probability density function $y \sim \mathcal{N} (F \hat\beta, \sigma^2 \Omega)$ and likelihood function

$$\mathcal{L}(\beta, \sigma, s, \theta|y) = \frac{1}{(2 \pi \sigma^2)^\frac{n}{2} \sqrt{|\Omega|}} e^{-\frac{1}{2 \sigma^2}(y - F \beta)^T \Omega^{-1} (y - F \beta)}$$
  
### Maximum likelihood

There exists $\beta^{\*} = arg\min_{\beta \in \mathbb{R}^4} (y - F \beta )^T \Omega{-1} (y - F \beta)$ such that the likelihood function is maximized:

$$\mathcal{L}(\beta, \sigma, s, \theta|y) =\frac{1}{(2 \pi \sigma^2)^\frac{n}{2} \sqrt{|\Omega|}} \exp(-\frac{\mathcal{R}^2}{2 \sigma^2}) \exp(\frac{1}{2 \sigma^2}(\beta - \beta^{\*})^T \Xi (\beta - \beta^{\*}))$$
 
where $\Xi = F^T \Omega^{-1} F$, $\mathcal{R}^2$ is the residual, and 
 
$$\mathcal{R}^2 = \min_{\beta \in \mathbb{R}^3} (y - F \beta)^T \Omega^{-1} (y - F \beta)  = (y - F \beta^{\*})^T \Omega^{-1} (y - F \beta^{\*})$$

We can also estimate the system's error through residual: $\hat \sigma^2 = \frac{\mathcal{R}^2}{n + 1}$.
 
### The prior

Assuming all parameters are independent. Their joint prior is
  
$$p(\beta, \sigma, \theta, s) = p(\beta) \cdot p(\sigma) \cdot p(\theta) \cdot p(s)$$
  
Since the prior is unknown, I use noninformative prior (jeffery's prior), $p(\vec{\theta}) \sim \sqrt{\det F(\vec{\theta})}$, where $F(\vec{\theta})$ is the Fisher information matrix: 

$$F(\theta) = \int (\frac{\partial}{\partial \theta} log (f(x; \theta)) )^2 f(x; \theta) dx$$ 
  
When x is normally distributed, or $f(x|\mu) = \frac{e^{-(x - \mu)^2 / 2\sigma^2}}{\sqrt{2 \pi \sigma^2}}$, $\theta$ is the average $\mu$, Jeffery's prior can be written as:
  
$$p(\vec{\mu}) \sim \sqrt{\det F(\vec{\mu})} = \sqrt{\int\limits_{-\infty}^{+\infty} f(x|\mu)\Big(\frac{x - \mu}{\sigma^2} \Big)^2 dx} = \sqrt{E\Big[\big(\frac{x - \mu}{\sigma^2} \big)^2 \Big]} = \sqrt{\frac{\sigma^2}{\sigma^4}} \sim 1$$  
  
Or the normal distributions Jeffery's prior is not dependent on the average. Based on which, we define the prior of location parameters $\beta, s, \theta$ as constants:
  
$$p(\theta) \sim 1, p(s) \sim 1, p(\beta) \sim 1$$
  
The multiplier $\sigma$ is a scale parameter, its Jeffery's prior is
 
$$p(\sigma) \sim \sqrt{\int\limits_{-infty}^{+infty} f(x|\sigma) \Big( \frac{(x - \mu)^2 - \sigma^2}{\sigma^3} \Big)^2 dx } = \sqrt{E\Big[ \big( \frac{(x - \mu)^2 - \sigma^2}{\sigma^3} \big)^2 \Big]} = \sqrt{\frac{2}{\sigma^2}} \sim \frac{1}{\sigma}$$
 
or simply the reciprocal: $p(\sigma) \sim \frac{1}{\sigma}$.
  
### Calculating the posterior

Now the problem is simplified into: given the likelihood function and the prior, calculate the posterior (Bayesian inference): $p(\theta|x) = \frac{p(x|\theta) p(\theta)}{p(x)}$.

Or simply $p(\theta|x) \sim p(x|\theta) p(\theta)$, $p(x)$ is called evidence, $p(x|\theta) = \mathcal{L}(\theta|x)$ is the likelihood function, $p(\theta)$ the prior.

Taking the priors and $\mathcal{L}$, we get: 

$$p(\beta, \sigma, \theta, s|y) \sim \mathcal{L}(\beta, \sigma, \theta, s|y) \cdot \frac{1}{\sigma}$$
  
Partially integrating all parameters will give us the posterior. 

Partial integration of $\beta$: $p(\sigma, \theta, s|y) \sim \frac{\sigma^{1 - n}}{\sqrt{|\Omega| |F^T \Omega^{-1} F|}} e^{ - \frac{1}{2 \sigma^2} \mathcal{R}^2 }$
  
of $\sigma$: $p(\theta, s|y) \sim \frac{\mathcal{R}^{-(n-2)} }{\sqrt{|\Omega| |F^T \Omega^{-1} F|} }$

Then I used a numeric method to integrate $s$. 

Final $\theta$ posterior: $p(\theta|y) = \int ds\cdot p(\theta, s|y)$

In the same way we can get the posteriors of all the parameters. A high probability in any of the posteriors indicate a change point of the corresponding type. 

### Implementation and sample code:

Calculating $\phi_-^\theta$, $\phi_+^\theta$, $\zeta_-^\theta$, $\zeta_+^\theta$:

```python
import numpy as np
from matplotlib import pyplot as plt
def xiMinus(theta, t, mode = 'constant'):
    scale = 1
    if mode == 'constant':
        if t <= theta:
            return -1.0 / scale
        else:
            return 0.0
    if mode == 'linear':
        if t <= theta:
            return 1.0 / scale * (theta - t)
        else:
            return 0.0

def xiPlus(theta, t, mode = 'constant'):
    scale = 1
    if mode == 'constant':
        if t <= theta:
            return 0.0
        else:
            return 1.0 / scale
    if mode == 'linear':
        if t <= theta:
            return 0.0
        else:
            return 1.0 / scale * (t - theta)

theta = 5
n = np.arange(11)

result = np.zeros(len(n))
for t in n:
    result[t] = xiMinus(theta, t, 'linear') + xiPlus(theta, t, 'linear')

fig, ax = plt.subplots()
plt.plot(result)
```

Calculating $F_\theta$:

```python
def fTheta(theta, n):
    f = np.zeros([n,4])
    for j in np.arange(0, n):
        f[j][0] = xiMinus(theta, j, 'constant')
        f[j][1] = xiMinus(theta, j, 'linear')
        f[j][2] = xiPlus(theta, j, 'linear')
        f[j][3] = xiPlus(theta, j, 'constant')
    return f

f = fTheta(2, 5)
f
```

    array([[-1.,  2.,  0.,  0.],
           [-1.,  1.,  0.,  0.],
           [-1.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  1.],
           [ 0.,  0.,  2.,  1.]])


Calculating covariance matrix $\Omega$:

```python
def omegaTheta(theta, s, n):
    om = np.zeros([n, n])
    for j in np.arange(0, n):
        om[j][j] = int(np.power (1 + s[1] * xiMinus(theta, j, 'linear') + s[2] * xiPlus(theta, j, 'linear'), 2))      
    return om
om = omegaTheta(2, [0, -1, 1, 0], 5)
om
```

    array([[ 1.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  0.,  0.,  4.,  0.],
           [ 0.,  0.,  0.,  0.,  9.]])



Maximum likelihood: 

Calculating $\beta^{\*}$, $\mathcal{R}^2$ and $\hat \sigma^2$:


```python
def betaStar(y, fTomInvf, fTomInv):
    tmp = fTomInvf
    target = np.matmul(fTomInv, y)
    return np.linalg.solve(tmp, target)

def residuumSq(y, betastar, omInv):      
    fBetastar = np.matmul(f, betastar)
    ytoFBetastar = np.subtract(y, fBetastar)
    ytoFBetastarT = np.transpose(ytoFBetastar)
    tmp = np.matmul(ytoFBetastarT, omInv)
    result = np.matmul(tmp, ytoFBetastar)
    rsq = result.tolist()[0][0]
    return rsq

def sigmaHat(rsq, n):
    sigmaHat = np.sqrt(rsq / (n + 1))
    return sigmaHat

def inverseFunc(mat):
    if np.linalg.cond(mat) < 1 / np.finfo(mat.dtype).eps:
        return np.linalg.solve(mat, np.eye(mat.shape[1], dtype = float))
    else:
        U, S, V = scipy.linalg.svd(mat)
        D = np.diag(S)
        tmp = np.matmul(V, np.linalg.inv(D))
        return np.matmul(tmp, np.transpose(U))

y = np.array([1,2,3,4,5])
y = y.reshape(-1,1)
om = np.where(om == 0, 1e-7, om)    

_, omLogdet = np.linalg.slogdet(om)
omDet = np.exp(omLogdet)
omInv = inverseFunc(om)
fTomInv = np.matmul(np.transpose(f), omInv)
fTomInvf = np.matmul(fTomInv, f)
_, logdet = np.linalg.slogdet(fTomInvf)
fTomInvfDet = np.exp(logdet)            
omDetfTomInvfDet = omDet * fTomInvfDet

betastar = betaStar(y, fTomInvf, fTomInv)
rsq = residuumSq(y, betastar, omInv)
sigmaHat = sigmaHat(rsq, len(y))

print(betastar)
print(rsq)
print(sigmaHat)
```

    [[-3.]
     [-1.]
     [ 1.]
     [ 3.]]
    8.486838935292248e-18
    1.18931625562e-09
    

### Results:

Test data from Nile river water level. Blue line is original data, yellow line is the posterior of the changing point $\theta$, green is $\beta$. The peak of yellow represents the point where probability of a change in pattern is most likely. 

  ![alt_text](doc/nileChangePointResult.png)

## Practicality concerns:

The most computational intensive part of the algorithm is the partial integration. The original version took 20 minutes to get the posterior of $\theta$ (by the time it's finished, the production line is already stopped...). Various performance improvement efforts in both the algorithm and the implementation reduced the time to 1 second in the final production version. 

### Sampling method

If we know approximately how the posterior looks like, and it doesn't change drastically, we can use sampling methods to reduce computation requirement. Fortunately this befits our requirement on the production line. 

I used Markov Chain Monte Carlo (MCMC) + Gibbs sampling to sample parameters and generate posterior. Pseudo code for integration over $s$ and $\theta$:

>initialize s1 and s2 to 0
>
>FOR i < iteration DO:
>
> - use s1 and s2 to get $p(\theta, s|y)$
> - take $p(\theta, s|y)$ as the estimated posterior and sample $\theta$ from it
> - use the sampled $\theta$ to calculate $p(s1|\theta, s2, y)$
> - take $p(s1|\theta, s2, y)$ as the estimated posterior and sample s1
> - use the sampled s1 to calculate $p(s2|theta, s1, y)$
> - take $p(s2|\theta, s1, y)$ as the estimated posterior and sample s2

Fortunately, $\sigma$ is steady regardless of the existance of a change point, and s1 and s2 are steady around average of 0. Therefore the convergence of our sampling method is fast. We also take 10 iterations, ignoring the burn-in period. Moreover, we fix the data batch size to 100 (each time take 100 historical data points to detect change point from), thus we only need to calculate the determinant once and save it for future use.

### multiprocessing

Using Python's multiprocessing. I used a Bayes class to pass global variables between processes. The tested result on 

- 2.7GHz/16GB dual core
- data batch size of 100
- MCMC algorithm with 10 iterations

The total runtime is lower than < 2.5 seconds. In reality, an Intel Xeon E7-4820 with 8 cores is used, and the runtime is less than 0.5 seconds.
