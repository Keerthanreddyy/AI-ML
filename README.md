# AI-ML
# Two Agents, One Language
### Emergent Communication and Compositionality in Neural Networks

Can two neural networks invent their own language simply by learning to communicate?

This project explores **emergent communication**: the process through which AI agents develop a communication protocol without being explicitly taught the meaning of individual symbols.

We build two neural agents that play a referential game. One agent observes an object and sends a message composed of discrete symbols. The other receives that message and must identify the correct object from a set of candidates. Through repeated interaction and feedback, the agents learn to communicate successfully.

Beyond achieving high accuracy, we investigate a deeper question: **Does the language they develop have compositional structure, or is it simply an arbitrary code?**

---

## Table of Contents

- [Project Overview](#project-overview)
- [How It Works](#how-it-works)
- [Research Questions](#research-questions)
- [Technical Approach](#technical-approach)
- [Evaluation Metrics](#evaluation-metrics)
- [Expected Results](#expected-results)
- [Technology Stack](#technology-stack)
- [Project Roadmap](#project-roadmap)
- [Team Contributions](#team-contributions)
- [References](#references)

---

## Project Overview

Natural languages allow us to express complex ideas using reusable components. For example, the words *red* and *circle* can be combined to describe a red circle, a red square, or a blue circle.

Can neural networks independently discover similar structure when they are rewarded only for successful communication?

To investigate this, we implement a sender–receiver system in which:

- The **Sender** observes a target object and generates a message using a learned vocabulary of discrete symbols.
- The **Receiver** observes the message and a set of candidate objects, then predicts which object the Sender intended.
- Both agents learn through a shared communication objective.

The environment will initially use synthetic objects described by attributes such as colour, shape, and size. This keeps the experimental setting controlled and allows us to measure the structure of the resulting communication protocol.

## How It Works

```text
             Target Object
                   |
                   v
          +------------------+
          |  Sender Network  |
          +------------------+
                   |
                   v
          Discrete Message
           [2, 0, 3, 1]
                   |
                   v
          +------------------+
          | Receiver Network |
          +------------------+
                   ^
                   |
          Candidate Objects
                   |
                   v
           Predicted Object
                   |
                   v
          Communication Loss
                   |
                   v
             Model Updates
```

### 1. Object Generation

Generate a collection of objects with different combinations of attributes.

Example:

| Object | Colour | Shape | Size |
|---|---|---|---|
| A | Red | Circle | Small |
| B | Blue | Square | Large |
| C | Red | Square | Large |
| D | Green | Circle | Large |

### 2. Sender Network

The Sender receives the target object's representation and produces a sequence of symbols from a finite vocabulary.

For example:

```text
Target: Red Circle
Message: [2, 0, 3]
```

The symbols have no predefined meanings. Their meaning, if any, emerges through training.

### 3. Receiver Network

The Receiver gets the message and a set of candidate objects. It predicts which candidate corresponds to the Sender's target.

The Sender and Receiver are trained to improve communication accuracy.

### 4. Learning to Communicate

We investigate a differentiable approach using **Gumbel-Softmax**, which approximates categorical sampling and enables gradient-based optimisation through discrete message choices.

We may also implement a reinforcement-learning baseline, such as REINFORCE, to compare learning speed and communication performance.

### 5. Language Analysis

After training, we analyse the learned messages to determine whether the agents have developed systematic relationships between object attributes and message structure.

For example, if a symbol consistently appears in messages describing red objects across different shapes and sizes, it may encode reusable information about colour.

This is a hypothesis to test, not an outcome we assume in advance.

---

## Research Questions

Our experiments focus on the following questions:

1. **Emergent communication:** Can two agents learn a reliable communication protocol without predefined symbol meanings?
2. **Compositionality:** Do individual message components correspond to reusable object attributes?
3. **Generalisation:** Can the Receiver identify objects with attribute combinations that were excluded from training?
4. **Learning efficiency:** How does Gumbel-Softmax compare with a reinforcement-learning approach?
5. **Training stability:** How sensitive are the learned protocols to random seeds, vocabulary size, message length, and optimisation settings?

---

## Technical Approach

### Model Architecture

The initial system will contain two neural networks.

**Sender**
- Input: target object attributes or feature representation.
- Encoder: multilayer perceptron (MLP).
- Message generator: categorical symbol selection over a fixed vocabulary.
- Output: a sequence of discrete communication symbols.

**Receiver**
- Input: received message and candidate object representations.
- Encoder: neural representation of the message and candidates.
- Scoring mechanism: computes a compatibility score for each candidate.
- Output: probability distribution over candidate objects.

### Training Objective

For a candidate set containing \(N\) objects, let \(y\) denote the index of the intended target and \(p_\theta(y \mid m, C)\) denote the Receiver's predicted probability for that target, given message \(m\) and candidate set \(C\).

The Receiver's cross-entropy loss is:

\[
\mathcal{L}_{\text{task}}
= -\log p_\theta(y \mid m,C)
\]

The training objective is to minimise this loss while accounting for the discrete message-generation mechanism.

We will begin with a task-only objective and investigate message-length penalties or other regularisation methods if needed.

### Gumbel-Softmax

Sampling a discrete symbol is not directly differentiable. Gumbel-Softmax provides a continuous relaxation of categorical sampling, allowing gradients to propagate through the message generator during training.

We will experiment with the temperature parameter, which controls how closely the relaxed distribution approximates discrete categorical choices.

### Experimental Baselines

Where feasible, we will compare:

- Gumbel-Softmax-based gradient training.
- REINFORCE-based policy-gradient training.
- Different vocabulary sizes and message lengths.
- Different random seeds and training configurations.

The comparison will consider convergence speed, communication accuracy, stability, and compositionality.

---

## Mathematical Foundations

Our project models communication between two neural agents as a differentiable learning problem. The Sender converts an object into a discrete message, and the Receiver uses that message to identify the intended object.

### 1. Referential Game

Let an object be represented by a feature vector:

\[
x = [x_{\text{colour}}, x_{\text{shape}}, x_{\text{size}}]
\]

The Sender observes a target object \(x\), while the Receiver receives a candidate set:

\[
C = \{x_1, x_2, \ldots, x_N\}
\]

The goal is to maximise the probability that the Receiver selects the intended target.

### 2. Sender Network

The Sender is a neural network parameterised by \(\theta_S\). It encodes the object and generates a probability distribution over a vocabulary of \(K\) symbols.

\[
h_S = f_{\theta_S}(x)
\]

\[
p_S(m_t \mid x)
= \operatorname{softmax}(W_Sh_S+b_S)
\]

The Sender produces a message of length \(L\):

\[
m = [m_1,m_2,\ldots,m_L]
\]

For example:

```text
Target object: Red Circle
Generated message: [2, 0, 3]
```

The symbols have no predefined meanings. Their usage emerges through training.

### 3. Gumbel-Softmax

Discrete sampling is not directly differentiable, making standard backpropagation through sampled messages difficult. We use Gumbel-Softmax as a differentiable relaxation of categorical sampling.

First, sample Gumbel noise:

\[
g_i=-\log(-\log u_i),
\qquad u_i\sim U(0,1)
\]

Then compute:

\[
y_i=
\frac{\exp((a_i+g_i)/\tau)}
{\sum_{j=1}^{K}\exp((a_j+g_j)/\tau)}
\]

where:

- \(a_i\) is the logit for symbol \(i\).
- \(g_i\) is the sampled Gumbel noise.
- \(\tau\) is the temperature parameter.
- \(y_i\) is the relaxed probability assigned to symbol \(i\).

A lower temperature produces more concentrated distributions. A straight-through estimator can use hard one-hot symbols in the forward pass while using relaxed gradients in the backward pass.

### 4. Receiver Network

The Receiver encodes the message and candidate objects, then computes a compatibility score for each candidate.

\[
h_m=f_{\theta_R}^{\text{msg}}(m)
\]

\[
h_i=f_{\theta_R}^{\text{obj}}(x_i)
\]

One possible scoring function is:

\[
s_i=h_m^\top h_i
\]

The Receiver converts the scores into a probability distribution:

\[
P(i\mid m,C)=
\frac{\exp(s_i)}
{\sum_{j=1}^{N}\exp(s_j)}
\]

The candidate with the highest probability is selected as the predicted target.

### 5. Training Objective

Let \(y\) denote the index of the correct candidate. We train the agents using cross-entropy loss:

\[
\boxed{
\mathcal L_{\text{task}}
=-\log P(y\mid m,C)
}
\]

A correct candidate assigned a high probability receives a lower loss. An incorrect or low-confidence prediction receives a higher loss.

The model parameters are updated using gradient-based optimisation:

\[
\theta_S \leftarrow
\theta_S-\eta\nabla_{\theta_S}\mathcal L_{\text{task}}
\]

\[
\theta_R \leftarrow
\theta_R-\eta\nabla_{\theta_R}\mathcal L_{\text{task}}
\]

where \(\eta\) is the learning rate.

Gumbel-Softmax allows an approximate gradient to propagate through the discrete communication channel, helping the Sender learn messages that improve the Receiver's predictions.

### 6. Measuring Compositionality

A language is compositional when reusable components of a message systematically correspond to reusable attributes or concepts.

Consider this hypothetical protocol:

| Object | Message |
|---|---|
| Red circle | `A X` |
| Red square | `A Y` |
| Blue circle | `B X` |
| Blue square | `B Y` |

Here, the first symbol represents colour and the second represents shape. The protocol reuses components across different objects.

However, communication accuracy alone cannot establish compositionality. An arbitrary code can also achieve perfect accuracy on familiar objects.

### 7. Topographic Similarity

Topographic similarity measures the relationship between distances in object space and distances in message space.

For categorical object attributes, Hamming distance can be defined as:

\[
d_X(x_i,x_j)=
\sum_{k=1}^{F}
\mathbf{1}[x_{ik}\ne x_{jk}]
\]

For fixed-length messages:

\[
d_M(m_i,m_j)=
\sum_{t=1}^{L}
\mathbf{1}[m_{it}\ne m_{jt}]
\]

We calculate pairwise distances and correlate the two sets using Spearman's rank correlation:

\[
\boxed{
\rho_{\text{topo}}=
\operatorname{Spearman}
\left(
\{d_X(x_i,x_j)\}_{i<j},
\{d_M(m_i,m_j)\}_{i<j}
\right)
}
\]

A higher positive correlation indicates that more dissimilar objects tend to have more dissimilar messages. This is evidence of systematic structure, but not definitive proof of compositionality.

### 8. Generalisation to Unseen Combinations

We will hold out selected combinations of object attributes during training and evaluate the agents on those combinations after training.

\[
\text{Generalisation Accuracy}
=
\frac{\text{Correct held-out predictions}}
{\text{Total held-out games}}
\]

This tests whether the agents can communicate about novel combinations of familiar attributes rather than merely memorising individual objects.

### 9. Information-Theoretic Perspective

Let \(X\) denote the target object and \(M\) the generated message. Mutual information measures how much observing the message reduces uncertainty about the target:

\[
\boxed{
I(X;M)=H(X)-H(X\mid M)
}
\]

Here, \(H(X)\) is the entropy of the target distribution and \(H(X\mid M)\) is the remaining uncertainty after observing the message.

High mutual information indicates that messages carry information about objects, but it does not necessarily imply compositionality. An arbitrary code can also encode substantial information.

### Summary

Our mathematical framework combines:

- **Neural networks** to generate and interpret messages.
- **Gumbel-Softmax** to enable gradient-based learning through discrete symbols.
- **Cross-entropy loss** to optimise communication accuracy.
- **Topographic similarity** to analyse the structure of the learned protocol.
- **Held-out evaluation** to test generalisation to unseen attribute combinations.
- **Information theory** to quantify the information messages carry about intended objects.

The central research question is whether successful communication leads to a structured, compositional language or merely an arbitrary code.


## Evaluation Metrics

High communication accuracy alone does not establish that a language is compositional. We will use multiple complementary metrics.

### 1. Communication Accuracy

The percentage of games in which the Receiver correctly identifies the target object.

\[
\text{Accuracy}
=\frac{\text{Correct predictions}}{\text{Total games}}
\]

### 2. Topographic Similarity

Topographic similarity measures the correlation between distances in the object space and distances in the message space.

For example, objects that differ only in colour might be expected to have more similar messages than objects that differ in every attribute.

We will compare pairwise object distances with pairwise message distances using an appropriate correlation measure, such as Spearman's rank correlation.

A higher score indicates a stronger relationship between object similarity and message similarity, but does not by itself prove compositionality.

### 3. Held-Out Combination Accuracy

We will exclude selected combinations of object attributes during training and evaluate the agents on these unseen combinations.

This tests whether the communication protocol generalises beyond memorised training examples.

The experiment must be designed carefully so that the intended generalisation task remains solvable from the information available to the agents.

### 4. Symbol and Position Analysis

We will inspect message sequences to investigate whether particular symbols or positions are associated with attributes such as colour, shape, or size.

This analysis can reveal reusable structure, but symbol meanings should be inferred from systematic evidence rather than a few examples.

### 5. Learning Curves and Stability

We will track:

- Training and evaluation accuracy.
- Loss over training steps.
- Convergence speed.
- Performance across independent random seeds.
- Variation in topographic similarity and held-out accuracy.

---

## Expected Results

We aim to produce:

- A functioning sender–receiver communication system.
- A trained emergent communication protocol.
- An interactive visualisation of messages and predictions.
- Quantitative measurements of communication accuracy and compositionality.
- An experimental comparison of selected learning methods.
- An analysis of generalisation, training stability, and limitations.

We do not assume that compositionality will emerge automatically. Discovering when it appears, when it fails, and how training choices affect it is a central part of the project.

---

## Technology Stack

The initial implementation is planned around:

| Component | Technology |
|---|---|
| Programming language | Python |
| Deep learning | PyTorch |
| Numerical computation | NumPy |
| Data visualisation | Matplotlib |
| Experimental analysis | pandas, SciPy |
| Interactive demo | Streamlit or a lightweight web interface |
| Version control | Git and GitHub |

These are proposed tools; the final stack may change as implementation progresses.

---

## Project Roadmap

- [ ] Review literature on emergent communication and compositionality.
- [ ] Implement the synthetic object-generation environment.
- [ ] Build the Sender and Receiver networks.
- [ ] Implement the referential game and training loop.
- [ ] Integrate Gumbel-Softmax message generation.
- [ ] Establish a reliable baseline and track communication accuracy.
- [ ] Implement topographic similarity analysis.
- [ ] Evaluate generalisation to held-out attribute combinations.
- [ ] Compare training configurations and random seeds.
- [ ] Build an interactive visual demonstration.
- [ ] Document experimental findings and limitations.

---

## Team Contributions

The project can be organised into parallel workstreams:

| Workstream | Responsibilities |
|---|---|
| Research and theory | Literature review, hypotheses, experimental design |
| Environment and data | Object generation, attribute combinations, game setup |
| Model development | Sender and Receiver architectures |
| Training and optimisation | Gumbel-Softmax, loss functions, training stability |
| Evaluation and analysis | Accuracy, topographic similarity, generalisation |
| Visualisation and integration | Interactive demo, results dashboard, final integration |

Responsibilities can be shared and adjusted according to the team's size and experience.

---

## References

The project builds on established research into emergent communication, multi-agent learning, and differentiable discrete sampling.

- Jang, E., Gu, S., & Poole, B. (2017). [Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144).
- Foerster, J. et al. (2016). [Learning to Communicate with Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1605.06676).
- Lazaridou, A., Peysakhovich, A., & Baroni, M. (2017). [Multi-Agent Cooperation and the Emergence of (Natural) Language](https://arxiv.org/abs/1612.07182).
- Brighton, H. (2002). [Compositional Syntax from Cultural Transmission](https://doi.org/10.1162/106454602753694765).

These references provide starting points for the literature review; the precise experimental design will be finalised after reviewing the relevant papers.

---

## Project Status

**Status:** Proposed / Initial implementation pending.

This is an experimental machine-learning project. Results, performance claims, and conclusions about compositionality will be reported after the experiments have been conducted.

