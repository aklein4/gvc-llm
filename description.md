# Token-level Generative Value Classification

Based on: https://alexanderswerdlow.github.io/gvc/

Our GVC adaptation learns a binary return-conditioned language model and uses
it to guide a frozen base policy. Each generated token is an action; the
verified outcome of the completed response supplies its return label
$r \in \{0,1\}$.

The architecture contains a frozen policy and a separate
critic initialized from the same pretrained model. The critic trains a shared LoRA adapter, two reward vectors $e_0,e_1 \in \mathbb{R}^{D}$, and a
linear binary return-prior head; its pretrained weights remain frozen.

Conditioning adds the selected reward vector to **every token's input hidden
state**, before the first Transformer layer:

$$
h_t^{(0)} = E[x_t] + e_r.
$$

This applies to prompt and response tokens.
Both reward branches share all critic parameters except their reward vectors.
The prior head reads the **positive-conditioned** hidden state to predict
$p(r\mid s)$, where $s$ is the prefix preceding the next token. Positive
conditioning is a fixed feature choice, not the prior's target label.

The negative reward vector starts at
$e_0=0.1\,\operatorname{std}_{\mathrm{vocab}}(E)$, using the per-dimension
population standard deviation, shape $[D]$. The positive reward vector starts
at $e_1=0$, with shape $[D]$. The two vectors are independent trainable
parameters. The prior head's weights and biases start at zero, yielding a
50/50 return prior.

Reward vectors and return-prior head use a different lr than the lora parameters, typically `3e-4`.

For each response token $a$, the objective is

$$
\mathcal L = -\log p_\theta(a\mid s,r)-\log p_\psi(r\mid s).
$$

Token likelihood trains only the observed-outcome branch. The prior trains on
both outcomes using positive-branch features. Returns are Monte Carlo outcomes;
there is no policy-gradient loss, TD bootstrap, or value-anchor loss.

At generation time, both branches score every candidate next token. Their
vocabulary-normalized likelihoods and the return prior define

$$
Q(s,a)=\frac{p_\theta(a\mid s,1)p_\psi(1\mid s)}
{\sum_{r\in\{0,1\}}p_\theta(a\mid s,r)p_\psi(r\mid s)}.
$$

The frozen policy is guided with logits
$(\ell_{\mathrm{base}}(a)+Q(s,a)/\beta)/T$, where $T$ is sampling
temperature. Larger positive $\beta$ weakens guidance; as
$\beta\to\infty$, sampling approaches the temperature-scaled base policy.
