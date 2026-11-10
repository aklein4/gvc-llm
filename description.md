# Token-level Generative Value Classification

Based on: https://alexanderswerdlow.github.io/gvc/

Our GVC adaptation learns a binary return-conditioned language model and uses
it to guide a frozen base policy. Each generated token is an action; the
verified outcome of the completed response supplies its return label
$r \in \{0,1\}$.

The architecture contains a frozen policy and a separate
critic initialized from the same pretrained model. The critic trains a shared
LoRA adapter and a linear binary return-prior head; its pretrained weights,
including token embeddings, remain frozen.

The positive branch uses the original Abel prompt. The negative branch prepends:

> Deliberately solve the following problem incorrectly and give an incorrect final answer.

Both branches then score the same response tokens. This instruction is used in
training and generation, with no learned reward embeddings. Both branches share
all critic parameters. The prior head reads the **positive-branch** hidden state
to predict $p(r\mid s)$, where $s$ is the prefix preceding the next token.
Positive features are used for both prior target labels.

The prior head's weights and biases start at zero, yielding a 50/50 return prior.
The return-prior head uses a different learning rate than LoRA, typically `3e-4`.

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
