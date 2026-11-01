# Confidence Score Formula Explained

This guide explains how SafeLens calculates the final **Confidence Score** from the AI's internal token log-probabilities.

Instead of asking an LLM to guess a number out loud (which causes text hallucination and extreme overconfidence), SafeLens derives confidence through a 3-stage mathematical progression:
1. **$z$ (Log-Odds Logit)**: The raw internal tug-of-war score.
2. **$P_{\text{raw}}$ (Raw Probability)**: Squashing $z$ into a standard 0% to 100% scale.
3. **$P_{\text{calibrated}}$ (Final Calibrated Confidence)**: Applying temperature scaling to soften overconfidence into statistically honest probabilities.

---

## Stage 1: The Log-Odds Logit ($z$)

### Layman's Explanation
Imagine a tug-of-war match inside the AI's brain between two teams: **Team "Harmful"** and **Team "Safe"**.

Instead of asking the AI to write down a percentage out loud, we listen directly to the AI's internal neurons at the exact moment it chooses between the decision words `"true"` (harmful) and `"false"` (safe).

- If the AI shouts `"true"` much louder than `"false"`, Team Harmful wins by a wide margin ($z$ is a large positive number like $+3.5$).
- If the AI shouts `"false"` much louder, Team Safe wins ($z$ is a large negative number like $-3.5$).
- If the AI is completely torn and shouts both with equal strength, the match is tied ($z = 0$).

$z$ is simply this **tug-of-war score** (the difference in volume between the two choices).

### The Formula
$$z = \ln p(\text{"true"}) - \ln p(\text{"false"})$$

### Components & What the Formula Does
- **$p(\text{"true"})$**: The AI's raw internal probability of choosing the token `"true"`.
- **$p(\text{"false"})$**: The AI's raw internal probability of choosing the token `"false"`.
- **$\ln$**: The natural logarithm (base $e$).
- **What it does**: In mathematics, subtracting two logarithms is identical to taking the log of their ratio:
  $$z = \ln \left( \frac{p(\text{"true"})}{p(\text{"false"})} \right)$$
  This ratio is called the **odds ratio**, and $z$ is the **log-odds**. It strips away all unrelated words in the dictionary and isolates purely how many times more likely the AI considers the content harmful compared to safe:
  - If $p(\text{"true"}) = 0.90$ and $p(\text{"false"}) = 0.10 \implies z = \ln(9) \approx +2.20$ (strongly harmful).
  - If $p(\text{"true"}) = 0.50$ and $p(\text{"false"}) = 0.50 \implies z = \ln(1) = 0.00$ (complete uncertainty / tie).
  - If $p(\text{"true"}) = 0.10$ and $p(\text{"false"}) = 0.90 \implies z = \ln(1/9) \approx -2.20$ (strongly safe).

---

## Stage 2: Converting $z$ into Raw Probability ($P_{\text{raw}}$)

### Layman's Explanation
The tug-of-war score $z$ can be any number on an infinite scale from negative infinity to positive infinity (like $-10$ or $+15$).

However, human moderators and software interfaces cannot work with an infinite scale. We need a standard probability dial between **0.0 and 1.0** (or **0% to 100%**).

$P_{\text{raw}}$ is the converter that squashes that unbounded tug-of-war score $z$ onto a clean 0% to 100% scale:
- A huge negative score turns into near 0%.
- A tie score of 0 turns into exactly 50%.
- A huge positive score turns into near 100%.

### The Formula
$$P_{\text{raw}} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

### Components & What the Formula Does
- **$z$**: The log-odds score calculated in Stage 1.
- **$e$**: Euler's mathematical constant ($\approx 2.71828$).
- **$e^{-z}$**: The exponential of the negative log-odds.
- **$\sigma(z)$**: The **Sigmoid function**, an S-shaped curve that smoothly compresses any real number into the open interval $(0, 1)$.
- **What it does**:
  - If $z = 0$: $e^0 = 1 \implies \frac{1}{1 + 1} = 0.50$ (exactly **50%** probability).
  - If $z = +2.20$: $e^{-2.20} \approx 0.111 \implies \frac{1}{1 + 0.111} \approx 0.90$ (**90%** probability).
  - If $z = -2.20$: $e^{2.20} \approx 9.0 \implies \frac{1}{1 + 9.0} \approx 0.10$ (**10%** probability).

---

## Stage 3: The Overconfidence Problem (Why $P_{\text{raw}}$ Cannot Be the Final Confidence)

### The Problem in Plain English
If $P_{\text{raw}}$ gives us a percentage from 0% to 100%, why can't we stop here?

Modern deep neural networks (especially LLMs trained with RLHF) suffer from severe **overconfidence**. When making classifications, their internal logits $z$ blow up to very large values (e.g. $z = +4.0$ or $+5.0$), which makes $P_{\text{raw}}$ jump straight to **$98\%$ or $99\%$**—even on ambiguous edge cases like historical documentaries or news footage.

If an AI says *"I'm 99% confident"* on 100 videos, but in reality 20 of them are false alarms, the AI is **uncalibrated**.

**A true confidence score must mean what it says**:
> If a calibrated system outputs **85% confidence**, exactly **85 out of 100** of those videos must actually be harmful in the real world.

To achieve this, we cannot use $P_{\text{raw}}$ directly. We must calibrate it.

---

## Stage 4: Temperature Scaling (Obtaining the Final Confidence Score)

### Layman's Explanation
To fix the AI's cockiness, SafeLens introduces a **"Humility Knob"** called the **Temperature ($T$)**.

Think of the raw logit $z$ as someone shouting in panic. When we divide $z$ by a temperature $T > 1$ (SafeLens uses $T = 1.35$), it acts as a shock absorber:
- It **calms down extreme predictions** (e.g., turning an arrogant $96\%$ into an honest, realistic $89\%$).
- It **leaves completely uncertain predictions alone** (a tie at 50% stays at exactly 50%).
- It **preserves the rank order** (if clip A was more harmful than clip B, clip A is still higher after scaling).

The result of this temperature scaling is the **Final Calibrated Confidence Score** ($P_{\text{calibrated}}$) that is saved to the database and shown on the dashboard.

### The Formula
$$P_{\text{calibrated}} = \sigma\left(\frac{z}{T} + b\right) = \frac{1}{1 + \exp\left(-\left(\frac{z}{T} + b\right)\right)}$$

### Components & What the Formula Does
- **$z$**: The log-odds logit from Stage 1.
- **$T$ (Temperature Parameter, $T = 1.35$)**: A learned scalar tuned on a validation benchmark to minimize Expected Calibration Error (ECE).
  - If $T = 1.0$: No scaling occurs ($P_{\text{calibrated}} = P_{\text{raw}}$).
  - If $T > 1.0$ (SafeLens $T = 1.35$): Logits are pulled closer to zero, softening overconfident peaks.
- **$b$ (Platt Scaling Bias, $b = 0.0$)**: An intercept parameter used if there is a systematic class imbalance bias.
- **$\sigma(\cdot)$**: The sigmoid function mapping the scaled logit into the final probability.

### Direct Relationship Between $P_{\text{raw}}$ and Final Confidence
Since $z = \ln\left(\frac{P_{\text{raw}}}{1 - P_{\text{raw}}}\right)$, you can also express the final confidence directly in terms of $P_{\text{raw}}$:

$$P_{\text{calibrated}} = \frac{1}{1 + \left(\frac{1 - P_{\text{raw}}}{P_{\text{raw}}}\right)^{1 / T}}$$

### What Happens Step-by-Step (Numerical Example)

| Logit $z$ | Raw Probability ($P_{\text{raw}}$) | Scaled Logit ($z / 1.35$) | Final Confidence ($P_{\text{calibrated}}$) | Effect of Calibration |
| :---: | :---: | :---: | :---: | :--- |
| **$+4.0$** | $98.2\%$ | $+2.96$ | **$95.1\%$** | Overconfidence softened by $3.1\%$. |
| **$+2.5$** | $92.4\%$ | $+1.85$ | **$86.4\%$** | Moderate extreme softened by $6.0\%$. |
| **$+1.0$** | $73.1\%$ | $+0.74$ | **$67.7\%$** | Mild belief adjusted toward center. |
| **$0.0$** | $50.0\%$ | $0.00$ | **$50.0\%$** | Neutral tie stays exactly 50%. |
| **$-2.5$** | $7.6\%$ | $-1.85$ | **$13.6\%$** | Safe-side overconfidence softened. |

---

## Stage 5: Edge-Case Protections in Code

In production ([`calibrator.py`](file:///d:/Sudeep/CODING(From%20Asus%20Tuf%20Laptop)/Visual%20studio%20code%20files/0(RESUME)/Major%20Project%20Prototypes/1_MVP/SafeLens/web/app/orchestration/calibrator.py)), two safety guards ensure the math never fails:

1. **Missing Complement Recovery**: If the AI is so confident that the alternate word (e.g. `"false"`) drops out of the top-5 logprobs list, SafeLens calculates it using probability theory:
   $$p(\text{"false"}) = \max\Big(10^{-7},\; 1 - p(\text{"true"})\Big)$$
   This prevents a division-by-zero error or missing logit.
2. **Missing Logprobs Fallback**: If an upstream third-party free API model omits token logprob metadata entirely, the calibrator gracefully falls back to clamping the prompted float to $[0.0, 1.0]$:
   $$C_{\text{final}} = \min\Big(1.0,\; \max(0.0,\; C_{\text{prompt}})\Big)$$
   and tags the result as `"prompt_fallback"` for audit transparency.

---

## Summary: The Complete Journey From Token to Confidence

```
[ LLM Decision Tokens: "true" vs "false" ]
                   │
                   ▼  (Stage 1)
       Log-Odds Logit: z = ln p("true") - ln p("false")
                   │
                   ├──► (Stage 2)  Raw Softmax: P_raw = σ(z)  [Uncalibrated, Overconfident]
                   │
                   ▼  (Stage 4)
    Temperature Scaling: z_scaled = z / 1.35
                   │
                   ▼
  Final Calibrated Confidence: P_calibrated = σ(z_scaled)  [Statistically Honest]
```

| Term | What It Is | Mathematical Formula | Code Reference |
| :--- | :--- | :--- | :--- |
| **$z$ (Log-odds logit)** | Raw tug-of-war point score ($-\infty$ to $+\infty$) | $z = \ln p(\text{"true"}) - \ln p(\text{"false"})$ | [`calibrator.py`](file:///d:/Sudeep/CODING(From%20Asus%20Tuf%20Laptop)/Visual%20studio%20code%20files/0(RESUME)/Major%20Project%20Prototypes/1_MVP/SafeLens/web/app/orchestration/calibrator.py) |
| **$P_{\text{raw}}$** | Raw squashed probability ($0.0$ to $1.0$) | $P_{\text{raw}} = \frac{1}{1 + e^{-z}}$ | [`calibrator.py`](file:///d:/Sudeep/CODING(From%20Asus%20Tuf%20Laptop)/Visual%20studio%20code%20files/0(RESUME)/Major%20Project%20Prototypes/1_MVP/SafeLens/web/app/orchestration/calibrator.py) |
| **Final `confidence`** | Calibrated probability shown to users | $P_{\text{calibrated}} = \frac{1}{1 + e^{-z / 1.35}}$ | [`calibrator.py`](file:///d:/Sudeep/CODING(From%20Asus%20Tuf%20Laptop)/Visual%20studio%20code%20files/0(RESUME)/Major%20Project%20Prototypes/1_MVP/SafeLens/web/app/orchestration/calibrator.py) |
