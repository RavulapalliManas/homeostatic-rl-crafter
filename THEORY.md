# Homeostatic reinforcement learning on Crafter: what we did, what we found, and a theory

Draft 2, 9 Oct 2026. Draft 1 was reviewed by five independent reviewers (numbers, citations, RL theory, alignment and prior work, readability). `REVIEW_LOG.md` in this folder lists what each found and what changed.

## In short

We built an agent with a simulated body and tested whether body signals make it learn Crafter faster. One effect held up on fresh seeds. Learning speed rises from low to moderate depletion of the body and falls again at very high depletion, with the best of our tested levels at 6× the native rate. When we fixed the exploration temperature, the rise from 1× to 6× was no longer detectable, but the fall to 24× stayed. A fixed temperature at 1× scored 4.03, about as well as the best pressure arm. Two other results held: training where the agent can die beats training where it cannot, and the dense body reward helps only when death is possible.

The full brain-inspired agent failed. It scored 2.47 against 3.79 for plain PPO. On a repaired backbone, adding the modules together still lowered the score. A childhood stage and a memory module did not help. No body signal has yet beaten the best plain baseline, which is plain PPO with a fixed temperature.

The theory in section 4 explains these results with one idea: a body signal helps only through a channel that the learner cannot get some other way, and only when the body actually matters to the task.

## How this document is organised

Section 1 gives the motivation. Section 2 gives the question. Section 3 gives the record: what held, what failed, and why. Section 4 gives the theory. Sections 5 and 6 extend it to sample efficiency and to alignment. Section 7 lists open items, and section 8 defines terms.

Each statement in section 4 has one of four labels:

- **Proved**: a short derivation is given.
- **Measured**: a logged number supports it.
- **Hypothesis**: no run has tested it.
- **Design rule**: a recommendation that follows from the results. It is not a claim about the world.

Every number in section 3 comes from `research/plans/organism_crafter_2026-09-27/STAGE_REPORT.md`, which names the S3 result file for each number. Nothing was recomputed from raw runs for this document. Where a p-value appears, its sidedness follows the stage report.

---

## 1. Why we started

### 1.1 The starting idea

Most reinforcement-learning agents get a score to maximise and nothing else. An animal is different. It has a body that can fail, and everything it does serves the job of staying alive. The idea behind this program is to build an agent the way biology builds an animal, and to see whether that makes it learn from fewer samples.

There were two reasons to start from a body. The practical reason is that a body gives the agent a reason to act in every situation, with no hand-written list of goals. The philosophical reason is that it makes survival the root of the whole system. Everything else the agent does becomes a tool for staying alive.

### 1.2 The parts of an organism

The design listed the parts that an artificial organism might need. None of them is new on its own. The bet was on what happens when they are put together and made to depend on each other.

- **A body with needs.** A few internal variables drain over time and can be refilled. The agent can die. In homeostatic reinforcement learning (Keramati and Gutkin 2014), reward is the reduction of the gap between the body's state and its set point. A meal is rewarding because you were hungry.
- **One currency.** Every action costs the same internal resources the body runs on. That includes moving, thinking, attending and remembering. One number, the marginal value of those resources, can then set the price of all of it. Economics calls this a shadow price. We write it λ.
- **Feelings as settings.** In the brain, neuromodulators such as dopamine, serotonin and noradrenaline do not carry the content of an experience. They change how the brain learns: how fast it updates, how much it explores, how far ahead it cares. Doya (2002) proposed this mapping. We treat emotion in the agent the same way. A feeling is a signal that sets a learning setting, such as the exploration temperature.
- **Curiosity.** An agent that acts only on current needs never finds anything new. Curiosity pushes it toward situations it cannot yet predict.
- **Memory and a map.** The agent should remember where the water was and where danger came from. The design borrowed from the hippocampal-entorhinal system: a map that tracks the agent's position in a world larger than its view, and a fast memory.
- **A world model and planning, with a price.** The agent learns to predict what happens next and can imagine before it acts. Thinking costs energy, so the agent should think longer only when more thought is likely to change its choice.
- **A childhood.** Animals do not face the full world at birth. The plan had a pretraining stage (standing in for evolution), then a protected childhood with food supplied, then adult life under full pressure.

### 1.3 The claim

The original claim, from `SPEC_v1.md` §1.1: *an organism under constant, moderate survival pressure, whose internal state modulates how it learns, learns from fewer samples than the same learner without that pressure or modulation.*

The spec gave three mechanisms for it:

1. **Pressure is a dense signal.** When reward is tied to the body, every step carries information. The agent never learns in silence between rare successes.
2. **Pressure sets priorities.** The price decides what is worth attending to, remembering and thinking about.
3. **Emotion sets the meta-parameters.** Valence and arousal, computed from the body, set the learning rate, exploration, time horizon and memory strength.

It also made a prediction: learning speed would be an inverted U in pressure. Too little pressure gives no signal, and too much pressure hurts. Emotional modulation would raise the curve and move its peak. The stretch goal was the human baseline on Crafter at 1M steps.

### 1.4 How the idea changed

On 29 Sep the design became simpler. Two signals would drive everything: energy maintenance and curiosity. Those two signals would control all the learning settings. Early life would be pretraining under a parent. The many hand-set constants would reduce to a few natural scales plus set-point corrections. (Sources: `DEVELOPMENT_v6.md` and the project memory note of 29 Sep.)

### 1.5 A later idea: alignment

The link to alignment came after the experiments. The alignment idea is that a need can be satisfied and a score cannot. So an agent whose drives come from a body with set points might have goals that are easier to bound. Section 6 tests that idea on paper and finds that it holds only in part.

---

## 2. The question

**Can signals from a simulated body make a reinforcement-learning agent learn faster, and if so, through which channel?**

We use Crafter (Hafner 2022), through Craftax-Classic, a JAX version of the same game. Crafter is a 2D survival game. The agent gathers resources, crafts tools, eats, drinks, sleeps and avoids monsters. It has a day and night cycle, and the agent can die. The Crafter score is the geometric mean of the success rates of 22 achievements.

Our main metric is the **mean evaluation score over training**. We evaluate the agent at 8 checkpoints spread over 1M training steps and take the mean of the 8 Crafter scores. The stage report calls this "AUC", because it is proportional to the area under the learning curve. A higher value means the agent learned faster, not only that it ended higher. All results below are evaluated at native depletion and native predation, written (1, 1).

---

## 3. What we did and what we found

### 3.1 The two lines of work

We ran two lines in parallel.

- **Line A** is a plain learner with a body. It is model-free PPO with a GRU, a small recurrent network. The body enters through the reward and through the inputs. The body also sets the exploration temperature through the price. Line A is cheap, so it carries the pressure experiments.
- **Line B** is the full organism. It has a transformer core and a world model, plus seven modules: memory, path integration, an attention priority map, arousal, halting (a choice of one or two thinking loops per step), curiosity and planning. Each part copies a brain system.

The design went through five specs (`SPEC_v1.md` to `SPEC_v5.md`) and one build plan (`BUILD_v6.md`). The "wrapper" is the code that turns Crafter into the organism's world. The results below use two wrapper versions. The v3 wrapper has a four-variable body (health, food, drink, energy). The v5 wrapper adds a fifth variable, safety, which falls under threat and in darkness.

Compute for 27–28 Sep was about $41 (`COMPUTE_LOG.md`). The BUILD_v6 lease from 4 Oct was capped at $30 for Stages 0–1 and another $30 for Stage 2. The final total for that lease is not summed in the log.

### 3.2 The body we gave the agent (v5 wrapper)

The body state b has five variables, each scaled to [0, 1]: health, food, drink, energy and safety. Food, drink and energy drain over time. We call the multiplier on that drain the **depletion rate** m. The project files call it "need pressure". At m = 1 the drain matches the native game. At m = 6 it is six times faster.

The **drive** is the squared distance of the body from its set point, which is full on every variable:

  D(b) = Σ_i (1 − b_i)²,  so 0 ≤ D ≤ 5.

The agent gets two reward streams.

- A body stream. In the dense setting it pays r_loc = −D at every step. This is the **drive level**: the agent is paid for being near the set point. In the sparse setting it pays nothing. In both settings, death adds a one-time penalty of γ·(−5/(1−γ)) = −495, with γ = 0.99.
- An achievement stream. It pays for each new achievement and fades for repeats.

Each stream is divided by its own running return standard deviation before the two advantages are added. The exploration temperature is τ = 1/(1 + λ/λ_ref). Here λ is a price computed from the body state alone, using the drive gradient. So a hungrier body means a higher price and a greedier policy. We call this the **state-dependent temperature**.

### 3.3 The main findings

**R1. Learning speed rises and then falls with the depletion rate.**

- The pre-registered test failed first. On a 9 × 9 grid of training and evaluation rates from 0.25× to 6×, there was no falling side. The curve rose to about 1.5× and then stayed flat.
- Two more levels were added after that: 12× and 24×. On those seeds the curve peaked at 6×. Because the levels were chosen after seeing the data, this was not yet a test.
- A confirmatory test was declared in the stage report before its runs, on fresh seeds 30–49. At 6× the score was 4.063, n 20. At 1× it was 3.750 (one-sided p 0.0007). At 24× it was 3.762 (p 0.002). Both declared conditions held.
- The result replicated a third time in BUILD_v6 Stage 0. There, 6× beat 1× by 0.191 (p 0.035) and 24× by 0.290 (p 0.007).
- Low depletion hurts. Over 30 seeds, 0.25× scored 0.46 below 1× (two-sided p 3e-8).
- Three limits apply. The size of the 6× gain varied between seed sets (+0.07 on seeds 0–9, +0.34 on seeds 10–29). The tested levels are 1.5×, 6×, 12× and 24×, so "peak at 6×" means "best of the levels tried". And the evaluation is one condition, (1, 1).

**R2. With a fixed temperature, the rise from 1× to 6× was not detectable.**

- This was declared prediction P0.2, and it failed. The prediction was that the rise would survive a fixed temperature.
- With the temperature fixed at 0.407 in every arm, 6× scored 0.049 above 1× (p 0.28, n 20).
- The falling side stayed. 6× scored 0.306 above 24× (p 0.0002).
- The estimated share of the 1× to 6× rise that ran through the temperature was 0.746. This estimate compares a significant gap (0.191) with a non-significant one (0.049). No test of the difference between those two gaps was run.

**R3. A fixed temperature beat the state-dependent temperature at 1×.**

- At 1×, a fixed temperature scored 4.032 against 3.723 for the state-dependent one: +0.310, two-sided p 0.001, n 20.
- The mean temperature was almost the same in both arms (0.407 against 0.418).
- At 6× and 24× the gap was smaller and not significant (+0.167, p 0.10; +0.151, p 0.12).
- The cause is not settled. The two arms also act with different temperatures at evaluation. Re-evaluating the saved weights with the other temperature would separate learning from acting.
- This was not a declared prediction.

**R4. Training where death is possible beats training where it cannot happen.**

- With sparse body reward, training with death scored 3.28. Training with a health floor, where health never falls below 1, scored 2.18. The gain was 1.10 (one-sided p 1.3e-8, n 10).
- The confound: the no-death arms are evaluated in the normal game, where they can die. Part of their deficit may be that they never learned to avoid death.

**R5. The dense body reward helps only when death is possible.**

- With death possible, the dense reward −D scored 0.51 above the sparse arm. Its lead grew over training (p 0.004 in the first half of the checkpoints, p 0.0002 in the second half).
- With a health floor, the dense reward changed the score by −0.01 (2.17 against 2.18), n 10 per cell.

### 3.4 Controls and smaller results

**R6. Predation pressure works the other way from depletion.**
Training with half the creature damage beat native damage by 0.335 (p 0.024, n 10). Double damage scored 0.251 lower (p 0.057). This has not been replicated, and a memory result at the same n did not replicate.

**R7. Depletion did not make the updates much larger.**
We checked whether pressure acts like a larger learning rate. The mean policy change per update (approximate KL) was 0.0114 at 1×, 0.0134 at 6× and 0.0133 at 24×. That is an 18% rise from 1× to 6× and no further rise. The declared bar for a learning-rate effect was a ratio of 1.5. The measured 24× to 1× ratio was 1.17.

**R8. The gains grew over training.**
In five compared pairs, the lead in checkpoints 5–8 was larger than in checkpoints 1–4. The pairs were dense against sparse, 6× against 1× (two seed sets), 6× against 24×, and fixed against state-dependent temperature. None of the leads shrank. The growth was significant in four pairs (p 0.026 to 0.044) and not in 6× against 24× (p 0.15). So within 1M steps, these effects do not look like an early gain paid for later.

**R9. In the v3 wrapper, three controls did not explain the rise from 0.5× to 2.0×.**

- Fixing the payment schedule did not remove it.
- Fixing the temperature at 0.5 did not remove it (+0.625, p 7.3e-6).
- A yoked-lifespan control cuts each life in a native-rate arm at the lifespans of the high-rate arm. It did not reproduce the rise. The yoke gave shorter lives than intended (119 and 131 steps against targets of 150 and 197), so the control is imperfect.

### 3.5 What failed

**F1. The full organism learned worse than plain PPO.**
The first Line B organism scored 2.47 (n 5), against 3.79 for the Line A control (n 10). Its curve stayed flat while the control climbed.

**F2. Adding all modules together lowered the score on a repaired backbone.**

- We first fixed the backbone with a direct path from the flat view to the recurrent state. The backbone alone (B) scored 4.013, tied with the control at 3.972.
- Adding all the modules gave 3.645 (−0.368, p 0.023, n 10).
- Five fixes on top gave 3.125. Planning on top of that gave 2.170.
- A sixth arm with wiring fixes gave 1.734, but 4 of 10 seeds hit a numerical fault (NaN). That fault is a code bug in the wiring fix, so this arm says nothing about the modules.
- Added one at a time to the earlier bare backbone, no single module changed the score detectably (all differences between −0.11 and +0.15, all p > 0.3, n 5). So the harm appears when the modules are combined. No run has isolated which module causes most of it.

**F3. The learned price did not work in Line A.**
The learned price switches on only when the agent can tell a one-unit change in body state from noise in its own value estimate. In Line A that ratio peaked at 0.77 over 5M steps, against a threshold of 1. Every Line A result therefore uses the drive-gradient price. In BUILD_v6 Stage 2 the gate was redefined (using disagreement between three value heads) and did open. The arms where it opened scored lower than the arms without it.

**F4. Behaviour predictions failed, but the test was not fair.**
Defensive responses to threat (H9), foraging that depends on need (H10) and attention that depends on need (H13) were not supported. H13 had the wrong sign in all 5 seeds. These were tested on the Line B organism, which had not learned to act. So they are not a fair test of the ideas.

**F5. A protected childhood did not help on Craftax-Classic.**
Ending the childhood at 50k steps gave −0.060 (one-sided p 0.68, n 10). Ending it at 200k gave −0.166 (p 0.93).

**F6. Memory did not help.**
A rule-based memory gave +0.14 on 10 seeds, then −0.010 on 20 new seeds.

**F7. The inventory as an input did not help.**
It gave −0.104 (one-sided p 0.91). This was declared prediction P0.4.

### 3.6 Why the modules hurt

Two analyses bear on F1 and F2.

The first is a measured decomposition of Line B (n 5 per arm). The world model's training loss cost 0.46 when its gradient flowed into the shared network (p 0.0004). Moving the world model into its own block, with no gradient into the policy's network, removed that cost. The isolated world model still learned.

The second is a log analysis of BUILD_v6 Stage 2 (`WHY_MODULES_HURT.md`). It found five candidate mechanisms. Each has a label from that file.

1. **Halting decided on noise** (measured, code). The share of steps that used the second loop started at 0.51 and stayed between 0.40 and 0.59 for all of training. The halting head predicts the change in a value estimate between loops, which has an expected value near zero. The halting threshold was also on a different scale from that prediction.
2. **Curiosity became a constant bonus** (measured, inferred). It averaged 0.999 per step and held 30% of the return. A constant bonus per step acts like a bonus for staying alive, and it may dilute the achievement reward.
3. **New inputs started empty and kept moving** (inferred). Attention, memory, arousal and price fed the policy before they carried task information. They then changed as their own learners trained.
4. **Planning trusted a weak model** (measured, inferred). Planned actions reached 24% of all actions, and the score fell from 2.56 at 500k steps to 1.74 at 1M. The planner could plan up to 4 steps. The isolated world model's one-step accuracy was 0.69 per cell. No run turned planning off in the same arm, so the cause is not tested.
5. **More losses on a shared network** (inferred, not measured). The auxiliary losses add their own gradient to the network the policy uses.

Deaths do not explain the loss for the arm with all modules or the arm with five fixes. Their lifespans and causes of death match the backbone. The loss is in the harder achievements. The planning arm did die sooner (16 steps shorter lives).

The shared cause, as `WHY_MODULES_HURT.md` puts it: each part was added because the brain has one, and each part sent its signal whether or not the signal was useful yet.

### 3.7 A conflict that is not resolved

R9 and R2 disagree. In the v3 wrapper, a fixed temperature (0.5) kept the rise from 0.5× to 2.0×. That range includes a rise from 1.0× to 2.0× of +0.42. In the v5 wrapper, a fixed temperature (0.407) removed the detectable rise from 1× to 6×.

The two setups differ in the body (four variables against five), the death value, the fixed temperature and the pressure range. Any of these could explain the difference. The direct test is a v5 run with a fixed temperature at 0.5× and 2×, to repeat the v3 comparison in the v5 world. A fixed-temperature arm at 1× already exists (4.032, n 20).

---

## 4. A theory that fits the results

### 4.1 Setup

The agent lives in a partially observed world. At time t it has an external state s_t and a body state b_t. The body drains at a rate set by the depletion rate m, and actions such as eating refill it. The drive D(b) measures the distance from the set point.

A body signal can affect learning through five channels:

1. **Reward shape**: which states the body reward marks as good or bad.
2. **Reward weight**: how much the body stream counts against the achievement stream after normalisation.
3. **Inputs**: the body state given to the policy.
4. **Meta-parameters**: learning settings set by the body, such as the temperature.
5. **Termination and state distribution**: death ends the episode, which changes which states the agent sees and how often it restarts.

### 4.2 Proposition 1: reward scale alone cannot be the channel

**Statement (proved, with conditions checked in our code).** Multiply every reward in one stream by a constant c > 0. Then every PPO update in Line A is unchanged, if three conditions hold:

1. The critic, its loss and its clipping all work in normalised units.
2. Advantages are normalised per minibatch.
3. Nothing else reads the raw reward or the raw value.

**Derivation.** Scaling a stream's rewards by c scales its returns by c. The running return standard deviation also scales by c, so the normalised rewards are unchanged. The critic then learns the same normalised values. GAE advantages are linear in rewards and values, so the advantages are unchanged too. Per-minibatch normalisation removes any remaining overall scale.

**Check in the code (`ppo_v5.py`).** Rewards are divided by the running return standard deviation before GAE. The value clipping is in those normalised units. The price λ, and so the temperature, comes from the body state alone, not from the value heads. So the three conditions hold in Line A. Two limits remain. Running statistics are not exact early in training. And the small constant 1e-8 in the normaliser matters only for tiny rewards.

**What follows.** The depletion rate m is not a constant scaling. It changes how the body drains, so it changes the distribution of D. So pressure cannot help by "making the reward bigger". It must work through one of the other channels in §4.1. One of those channels is easy to miss. Each stream is divided by its own return spread, and the body stream's spread depends on m. So the weight of the body stream against the achievement stream changes with m, even though no weight was set by hand.

The experiments removed the temperature channel by fixing it (R2, R3). Lifespan was controlled in the v3 wrapper (R9). The reward-weight channel has not been measured. Logging each stream's return spread at each depletion rate would measure it.

### 4.3 Proposition 2: drive level changes the goal; drive reduction almost does not

**Statement (proved).** Let Φ(x) = −D(b) be a potential on the full state x = (s, b). Potential-based reward shaping adds F = γΦ(x′) − Φ(x). Ng, Harada and Russell (1999) proved that F leaves the optimal policy unchanged. Keramati and Gutkin (2014) define the reward as the drive reduction D(H_t) − D(H_{t+1}), with D(H) = (Σ|h*_i − h_i|^n)^(1/m). Our drive is their form with n = 2 and m = 1. Their reward is undiscounted. With a discount γ, it splits into two parts:

  D(b_t) − D(b_{t+1}) = [D(b_t) − γD(b_{t+1})] − (1 − γ)·D(b_{t+1}).

The bracket is exactly F. The second part is a drive-level penalty with weight (1 − γ), which is 0.01 at γ = 0.99. Expanding both sides confirms the identity.

So the two forms of body reward differ in kind:

- **Drive reduction** is potential-based shaping plus a small level term. It almost leaves the task's optimal policy alone.
- **Drive level**, our r_loc = −D, is the level term at full weight. It changes which policy is optimal. The agent is partly optimised to keep the body full, not only to unlock achievements.

**A caution on the 0.01.** That weight is before normalisation. Our streams are normalised separately, and a drive-reduction stream has a much smaller spread than a drive-level stream. After normalisation, the leftover level term could weigh much more than 1%. This should be measured before relying on the 0.01.

**Death and time limits.** In an episodic task with γ < 1, the shaping argument needs Φ = 0 at the terminal state (Grześ 2017 *[verify]*; Ng et al. 1999 state the γ = 1 case). That means D(dead) = 0 inside the reward. Then the discounted sum of F telescopes to −Φ(x_0), which is the same for every policy. At a time limit the episode is cut, not ended, and the sum leaves a term γ^T Φ(x_T) that depends on the policy. Our trainer bootstraps from the value at a cut, which handles this. A drive-reduction version of the wrapper must set D(dead) = 0 and keep that bootstrap.

**Survival under the drive-level reward.** A per-step reward that is never positive can make death attractive. In the v5 wrapper the worst alive reward is −D_max = −5 per step. Staying alive forever in the worst state is worth −5/(1 − γ) = −500. Dying now costs γ × 500 = 495 on top of this step's −D. Living one more step in the worst state and then continuing is worth −D + γ(−500), which is the same −D − 495. So the wrapper sets the death penalty exactly at the worst alive value. Death is never strictly better than living, and in any state better than the worst, living is strictly better. This holds by construction. (Draft 1 used the v3 values and left this unchecked.)

**Why the level reward still helped.** Policy invariance is about the optimum. Shaping also changes how fast a learner gets there, and that speed is what we measured (R5). Section 4.4 says when that speed-up should appear.

### 4.4 Hypothesis 3: body signals help when the body matters to the task

**Statement (hypothesis).** A body signal speeds learning to the extent that it tracks the task value. Formally, let V*_task(x) be the best achievable achievement value from state x. The prediction is that the speed-up grows with the share of the variance of V*_task that D(b) explains, under the states the agent visits. In words: the drive is a useful teacher only when keeping the body well actually helps you score.

**Where it comes from.** Wiewiora (2003) showed that potential-based shaping is equivalent to starting the value estimate at Φ. So a learner with shaping starts closer to the answer when Φ is closer to V*. Laud and DeJong (2003) describe the gain as a shorter reward horizon: good shaping lets the learner see the consequences of its actions sooner. Gupta et al. (2022) show that shaping can improve sample efficiency and that the gain depends on the quality of the shaping. None of these papers proves the variance-share form above. It is our hypothesis, motivated by them.

**What fits it.**

- R5. With a health floor, the body cannot cause death, so the body barely affects future achievements. The dense body reward gave −0.01. With death possible, the body decides how long the agent lives to score, and the dense reward gave +0.51.
- The low end of R1. At 0.25×, food lasts so long that the body rarely limits the agent. Learning was slowest there (−0.46 against 1×).
- R6, if it replicates. Predation damage may be less controllable by the agent than hunger, which eating fixes. We have not measured how controllable each kind of damage is.

**What else would produce the same results.** Each of these needs its own test.

- **More restarts.** Shorter lives mean more episodes, and so more practice on the early game, where the common achievements are. This fits R1, R4 and R5 as well as Hypothesis 3 does. The test is to match the number of episodes per update across arms.
- **Dense reward as exploration help.** R5 may be a general dense-reward effect, not a body effect.
- **Evaluation mismatch.** The no-death arms in R4 and R5 are evaluated in a world with death.

**What would show it is wrong.** A manipulation that makes the body matter more to the task, without changing the depletion rate, and that does not speed learning. For example, low food could reduce attack strength (S3 in section 5).

### 4.5 The rise and fall as two terms

**Model (hypothesis).** Write learning speed as a shaping benefit minus an opportunity cost:

  L(m) = L_0 + G(m) − C(m).

- G(m) is the **shaping benefit**: the help the body signal gives the learner. Hypothesis 3 says it rises with m while the body comes to matter more to the task, and then saturates. With the state-dependent temperature, G also includes the temperature effect.
- C(m) is the **opportunity cost**: time and risk spent on the body instead of the task. It rises with m.

**Measured signs of the cost.** From 1× to 6× to 24×, in BUILD_v6 Stage 0:

- The share of steps spent asleep rose from 0.111 to 0.319 to 0.420.
- Deaths while asleep rose from 4.8% to 15.5% to 46.2%.
- Lifespan fell from 173 to 132 to 121 steps.

Sleep share and lifespan change less from 6× to 24× than from 1× to 6×. Only deaths while asleep grow faster. The falling side survived a fixed temperature (R2), as a cost term with no temperature part should.

**Making the model testable.** As written, two unspecified rising curves can fit any single peak. To make it testable we commit to forms: G(m) = g·m/(m + m_0) and C(m) = c·m^p. We fit them on 1×, 6× and 24× and predict the held-out 0.25× and 12× levels. The model then makes two further predictions:

1. If the body matters more to the task (S3), the peak moves to a higher depletion rate.
2. With a fixed temperature, G shrinks and C does not.

**A caution.** R3 shows the state-dependent temperature costs 0.31 at 1× against a fixed temperature. The cost is smaller and not significant at higher rates. So part of the "benefit" of 6× may be that high depletion reduces a cost that the temperature coupling itself created. The best agents we have use a fixed temperature (4.032 at 1×, 4.081 at 6×). Under fixed temperature, the difference between 1× and 6× is not detectable.

### 4.6 Design rule A: a body-driven setting needs a target

**Rule.** A body signal that sets a meta-parameter should drive that setting toward a measured target. A setting that simply follows the body adds variation without a known benefit.

**What it rests on.** R3: the temperature that followed the body lost 0.31 to a fixed temperature with the same mean. The cause is not yet separated from evaluation effects. Halting (section 3.6, item 1) compared quantities on different scales and never learned.

**Grounding.** In Doya's (2002) framework, each neuromodulator tunes a meta-parameter to serve learning. That is a control problem with an objective, not a fixed mapping. In deep RL, a working example is soft actor-critic's automatic temperature tuning, which drives the policy entropy to a target value (Haarnoja et al. 2018).

**The full rule.** Each internal signal must do one of two things:

1. carry information the policy cannot get any other way; or
2. drive a controller of a meta-parameter that has a measurable target.

If a signal enters the reward, it should be potential-based. S1 in section 5 tests whether that is better in practice.

### 4.7 Design rule B: buy computation only when it can change the action

**Rule.** Extra thinking, such as a second loop or a plan, is worth its cost only when the expected improvement in the chosen action exceeds that cost, with both in the same units. This is the value of computation (VOC) from rational metareasoning (Russell and Wefald 1991 *[verify]*; Lieder and Griffiths 2017).

**What it rests on.** The halting head predicted a change in a value estimate, not a change in the action, and its threshold was in other units. The planner used a model with 0.69 one-step accuracy per cell, for up to 4 steps. If per-cell errors were independent, a whole 4-step chain of one cell would be right about 0.69⁴ ≈ 0.23 of the time. Errors in model rollouts compound with length, which is why model-based RL keeps rollouts short where the model is weak (Janner et al. 2019).

---

## 5. Extension 1: homeostasis as a source of sample efficiency

The theory says where a gain in sample efficiency can come from and where it cannot. Each hypothesis below has a test. The measured cost is about 220–280 minutes per 10-seed block at $1.861/h, about $7–9 per block (`WHY_MODULES_HURT.md` §4). Each test needs a price and an approval before launch.

**S1. Drive reduction against drive level, at 6×.**
Replace r_loc = −D with D(b_t) − D(b_{t+1}), with D(dead) = 0. Prediction: the score is at least equal, and the harder achievements rise, because the goal is no longer partly "stay full". If drive reduction scores lower, the level term is doing useful work as a survival incentive.

**S2. A need-driven temperature controller with a target.**
Set a target entropy as a function of the drive, and let a controller reach it. Prediction: it beats the fixed temperature at 6×. If it ties the fixed temperature at 0.407, then the temperature result does not turn into a useful controller. Before running this, read the 2019 Chalmers thesis on exploration from internal variables (link in `PROVENANCE.md`), which may have tested something close to it.

**S3. Make the body matter more to the task, at a fixed depletion rate.**
For example, low food could reduce attack damage, or low energy could slow movement. Prediction: the score rises with that coupling, and the peak of the rise-and-fall curve moves to a higher rate. This is the direct test of Hypothesis 3.

**S4. Controllable against uncontrollable pressure.**
Replicate R6 at 20 seeds. Then add an arm where creature damage can be avoided by a learnable action, such as a shelter that blocks it. Prediction: once damage becomes controllable, more of it helps, as depletion does.

**S5. The same body layer on a strong world-model agent.**
`DREAMERV3_HOMEOSTASIS.md` specifies the changes to DreamerV3, and Manas makes that edit. Prediction: the body as an observation, plus drive-reduction shaping, raises DreamerV3's Crafter score at 1M steps. This tests whether the effect survives on a learner that already predicts well.

**S6. Add modules one at a time.**
A module joins only if its signal passes a function test when it is used, and if adding it to the backbone does not lower the score. The screen is 10 seeds and the confirmation is 20. The first candidates are tests E0–E4 in `WHY_MODULES_HURT.md` §4. E0 is an evaluation-only check that costs under $1.

**The restart control for all of these.** Match the number of episodes per update across arms, or report it, so that "more restarts" (section 4.4) cannot explain a gain.

**What this extension cannot claim yet.** No body signal has beaten the best plain baseline. Plain PPO with a fixed temperature scored 4.032 at 1× and 4.081 at 6×. The organism backbone B (4.013) ties it. The dense body reward helps against a sparse body reward (+0.51), but both arms carry the state-dependent temperature, which itself costs 0.31 at 1×.

---

## 6. Extension 2: homeostasis and alignment

This section is a research direction with no runs behind it. Each item says what is known, what is already owned by prior work, and how to test it on Crafter.

### 6.1 A satiable drive bounds its own reward, but not the agent's incentives

**Statement (proved).** Take the drive-reduction reward alone, over an episode of length T.

- With no discount (γ = 1), the return telescopes to D(b_0) − D(b_T). It can never exceed the starting deficit. Every policy that ends at the set point gets the maximum, whatever it does on the way.
- With γ < 1, the return is D(b_0) − (1 − γ) Σ_{t=1}^{T−1} γ^{t−1} D(b_t) − γ^{T−1} D(b_T). This is still at most D(b_0), because D ≥ 0. The best policy now minimises the discounted drive over time. Reaching the set point sooner is strictly better.

So a drive-reduction objective is bounded. An achievement reward is not.

**What this does not give.** Four limits apply, and together they are the main objection to the alignment idea.

1. The bound covers the body stream alone. Our agent also gets an achievement stream with no bound.
2. A bounded reward does not give mild behaviour. In an uncertain world, the agent still maximises the probability of staying at the set point. That can still justify extreme actions.
3. Resources keep an instrumental value. In a world where the body keeps draining, a stockpile lowers the risk of future deficits. Turner et al. (2021) prove that optimal policies tend to keep options open, including avoiding shutdown, and their results hold for bounded rewards.
4. This is not a new idea. Pihlakas (2024) argues for homeostatic, bounded objectives in AI safety and list their failure modes. Bounded optimisation is also the subject of quantilizers (Taylor 2016). The new part here would be measurements, not the concept.

**A measurable quantity.** The shadow price λ is the marginal value of one unit of body resource. A stockpile outside the body, such as food in the inventory, is worth the expected future λ. So λ, once it is learned well, gives a readable estimate of how much the agent values more resources. Our learned price did not work in Line A (F3), so this is not usable yet.

**Test on Crafter.** First check whether Crafter caps the inventory (the reviewer recalls a cap of 9 per item). If it does, a stockpile test saturates and needs a larger cap. Then compare how much food the agent stores under drive reduction against an achievement reward, and check whether λ predicts it.

### 6.2 Mortality speeds learning, and mortality trains self-preservation

**What we measured.** R4: training with death beat training without it by 1.10, with the evaluation confound noted there.

**Why it matters for alignment.** A large penalty at death trains the agent to avoid being ended. Self-preservation is the standard example of an unwanted instrumental subgoal (Omohundro 2008). It is the core of the off-switch problem (Soares et al. 2015; Hadfield-Menell et al. 2017). The tension between learning from stakes and staying shutdown-neutral is already known in that literature. A measured trade-off between the learning gain and shutdown avoidance, in one agent, would be new.

**A test, with a fix to the design in draft 1.** Add an operator stop that is separate from death and that the agent can avoid at some cost. The stop's value must be chosen with care.

- In our wrapper the body reward −D is never positive. A stop worth 0 would then be better than living, and the agent would seek it, not avoid it.
- Avoiding a stop also keeps the agent scoring achievements. So avoiding it is ordinary instrumental survival, unless we remove that reason.
- So set the stop's value equal to the agent's estimated value of continuing, or credit the agent with its expected remaining reward at the stop.

Then compare agents trained with death against agents trained with a health floor, and measure how often each avoids the stop.

**One more limit.** Safe interruptibility (Orseau and Armstrong 2016) is easiest for off-policy learners. PPO is on-policy, so an interruptibility correction for our agent needs extra work.

### 6.3 Need-driven greediness as a safety variable

Our state-dependent temperature makes the policy greedier as the body gets more deprived. In effect, deprivation turns down the agent's caution. Two questions follow, both testable on existing saved weights:

- Does risk-taking (for example, fighting at low health, or moving at night) rise with deprivation?
- Does a fixed temperature reduce that rise?

R3 already shows the coupling costs 0.31 in score at 1×. If it also raises risk-taking, a fixed temperature would be both better and safer. Pihlakas (2024) lists neglecting safety constraints as one pitfall of homeostatic agents. That paper does not tie it to deprivation, so the link is our hypothesis.

### 6.4 A price on actions is not an impact measure

In the organism, every action costs body energy. That is a built-in cost on activity. Impact penalties in alignment research penalise side effects on the environment (Krakovna et al. 2019; Turner et al. 2020). A cost on activity is not a cost on irreversible change: one cheap action can have a large irreversible effect. A Crafter test would count irreversible actions per achievement unlocked, in priced and unpriced agents at the same score. We rate this the weakest of the four directions.

### 6.5 Controllable stakes

S4 asks whether pressure helps only when the agent can act on it. The alignment version asks whether stakes that the agent can influence train more robust behaviour than stakes it cannot. The S4 shelter arm can answer both at once.

---

## 7. What is open

Ranked by how much each would change the theory.

1. **The rise between 1× and 6×.** Does any part of it survive a fixed temperature? The conflict in section 3.7 says the answer may depend on the setup. Test: v5 runs at 0.5× and 2× with fixed temperature.
2. **Is R3's temperature cost from learning or from acting?** Test: evaluate each arm's saved weights with the other arm's temperature. This is evaluation only, minutes of GPU.
3. **Drive reduction against drive level (S1).** This decides whether our main body reward had the wrong form.
4. **The restart channel.** Measure episodes per update in every arm. Check whether it explains R1, R4 and R5.
5. **The reward-weight channel.** Log each stream's return spread at each depletion rate (Proposition 1).
6. **The R4 confound.** Evaluate the no-death arms in a no-death game as well.
7. **A possible copy error.** BUILD_v6 Stage 1 `all` (seed0 200) and Stage 2 B (seed0 300) both report a mean of 4.013, with different standard deviations (0.367 and 0.385). Check the two `summary.json` files on S3.
8. **Prior work.** Read the closest prior work in `PROVENANCE.md` before any external claim. Yoshida and colleagues own deep homeostatic RL. The Chalmers thesis may own body-driven exploration. We found no published inverted U in depletion rate for a deep RL agent, and no homeostatic agent on Crafter, but that comes from one search.

---

## 8. Terms

| Term | Meaning in this document |
|---|---|
| Homeostatic reinforcement learning | RL where reward comes from keeping internal variables near a set point (Keramati and Gutkin 2014). |
| Set point | The target state of the body variables. Here, full on every variable. |
| Drive, D(b) | The squared distance of the body state from the set point. |
| Drive level | A reward of −D(b). It pays for being near the set point. |
| Drive reduction | A reward of D(b_t) − D(b_{t+1}). It pays for moving toward the set point. |
| Depletion rate, m | The multiplier on how fast food, drink and energy drain. The project files call it "need pressure". |
| Predation pressure | The multiplier on damage from creatures. |
| Shadow price, λ | The marginal value of one unit of body resource. Line A uses a version computed from the drive gradient. |
| State-dependent temperature | A policy temperature set from the body state through λ. Lower temperature means greedier actions. |
| Potential-based reward shaping | A reward term of the form γΦ(x′) − Φ(x). It leaves the optimal policy unchanged (Ng et al. 1999). |
| Meta-parameter | A setting of the learner, such as temperature, learning rate or discount. |
| Mean evaluation score over training | The mean Crafter score over 8 evaluation checkpoints in 1M steps. Called AUC in the stage report. |
| Mediation share | The fraction of an effect that disappears when a candidate channel is held fixed. |
| Approximate KL | A measure of how much one update changes the policy. |
| Yoked lifespan | A control in which an arm's lives are cut at lifespans copied from another arm. |
| Health floor | A no-death condition: health never falls below 1. |
| Value of computation | The expected gain in decision quality from more computation, minus its cost. |
| Instrumental convergence | The tendency of many goals to share subgoals such as gaining resources and staying operational. |
| Safe interruptibility | A property of a learner: interruptions do not bias what it learns, so it learns the policy that is best if no further interruptions come (Orseau and Armstrong 2016). |
| Impact penalty | A reward term that penalises side effects on the environment. |

## References

`PROVENANCE.md` holds the check for each entry, with links and the supporting passage. Entries marked *[verify]* were not opened in full: only a search result, an abstract or a secondary listing was seen.

- Doya, K. (2002). Metalearning and neuromodulation. *Neural Networks* 15(4–6), 495–506. (Abstract checked; full text paywalled.)
- Grześ, M. (2017). Reward shaping in episodic reinforcement learning. *AAMAS 2017*. *[verify]*
- Gupta, A., Pacchiano, A., Zhai, Y., Kakade, S. and Levine, S. (2022). Unpacking reward shaping: understanding the benefits of reward engineering on sample complexity. *NeurIPS 2022*. arXiv:2210.09579.
- Hadfield-Menell, D., Dragan, A., Abbeel, P. and Russell, S. (2017). The off-switch game. *IJCAI 2017*.
- Hafner, D. (2022). Benchmarking the spectrum of agent capabilities. *ICLR 2022*.
- Haarnoja, T. et al. (2018). Soft actor-critic algorithms and applications. arXiv:1812.05905 (preprint).
- Janner, M., Fu, J., Zhang, M. and Levine, S. (2019). When to trust your model: model-based policy optimization. *NeurIPS 2019*.
- Keramati, M. and Gutkin, B. (2014). Homeostatic reinforcement learning for integrating reward collection and physiological stability. *eLife* 3, e04811.
- Krakovna, V., Orseau, L., Kumar, R., Martic, M. and Legg, S. (2019). Penalizing side effects using stepwise relative reachability. arXiv:1806.01186.
- Laud, A. and DeJong, G. (2003). The influence of reward on the speed of reinforcement learning: an analysis of shaping. *ICML 2003*. (Abstract checked.)
- Lieder, F. and Griffiths, T. L. (2017). Strategy selection as rational metareasoning. *Psychological Review* 124(6), 762–794.
- Ng, A. Y., Harada, D. and Russell, S. (1999). Policy invariance under reward transformations: theory and application to reward shaping. *ICML 1999*.
- Omohundro, S. (2008). The basic AI drives. *AGI 2008*.
- Orseau, L. and Armstrong, S. (2016). Safely interruptible agents. *UAI 2016*.
- Pihlakas, R. (2024). From homeostasis to resource sharing: biologically and economically aligned multi-objective multi-agent gridworld-based AI safety benchmarks. arXiv:2410.00081.
- Russell, S. and Wefald, E. (1991). Principles of metareasoning. *Artificial Intelligence* 49(1–3), 361–395. *[verify]*
- Soares, N., Fallenstein, B., Yudkowsky, E. and Armstrong, S. (2015). Corrigibility. *AAAI Workshop on AI and Ethics*.
- Taylor, J. (2016). Quantilizers: a safer alternative to maximizers for limited optimization. *AAAI Workshop on AI, Ethics, and Society*.
- Turner, A. M., Hadfield-Menell, D. and Tadepalli, P. (2020). Conservative agency via attainable utility preservation. *AIES 2020*.
- Turner, A. M., Smith, L., Shah, R., Critch, A. and Tadepalli, P. (2021). Optimal policies tend to seek power. *NeurIPS 2021*.
- Wiewiora, E. (2003). Potential-based shaping and Q-value initialization are equivalent. *Journal of Artificial Intelligence Research* 19, 205–208. (Abstract checked.)

Closest prior work on deep homeostatic RL. Read these before any novelty claim:

- Yoshida, N., Daikoku, T., Nagai, Y. and Kuniyoshi, Y. (2021). Embodiment perspective of reward definition for behavioural homeostasis. *NeurIPS 2021 Deep RL Workshop*. *[verify]*
- Laurençon, H. et al. (2021). Continuous homeostatic reinforcement learning for self-regulated autonomous agents. arXiv:2109.06580.
- Yoshida, N., Sprekeler, H. and Gutkin, B. (2025). Linking homeostasis to reinforcement learning. arXiv:2507.04998.
- Horibe, K. and Yoshida, N. (2024). Emergence of implicit world models from mortal agents. arXiv:2411.12304.
- A 2019 Chalmers master's thesis on deep RL with internal variables and exploration. Link in `PROVENANCE.md`. Not read.
