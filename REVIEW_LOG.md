# Review log for THEORY.md

Draft 1 was written on 9 Oct 2026 from `STAGE_REPORT.md`, `WHY_MODULES_HURT.md`, `SPEC_v1.md`, `DREAMERV3_HOMEOSTASIS.md` and the blog post of 8 Oct. Five reviewers (Sonnet subagents) then read it independently. Each saw only the draft and its sources. Draft 2 applies their findings.

## Numbers reviewer (against STAGE_REPORT.md)

The reviewer found 17 matches, 5 mismatches, 14 overstatements, 3 untraced items and 6 untested mechanisms. Changes in draft 2:

- R1 now states that the pre-registered H1 failed first, that 12× and 24× were added after the data, and that the gain size varied between seed sets.
- R2 is now labelled as failed prediction P0.2. The mediation share carries a note that no test of the difference was run.
- R3 no longer states a cause. The evaluation-temperature confound is stated.
- R6 moved from the main findings to the controls, because it has not been replicated.
- R7 now gives the 18% rise from 1× to 6× and no longer claims "direction, not size".
- R8 now gives which growth tests were significant.
- R9 now states that the yoke gave shorter lives than intended.
- F2 now says the NaN arm reflects a code bug, and that single-module add-ons had no detectable effect.
- F3 now says the redefined price gate did open in Stage 2.
- F4 now says the behaviour tests ran on an agent that had not learned to act.
- F7 (inventory, P0.4) was added.
- Section 3.6 now matches the five mechanisms in WHY_MODULES_HURT.md, and lists the world-model gradient result separately.
- The 0.69 accuracy is now attributed to the isolated world model, per cell, with the independence assumption stated.
- Lifespan is no longer labelled "training lifespan".
- The claim that the cost "grows faster at high pressure" was wrong for sleep share and lifespan. It was corrected.
- The section 3.7 conflict no longer claims a resolution. The test was changed, because a fixed-temperature arm at 1× already exists.
- Open item 7 records a possible copy error (4.013 in two different arms).

## RL theory reviewer

- Proposition 1: the conditions are now stated and checked in `ppo_v5.py`. The rewards are normalised by the running return spread, the value clipping is in normalised units, and λ comes from the body alone. The reward-weight channel (each stream's spread depends on m) was added.
- Proposition 2: the survival bound now uses the v5 wrapper (D_max = 5, death penalty 495). In the code, the death penalty equals the worst alive value by construction (`organism_v5.py`, `v_dead`). The draft 1 bound used v3 values and was unchecked. A note on time-limit bootstrapping was added, along with a caution that the 1% weight is before normalisation.
- Proposition 3 was relabelled as Hypothesis 3. It now rests on Wiewiora 2003 and Laud and DeJong 2003, and lists alternative explanations: more restarts, a general dense-reward effect, and evaluation mismatch.
- Section 4.5 now commits to functional forms and held-out predictions.
- Propositions 4 and 5 were relabelled as design rules.
- Section 6.1: the γ = 1 and γ < 1 cases are now both stated. The false sentence "every policy that reaches the set point and stays there is optimal" was replaced.
- Terminology changes:
  - "need pressure" became "depletion rate";
  - "price" became "shadow price";
  - "price-set temperature" became "state-dependent temperature";
  - "guidance" and "maintenance cost" became "shaping benefit" and "opportunity cost".
- Checked in the code and not changed: the reviewer's concern that the PPO ratio might use untempered log-probs. `ppo_v5.py` stores and reuses the tempered log-probs, so the ratio is consistent.

## Alignment and prior-work reviewer

- The section 6.2 test had a sign flaw. With r_loc ≤ 0, a stop worth 0 is attractive. The stop's value must equal the value of continuing. This was fixed.
- Section 6.1 now states that the bound covers only the body stream, that bounded reward does not give mild optimisation, and that Pihlakas (2024) and quantilizers own the concept.
- A note was added that PPO is on-policy, so safe interruptibility needs extra work.
- Section 6.4 (internal states as a readable interface) was cut, because F3 and F4 give it no support.
- Two new directions were added: need-driven greediness as a safety variable (6.3) and controllable stakes (6.5).
- Prior work added: Yoshida et al., Laurençon et al., Yoshida, Sprekeler and Gutkin, Horibe and Yoshida, Laud and DeJong, Wiewiora, Gupta et al., and a Chalmers thesis to read before S2.
- Searches found no published homeostatic agent on Crafter and no inverted U in depletion rate for a deep RL agent. One search is not proof of absence.

## Readability reviewer

- An "In short" summary was added at the top.
- Section 1 now covers the full parts list, the two reasons for a body, the spec's three mechanisms, the childhood plan and the stretch goal. The alignment idea is labelled as later.
- Terms are defined at first use: depletion rate, drive level, mediation share, approximate KL, yoked lifespan, health floor, the wrapper versions, weaning.
- "AUC" was renamed "mean evaluation score over training".
- The closing recap (old §4.8) was removed.
- Results are split into main findings, controls and failures.

## Citation reviewer

The full table is in `PROVENANCE.md`. Changes:

- Lieder and Griffiths 2020 does not support the value-of-computation claim. It was replaced with Lieder and Griffiths 2017.
- Pihlakas 2024 has a single author. The title was corrected, and the "under deprivation" link is now marked as our hypothesis.
- The glossary definition of safe interruptibility was corrected.
- Ng et al. 1999 state the terminal-potential condition for γ = 1 only. The γ < 1 condition is attributed to Grześ 2017, which was not opened.
- Removed [verify] from entries that the checker confirmed.

## Still not done

- Russell and Wefald 1991, Grześ 2017 and Yoshida et al. 2021 were not opened.
- No number was recomputed from the raw S3 files.
- The Chalmers thesis was not read.
