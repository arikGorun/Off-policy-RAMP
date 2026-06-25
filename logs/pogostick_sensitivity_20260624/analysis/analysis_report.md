# Pogo-stick sensitivity analysis

Completed runs analyzed: 30 / 30 expected runs.

## Coverage by condition

- PPO easy: complete=5, partial=0, missing=0
- HER easy: complete=5, partial=0, missing=0
- PPO medium: complete=5, partial=0, missing=0
- HER medium: complete=5, partial=0, missing=0
- PPO hard: complete=5, partial=0, missing=0
- HER hard: complete=5, partial=0, missing=0

## Highlights

- Best completed run: `her_pogo_stick_easy_seed2_300eps` with success_rate=0.650, final_success_ma25=0.720.
- Worst completed run: `ppo_pogo_stick_hard_seed2_300eps` with success_rate=0.233, final_success_ma25=0.120.

## Aggregate summary

| algorithm | difficulty | seeds_completed | success_rate_mean   | success_rate_std     | final_success_ma25_mean | final_success_ma25_std | mean_steps_mean    | mean_steps_std     | mean_reward_mean    | mean_reward_std      | mean_episode_seconds_mean | mean_episode_seconds_std | mean_rl_batch_seconds_mean | mean_rl_batch_seconds_std | planner_used_rate_mean |
| --------- | ---------- | --------------- | ------------------- | -------------------- | ----------------------- | ---------------------- | ------------------ | ------------------ | ------------------- | -------------------- | ------------------------- | ------------------------ | -------------------------- | ------------------------- | ---------------------- |
| her       | easy       | 5               | 0.63                | 0.02094967514996091  | 0.6880000000000001      | 0.08671793355471512    | 147.70066666666668 | 3.2822074211657557 | 0.63                | 0.02094967514996091  | 2.390323374400575         | 0.06058370053788664      | 1.244362703133394          | 0.03031949088657978       | 0.9966666666666667     |
| ppo       | easy       | 5               | 0.5313333333333333  | 0.026098105508084526 | 0.552                   | 0.07694153624668539    | 169.42266666666666 | 5.667363878677045  | 0.5313333333333333  | 0.026098105508084526 | 2.0231460428003145        | 0.3894516386872928       | 1.0191130357337146         | 0.18542110855190494       | 0.8320000000000001     |
| her       | hard       | 5               | 0.284               | 0.017543596489254343 | 0.24000000000000005     | 0.07999999999999999    | 201.59666666666666 | 4.346374606752417  | 0.284               | 0.017543596489254343 | 15.256213247599508        | 0.31211655009200295      | 5.050138636667009          | 0.40565718983641585       | 0.13                   |
| ppo       | hard       | 5               | 0.2626666666666667  | 0.02046677524401167  | 0.264                   | 0.09208691546577072    | 204.63533333333334 | 4.953313481342013  | 0.2626666666666667  | 0.02046677524401167  | 14.60201761766659         | 1.5314288433630039       | 4.294337696066824          | 0.33358366946198365       | 0.13933333333333334    |
| her       | medium     | 5               | 0.37866666666666665 | 0.016431676725154977 | 0.42400000000000004     | 0.11523888232710346    | 188.49666666666667 | 6.248817665944669  | 0.37866666666666665 | 0.016431676725154977 | 6.766425225066875         | 0.9667762780898888       | 2.5854089398668147         | 0.2941911338630587        | 0.9966666666666667     |
| ppo       | medium     | 5               | 0.3446666666666666  | 0.021807236311728182 | 0.336                   | 0.13740451229854134    | 193.61599999999999 | 4.805264127322588  | 0.3446666666666666  | 0.021807236311728182 | 6.736629398533841         | 0.26218032665874713      | 2.5990899659996307         | 0.15570016680664306       | 0.9966666666666667     |

## Notes

- Hard-difficulty runs are currently missing from the analyzed batch and are excluded from aggregate plots.
- `ppo_pogo_stick_easy_seed0_300eps` is also missing from the analyzed batch.