# Benchmark results

Dataset: 50 human-labelled entries (`data/sample_feedback.csv`)

| Model | Category acc. | Severity MAE | Churn precision | Churn recall | Median latency (ms) | Consistency |
|---|---|---|---|---|---|---|
| keyword-baseline | 0.82 | 0.68 | 1.0 | 0.188 | 0.0 | — |

## Misclassified by keyword-baseline (first 10)

- "OTP not received for login since yesterday. I'm locked out of my account." — label `bug`, predicted `other`
- "Keeps logging me out every few hours and I have to enter OTP again. Annoying." — label `bug`, predicted `other`
- "Login with Google button does nothing on Android 14." — label `bug`, predicted `other`
- "Videos keep buffering even on fast wifi. Other apps play fine." — label `performance`, predicted `other`
- "Getting way too many notifications, it feels like spam. Going to uninstall if th" — label `ux`, predicted `other`
- "Notifications arrive hours late, useless for live updates." — label `bug`, predicted `other`
- "Push notifications stopped working after the last update." — label `bug`, predicted `other`
- "My saved settings reset after every update." — label `bug`, predicted `other`
- "Cancelled my subscription because the app is too slow and buggy." — label `performance`, predicted `bug`
