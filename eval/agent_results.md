# Investigator agent: hidden-cause evaluation

3 scenarios with a planted root cause hidden among same-area noise (`eval/agent_cases.json`).

**Root cause found — agent: — · scripted baseline: 0/3**

| Case | Mode | Run | Found cause | Status | Tool calls | Verifier rejections | Fabricated IDs caught | Final support | Seconds | Planted reports in issue |
|---|---|---|---|---|---|---|---|---|---|---|
| upi_android14 | scripted | 1 | ❌ | incomplete | 5 | 1 | 0 | 0.5 | 0.0 | 11/12 |
| otp_jio | scripted | 1 | ❌ | incomplete | 5 | 1 | 0 | 0.0 | 0.0 | 2/10 |
| samsung_video_crash | scripted | 1 | ❌ | incomplete | 5 | 1 | 0 | 0.5 | 0.0 | 8/11 |

## Briefs

### upi_android14 — scripted
- Planted: UPI payments fail on Android 14 since app update 5.2
- Found: (no report) — segment: —
- Path: get_issue_details → extract_mentions → feedback_timeline → find_related_issues → verify_hypothesis

### otp_jio — scripted
- Planted: Login OTP SMS not delivered to Jio numbers
- Found: (no report) — segment: —
- Path: get_issue_details → extract_mentions → feedback_timeline → find_related_issues → verify_hypothesis

### samsung_video_crash — scripted
- Planted: Video playback crashes on Samsung devices after update 5.2
- Found: (no report) — segment: —
- Path: get_issue_details → extract_mentions → feedback_timeline → find_related_issues → verify_hypothesis

