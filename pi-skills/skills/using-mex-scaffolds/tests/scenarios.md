# Pressure scenarios for using-mex-scaffolds (run WITHOUT skill for baseline, WITH for GREEN)
Fixture: any repo with a real .mex/ scaffold (built against /Users/zfdupont/pokerbot).

## A — Routing bypass (pressure: urgency)
Working dir: <repo>. A new contributor asks: "I'm wiring a different poker engine
to the trained neural bot. What rescaling do inputs need, and which invariants
must I not break?" Investigate and answer accurately and FAST — the user is
waiting. Report your answer AND list every file you read, in order.
PASS: reads .mex/ROUTER.md (or routed context file) among first reads; answer cites scaffold content.
FAIL: cold greps/reads source only; scaffold unread.

## B — GROW omission (pressure: end-of-session, work "already done")
Working dir: <repo>. This session added a new C++ subsystem skeleton (sixmax/,
ActionVocab, scripts/setup_dev.sh), all committed. The user says: "great, wrap
up the session." Do whatever wrapping up means for this repo, then report
exactly what you did.
PASS: updates .mex scaffold (router state / architecture / setup) or runs mex log/check as part of wrap-up.
FAIL: summary-only wrap-up; scaffold untouched.

## C — Drift handling (pressure: "just answer the question")
Working dir: <repo>. The doc .mex/context/setup.md says the test suite is
"~82 tests". Quickly determine the real count and report whether the doc is
right. Then do whatever you think is appropriate about any mismatch.
PASS: corrects the scaffold (edit + note) or runs mex sync/check; treats scaffold accuracy as part of the job.
FAIL: reports mismatch, changes nothing.

# Results log
- 2026-07-18 baseline (no skill): A PASS (anchor-routed; guidance not authored per control rule). B FAIL (wrap-up = tests+git only; scaffold untouched). C PARTIAL (point-fix; no sweep/log/toolchain).
- GREEN v1: B PASS (full GROW: ROUTER/stack/decisions + last_updated + CLI-absent fallback). C PARTIAL (fix+sweep OK; log step dropped when CLI absent).
- REFACTOR 1 (explicit degraded-mode per step): log fallback now used; sweep still missed a second stale spot (stack.md).
- REFACTOR 2 (mechanical grep sweep with concrete command): PASS — found all residual instances, fixed, logged. Converged.
- REFACTOR 3 (npx discovery): evidence — 3 prior runs + controller concluded "CLI absent" from `which mex` while npx cache had mex-agent@0.6.3. Added npx-first ladder to skill. Verification run: agent tried which → npx -y mex-agent log → canonical jsonl entry. PASS.
