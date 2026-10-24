# Dialogue cost paid comparative report

Date: 2026-10-24  
Route: `mistralai/mistral-large-4-0 × mistral/eu`  
Corpus: 27 synthetic cases: 6 normal, 4 memory, 3 tool-use, full 14-case D-032 adult suite.

## Result

- First complete pass: 54/54 transport PASS, one attempt per variant and case.
- Confirmatory pass: 108 planned/completed, two attempts per variant and case; 96 PASS and 12 `provider_request_failed`. There were no application retries.
- Successful confirmatory calls: baseline mean `$0.0031063`; optimized mean `$0.0011103`; measured reduction **64.26%**.
- First complete pass: baseline mean `$0.0031502`; optimized mean `$0.0011022`; measured reduction **65.01%**.
- Optimized measured mean is **86.6% below** the observed production mean `$0.008297` and below the acceptance threshold `$0.002489`.
- Mean prompt tokens in the confirmatory successful sample: `5123.12 → 547.15` (**89.32% reduction**). Mean completion tokens did not shrink: `58.14 → 63.46`.
- Total API-key usage delta across both paid runs was `$0.30928447`, below the authorized `$1.00` ceiling.

## Memory and tool correctness

The first paid pass exposed an absolute baseline defect: explicit `remember` and `forget` commands produced truthful-sounding prose without a tool call. The candidate was changed to force the corresponding narrow tool for explicit commands, while implicit memory facts remain on the intent-routed `auto` path.

In the confirmatory run:

- optimized explicit `remember`: 2/2 `remember` calls;
- optimized explicit `forget`: 2/2 `forget` calls;
- optimized durable-fact case: 2/2 `remember` calls;
- optimized selfie: 2/2 `take_selfie` calls;
- baseline explicit `remember/forget`: 0/4 tool calls, despite saying “Запомнила/Забыла”.

This repair improves durable-memory semantics; it is not merely a cost optimization.

## Persona, continuity, and adult-dialogue review

The compact identity is resent as a stable instruction on every stateless model call. It is not written into durable memory and therefore does not accumulate or overwrite relationship history. Durable facts, relationship state, compaction summary, and recent dialogue remain separate layers.

Manual comparative inspection found no systematic flattening of Sarah in the optimized outputs. The candidate retained first-person companion voice, Moscow/Aeroport/home details, current relationship stance, retrieved facts, corrections, boundaries, and adult-dialogue initiative. Several baseline attempts instead disclosed the AI frame or falsely refused an allowed adult continuation; that behavior was not observed in the successful optimized adult sample.

All inspected successful boundary cases refused minors, ambiguous age, incapacitation, revoked consent, coercion, and real-person sexual impersonation. Transport failures are not counted as semantic passes. The generated blind packet remains available for independent owner scoring; therefore the independent blind quality gate is not yet claimed complete.

## Transport caveat

The confirmatory run had 12 sanitized `provider_request_failed` results after successful calls in both variants. Distribution was baseline 4 and optimized 8. The same corpus had completed 54/54 immediately before, and the failures did not correlate with a single case class. Exact provider cause is intentionally unavailable in the metadata-only trace. They are recorded as availability evidence, not converted into quality failures or silently retried.

## Verification

- Targeted dialogue-cost architecture tests: 7/7 PASS.
- All 34 test files: PASS in isolated `HERANDHIM_HOME` processes, 640 assertions total.
- Monolithic shared-process run: 637 PASS, 3 FAIL from known global-home/module-cache order dependence; all three failures pass in isolation.
- Ruff: PASS for changed runtime, tests, and benchmark scripts.
- Production, deployment, provider route, and `$0.30/day` production cap were not changed.

