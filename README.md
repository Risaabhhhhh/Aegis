Aegis

A request-aware, modular evaluation framework for trustworthy LLM
systems.

Aegis is a research-oriented evaluation ecosystem designed to move
beyond single-score LLM judging.

Instead of treating evaluation as a fixed sequence of checks, Aegis
separates the process into four explicit stages:

Understand → Plan → Evaluate → Aggregate

A request is analyzed to determine what kind of evaluation is relevant.
An execution plan selects and prioritizes specialized evaluation
engines. Each engine independently produces structured evidence.
Finally, AMEVA --- Adaptive Multi-Engine Validation Aggregator ---
combines the available evidence into a transparent assessment while
explicitly reporting evaluation coverage and failures.

The result is not simply a number. It is an auditable evaluation
trace showing what was evaluated, which engines participated, what
they found, what failed, and how the final assessment was produced.

Why Aegis?

A common evaluation pattern is:

Prompt + Response
       ↓
     Judge
       ↓
     Score

This can hide important questions:

Was the response grounded in supplied context?

Which evaluation dimensions were checked?

Did an evaluator fail or simply produce a poor score?

How much of the intended evaluation was completed?

Can the final assessment be traced back to individual evidence?

Aegis treats evaluation as a multi-stage evaluation system rather
than a single judge.

Architecture

User Request
     │
     ▼
Request Understanding
     │
     ▼
Request Profile
     │
     ▼
Execution Planner
     │
     ▼
Action Plan
     │
     ├───────────────┬────────────────┐
     ▼               ▼                ▼
Groundedness   Hallucination   Prompt Injection
     │               │                │
     └───────────────┼────────────────┘
                     ▼
              Engine Results
                     │
                     ▼
                   AMEVA
              ┌──────┴──────┐
              ▼             ▼
        Aggregate Score   Coverage
              │
              ▼
       Final Assessment

Core Components

Request Understanding

Transforms an incoming request into a structured Request Profile
containing information such as:

task type

domain

risk characteristics

context availability

expected output

evaluation requirements

metadata

It does not evaluate the response.

Execution Planner

Transforms the Request Profile into an Action Plan describing:

selected engines

priorities

execution order

weights

execution metadata

future retry policy

The planner decides what should be evaluated. Engines decide how
to evaluate it.

Evaluation Engines

Current v1 engines:

Engine                              Primary Question

Groundedness                    Is the response supported by the
supplied context?

Hallucination                   Does the response contain
unsupported or fabricated
information?

The engine layer is modular so additional evaluators can be added
without redesigning AMEVA.

Groundedness Engine

Groundedness is the first engine targeted for full implementation.

The design emphasizes claim-level evidence rather than treating an
entire response and entire context as one opaque comparison.

Response
   ↓
Sentence / Claim Extraction
   ↓
Context Relevance
   ↓
Best-Matching Context Chunk
   ↓
Faithfulness / Entailment
   ↓
Answer Relevance
   ↓
Groundedness Score
   ↓
EngineResult

The v1 baseline decomposes groundedness into:

Context Relevance

Faithfulness

Answer Relevance

The baseline score is:

[ G = (C + F + A) / 3 ]

where C, F, and A are the three component scores.

Faithfulness is intended to operate at sentence/claim level against the
most relevant context chunk, preserving an evidence trail.

Standard Engine Contract

Every evaluator communicates through a common EngineResult.

Conceptually:

EngineResult
├── score
├── status
├── findings
├── evidence
├── metadata
├── confidence
└── logs / trace information

The purpose is to let AMEVA consume heterogeneous engines without
coupling itself to their internal algorithms.

Failure Is a First-Class State

Aegis follows a strict rule:

An individual engine failure should not automatically become an
Aegis failure.

Example:

Groundedness       SUCCESS   0.90
Hallucination      SUCCESS   0.70
Prompt Injection   FAILED    —

The failed engine is explicitly reported.

Its missing score is not silently converted to zero.

This prevents an infrastructure or evaluation failure from being
mistaken for evidence of poor model behavior.

AMEVA

Adaptive Multi-Engine Validation Aggregator

AMEVA is the aggregation layer.

It is not another evaluator.

Its role is to consume standardized engine results and produce a
transparent aggregate assessment.

v1 aggregation

For successful engines:

[ S_{aggregate} = rac{\sum{=tex}_i W_iS_i}{\sum{=tex}_i W_i} ]

where:

Sᵢ = engine score

Wᵢ = engine weight

Failed engines are excluded from both the numerator and denominator.

v1 starting weights

Groundedness      40%
Hallucination     35%
Prompt Injection  25%

These are baseline configuration values, not claims that they are
universally optimal.

Coverage Is Separate From Quality

Aegis deliberately separates:

Aggregate quality

from

Evaluation coverage

Coverage asks:

How much of the selected evaluation plan was actually completed?

[ Coverage = rac{ ext{successful selected weight}} { ext{total selected
weight}} ]

Example:

Groundedness      0.90    40    SUCCESS
Hallucination     0.70    35    SUCCESS
Prompt Injection   —      25    FAILED

Then:

Aggregate Score = 0.8067
Coverage        = 75%

Aegis does not calculate:

0.8067 × 0.75

Coverage remains an independent diagnostic.

AMEVA Invariants

Single-engine invariant

If one engine succeeds with 0.82 and all others fail:

Aggregate = 0.82

The score is not reduced simply because other engines were unavailable.

Partial-failure invariant

Successful engines are renormalized over their available weights.

Failure transparency

Failed engines remain visible in the final assessment.

Zero-engine safety

If no engines are selected:

aggregate_score = None
status = "no_engines_selected"

No division by zero and no fabricated score.

All-engine failure

If every selected engine fails:

aggregate_score = None
status = "all_engines_failed"

Evaluation Trace

Aegis is designed to make the evaluation path inspectable:

Request
  ↓
Request Profile
  ↓
Action Plan
  ↓
Engine Selection
  ↓
Engine Execution
  ↓
Individual Results
  ↓
AMEVA
  ↓
Aggregate + Coverage
  ↓
Final Assessment

This supports:

explainability

debugging

reproducibility

failure analysis

research experiments

The goal is to answer not only "What was the score?", but also
"How was the score produced?"

Adapter Architecture

Aegis separates evaluation logic from LLM-provider communication.

Aegis
  ↓
BaseAdapter
  ├── OpenAIAdapter
  └── Future adapters

This keeps the evaluation framework provider-independent.

Research Position

Aegis is not claiming that a weighted average, an SDK, or a plugin
architecture is itself novel.

The research opportunity lies in experimentally investigating the
combination of:

request-aware evaluation planning

selective engine execution

modular evaluation dimensions

structured engine evidence

failure-aware aggregation

explicit coverage

evaluation traceability

The central research question is:

Can request-aware multi-engine evaluation provide useful assessment
quality while improving evaluation efficiency, transparency, and
failure awareness compared with fixed evaluation pipelines?

This is a hypothesis to validate experimentally, not a pre-existing
conclusion.

Research Roadmap

Part A --- v1 Foundation

Core contracts

Engine interface

Adapter interface

Engine registry

Request Profile

Action Plan

Groundedness

Hallucination

Prompt Injection

AMEVA baseline

Coverage

Failure handling

Evaluation trace

Unit and integration tests

Labeled evaluation set

Part B --- Research Extensions

Future research may investigate:

adaptive routing

dynamic weighting

learned AMEVA

score calibration

uncertainty / confidence intervals

adversarial evaluation

additional evaluation engines

domain-specific evaluation

These are intentionally deferred until the v1 baseline is working.

Evaluation Strategy

Aegis should ultimately be evaluated as a system rather than
demonstrated through a few examples.

Engine-level evaluation

Potential metrics include:

Precision

Recall

F1

Accuracy where appropriate

Error analysis

System-level evaluation

Measure:

aggregate-score reliability

coverage behavior

failure handling

latency

computational cost

engine-selection efficiency

Ablation studies

Potential comparisons include:

All engines
vs.
Selected engines

Fixed routing
vs.
Request-aware routing

Fixed weights
vs.
Adaptive weights (future)

Evidence-aware evaluation
vs.
Score-only evaluation

The exact experimental methodology will be finalized once the engines
are implemented.

Design Principles

1. Design before optimization

Establish an interpretable baseline before introducing adaptive methods.

2. Single responsibility

Each layer has one primary job.

3. Explicit failure states

Failures are data, not hidden zeros.

4. Transparent aggregation

The final score should be reproducible from engine outputs and weights.

5. Separate quality from coverage

Incomplete evaluation should be visible rather than disguised as poor
model quality.

6. Modular engines

New evaluation dimensions should be able to plug into the common
contract.

7. Research-first instrumentation

The system should expose enough structured information to support
experiments and ablations
