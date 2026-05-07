# Presentation Plan

## Goal

Prepare a 30-minute research presentation that explains the full project clearly: background, motivation, research gap, problem formulation, method, implementation choices, evaluation protocol, baselines, results, limitations, and future work.

The target deck length is 25 slides, which gives roughly 60-75 seconds per slide plus a small buffer for transitions.

## Audience Assumption

The audience understands networking, IoT, or machine learning at a graduate-course level, but they may not already know cooperative caching. The presentation should therefore introduce caching concepts visually before showing equations or DQN details.

## Narrative Arc

1. **Why the problem matters:** Edge caching reduces latency, but IoT edge nodes are small, mobile, and unstable.
2. **Why simple caching is insufficient:** Local/global popularity ignore cooperation, topology, redundancy, and update cost.
3. **What the project models:** APs are nodes, handoffs are weighted edges, demand is node-content probability, placement is binary.
4. **What is optimized:** access, replication, redundancy, and relocation under perturbed graphs.
5. **What is proposed:** topology-aware greedy gives structure; Online DQN performs local transition-aware refinement.
6. **How it is evaluated:** reproducible trace-compatible mobility, Zipf-locality demand, strong baselines, repeated perturbation samples.
7. **What the results show:** Online DQN is useful as online local refinement, but no-update wins slightly at `p=0.25`, so cache rewriting should be statistically gated.
8. **What this means:** cooperative caching is a sequential control problem, not a one-shot static placement problem.

## Recommended Timing

| Segment | Slides | Time | Purpose |
| --- | --- | --- | --- |
| Opening and motivation | 1-5 | 5 min | Establish why mobility-aware cooperative caching matters. |
| Research questions and formulation | 6-10 | 7 min | Define the scientific problem and modeling choices. |
| Method | 11-15 | 7 min | Explain pipeline, baselines, greedy, DQN, and protocol. |
| Results | 16-21 | 8 min | Show objective, p=0.25 caveat, hit ratios, costs, and baseline interpretation. |
| Synthesis | 22-25 | 3 min | Answer RQs, limitations, future work, conclusion. |

## Visual Design

- Use a light background with dark text.
- Minimum font size: 18 pt. Titles: 34-44 pt. Section labels: 24-30 pt.
- Use one dominant figure per slide whenever possible.
- Use tables sparingly; when a table is needed, show only the rows/columns that support the point.
- Use consistent colors:
  - Online DQN: blue.
  - Keep Current: green.
  - Fixed Greedy+DQN: orange/red.
  - Topology-aware greedy: magenta.
  - Diversified popularity: amber.
- Avoid paragraphs on slides. Put details in speaker notes.

## Scientific Decision Justification Map

| Decision | Why It Is Scientifically Justified | Where To Present |
| --- | --- | --- |
| APs as graph nodes | APs are the caching/control points in WLAN-like IoT edge networks. | Slides 7, 10 |
| Handoffs as weighted edges | Handoffs reveal behavioral cooperation and shared user movement. | Slides 4, 7 |
| Zipf global demand | Zipf-like popularity is standard in caching research. | Slide 10 |
| Spatial locality | Edge APs serve different local populations, so demand cannot be identical everywhere. | Slide 10 |
| Perturbed graph `G_tilde` | APs/links can fail, disappear, or weaken; placement value changes under active topology. | Slide 8 |
| Access cost | Caching is primarily about latency and origin avoidance. | Slide 9 |
| Replication cost | Replicas consume cache capacity and update resources. | Slide 9 |
| Redundancy cost | Duplicating content at strongly related APs can waste cooperative catalog coverage. | Slide 9 |
| Relocation cost | Online cache updates consume bandwidth and write operations. | Slides 9, 14, 20 |
| Topology-aware greedy | Gives a strong interpretable structured placement in a combinatorial space. | Slide 13 |
| DQN local refinement | Learns useful local substitutions without learning the full placement matrix from scratch. | Slide 14 |
| Accepted-action inference | Prevents learned actions from degrading the measured objective. | Slides 14, 18 |
| Keep Current baseline | Tests whether rewriting is worth relocation cost. | Slides 12, 18 |
| Diversified popularity baseline | Tests whether simple catalog coverage explains the gains. | Slides 12, 21 |
| Reporting full costs | Hit ratio alone can hide replication, redundancy, or churn. | Slides 19-20 |

## Core Claims To Make

1. Mobility-derived topology matters for cooperative caching.
2. Strong baselines are necessary because diversified popularity can be competitive.
3. DQN is useful when used as transition-aware local refinement, not as a fixed replacement policy.
4. The no-update baseline is essential under severe perturbation.
5. The method should be understood as a sequential online control framework with explicit operational costs.

## Claims To Avoid

- Do not say Online DQN is best in every perturbation window.
- Do not imply the Dartmouth raw trace was directly used if the current run uses trace-compatible mobility data.
- Do not claim the DQN learns a full cache placement from scratch.
- Do not claim the objective weights are universal; they are deployment-dependent.
- Do not present hit ratio as the only success metric.
