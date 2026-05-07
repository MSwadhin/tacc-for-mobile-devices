# Slide Contents

This is a 25-slide, 30-minute deck plan. Each slide is designed for font size >= 18 pt, with minimal visible text and detailed speaker notes.

## Slide 1: Title

**Time:** 0:45

**Visual:** Full title slide, optional subtle network background from `figures/fig_mobility_graph_concept.svg`.

**On-slide text:**
- Learning Topology-Aware Cooperative Caching Policies for IoT Edge Networks
- Authors: Abdullah Al Mujahid, Asir Abrar, Monideepa Kundu Sinthi
- 30-minute research presentation

**Speaker notes:** Introduce the project as a study of how AP-like edge caches should coordinate when mobility changes the effective cooperation graph. Emphasize that the focus is not only cache hit rate, but also latency, redundancy, and update cost.

**Source:** Paper title page and abstract.

## Slide 2: One-Slide Research Story

**Time:** 1:00

**Visual:** `figures/fig_research_flow.svg`

**On-slide text:**
- Mobility creates cooperation.
- Perturbation changes placement value.
- Online refinement must justify cache rewrites.

**Speaker notes:** Walk through the high-level flow: mobility events become a graph; the graph shapes a stochastic objective; Online DQN refines the currently deployed placement; evaluation reports objective and operational costs. This slide is the roadmap for the talk.

**Source:** Introduction, Methodology, Evaluation.

## Slide 3: Why Edge Caching Matters

**Time:** 1:10

**Visual:** `figures/fig_edge_caching_motivation.svg`

**On-slide text:**
- Serve content near users.
- Reduce origin traffic.
- Improve latency and availability.

**Speaker notes:** Edge caching stores content at APs, gateways, or edge nodes. A local cache hit avoids the origin server. In IoT and MEC systems, this matters for sensing, AR, mobility, and real-time applications. The scientific question is where objects should be placed, not simply whether caching helps.

**Justification:** Access latency is the dominant operational motivation, which is why access cost is the primary objective term.

**Source:** Background and Introduction.

## Slide 4: Why Simple Caching Fails

**Time:** 1:15

**Visual:** `figures/fig_challenge_mapping.svg`

**On-slide text:**
- Local popularity ignores neighbors.
- Global popularity over-duplicates.
- Online rewriting is not free.

**Speaker notes:** Local popularity can improve local hits but misses cooperative opportunities. Global popularity stores the same objects everywhere, reducing catalog diversity. A learning agent that rewrites caches too often may reduce access cost but create unrealistic update overhead. These failures motivate topology, redundancy, perturbation, and relocation in the model.

**Justification:** Each challenge directly maps to a model component rather than being added for decoration.

**Source:** Background and Table I.

## Slide 5: Research Gap

**Time:** 1:10

**Visual:** `figures/fig_research_gap.svg`

**On-slide text:**
- Mobility-derived topology is underused.
- Static placement hides online costs.
- Weak baselines can exaggerate gains.

**Speaker notes:** Many caching studies optimize placement under fixed topology or evaluate against simple popularity baselines. This project asks a more operational question: when the graph changes, should the system modify the deployed cache, and is that modification worth the relocation cost?

**Justification:** The study includes no-update and diversified baselines specifically to avoid overclaiming.

**Source:** Related Work and Study Design.

## Slide 6: Research Questions

**Time:** 1:20

**Visual:** `figures/fig_research_questions.svg`

**On-slide text:**
- RQ1: Convert mobility to graph?
- RQ2: Does topology help?
- RQ3: Does learning add value?
- RQ4: When should we update?

**Speaker notes:** RQ1 concerns graph construction from AP handoffs. RQ2 tests graph-aware placement against popularity baselines. RQ3 tests whether DQN improves a strong structured placement. RQ4 tests sequential online behavior against Keep Current when relocation is charged.

**Justification:** The RQs correspond to the whole experimental design, not just final performance numbers.

**Source:** Research Questions section.

## Slide 7: System Model

**Time:** 1:25

**Visual:** `figures/fig_system_model.svg`

**On-slide text:**
- Nodes: AP-like edge caches.
- Edges: mobility handoff strength.
- Decision: binary placement `x_i,c`.

**Speaker notes:** The network is a weighted graph `G=(V,E,W)`. Each AP has capacity `K_i`; each content object can either be stored or not stored at a node. Demand is node-specific: `P_i,c` is the probability that node `i` requests object `c`. This lets neighboring APs have related but non-identical preferences.

**Justification:** A graph model is necessary because cooperative retrieval and redundancy depend on relationships among caches.

**Source:** System Model.

## Slide 8: Perturbation Model

**Time:** 1:10

**Visual:** `figures/fig_mobility_graph_concept.svg`

**On-slide text:**
- `G_tilde=(V_tilde,E_tilde,W_tilde)`
- Nodes removed with probability `p`.
- Edges removed with probability `p/2`.

**Speaker notes:** Perturbation represents temporary AP unavailability and weakened cooperation links. The active graph has active nodes `V_tilde` and active edges `E_tilde`. Evaluation keeps the largest connected component so cooperative retrieval remains meaningful. This is why costs are normalized over the active graph.

**Justification:** A placement that is good on the nominal graph may fail when holders disappear or paths become longer.

**Source:** Perturbed Cooperation Graph subsection.

## Slide 9: Objective Function

**Time:** 1:30

**Visual:** `figures/fig_system_model.svg` or a large equation block.

**On-slide text:**
- Access: latency and origin misses.
- Replication/redundancy: storage quality.
- Relocation: update cost.

**Speaker notes:** The objective is `J = C_access + lambda C_rep + beta C_red + mu C_rel`. Access captures local, cooperative, and origin retrieval cost. Replication captures storage/update overhead. Redundancy penalizes duplicate objects at strongly related APs. Relocation compares the current placement with the previously deployed placement.

**Justification:** Hit ratio alone is insufficient because a high-hit policy can still duplicate too much content or rewrite too aggressively.

**Source:** Problem Formulation.

## Slide 10: Dataset and Demand Strategy

**Time:** 1:15

**Visual:** `figures/fig_mobility_graph_concept.svg`

**On-slide text:**
- 20 AP nodes, 190 mobility edges.
- 48 objects, cache capacity 5.
- Zipf demand + spatial locality.

**Speaker notes:** The project records the official Dartmouth access path but uses deterministic trace-compatible mobility data because raw legacy downloads were not directly available. Content requests are modeled with Zipf popularity and AP-level locality because mobility traces often do not include application-layer object logs.

**Justification:** Zipf preserves standard caching behavior; locality prevents the unrealistic assumption that all APs request the same objects.

**Source:** Dataset and Workload Construction; run metadata.

## Slide 11: End-to-End Pipeline

**Time:** 1:15

**Visual:** `figures/fig_method_pipeline.svg`

**On-slide text:**
- Build graph.
- Generate demand.
- Optimize, refine, evaluate.

**Speaker notes:** The executable pipeline converts mobility events to a graph, creates demand, builds baselines and topology-aware placements, runs Online DQN from the currently deployed cache, and evaluates policies under repeated perturbation samples.

**Justification:** Reproducibility is a contribution: the study fixes graph construction, workload parameters, baselines, and evaluation metrics.

**Source:** Methodology and experiment code.

## Slide 12: Baselines Compared

**Time:** 1:20

**Visual:** `figures/fig_baseline_taxonomy.svg`

**On-slide text:**
- Simple: random/local/global.
- Strong: diversified/topology-greedy.
- Online: DQN vs Keep Current.

**Speaker notes:** The baseline set is intentionally strong. Diversified popularity checks whether simple catalog coverage explains the gains. Fixed Greedy+DQN checks whether learning helps as a static refinement. Keep Current checks whether rewriting the deployed cache is actually worthwhile.

**Justification:** This avoids the common mistake of comparing a complex method only against weak popularity baselines.

**Source:** Evaluation protocol.

## Slide 13: Topology-Aware Greedy Placement

**Time:** 1:15

**Visual:** `figures/fig_greedy_flow.svg`

**On-slide text:**
- Search node-content pairs.
- Estimate marginal gain under perturbation.
- Add best feasible pair.

**Speaker notes:** The greedy stage repeatedly evaluates feasible node-content additions and chooses the one with highest expected objective reduction. Candidate filtering focuses on high-demand objects so the search remains tractable. Greedy is interpretable and gives a strong warm start.

**Justification:** The placement problem is combinatorial, so a structured approximation is needed before learning.

**Source:** Algorithm 1 and placement code.

## Slide 14: Online DQN Refinement

**Time:** 1:35

**Visual:** `figures/fig_dqn_loop.svg`

**On-slide text:**
- State: cache + demand + graph features.
- Action: node-content toggle.
- Accept only if objective improves.

**Speaker notes:** The DQN does not build a full placement from scratch. It starts from the previous deployed placement. The action toggles a node-content entry; if capacity is violated, the lowest-demand cached object at that node is evicted. Reward is objective reduction relative to the previous placement. During inference, actions are accepted only when measured objective improves.

**Justification:** This makes learning a bounded local-refinement operator and keeps relocation visible.

**Source:** Algorithm 2 and DQN code.

## Slide 15: Experimental Protocol

**Time:** 1:15

**Visual:** `figures/fig_experiment_protocol.svg`

**On-slide text:**
- `p`: 0.00 to 0.25.
- 12 samples per rate.
- DQN: 85 episodes, 16 steps.

**Speaker notes:** The final run uses seed 11, 20 AP nodes, 190 edges, 48 objects, capacity 5, Zipf alpha 0.85, locality 2.8, and six perturbation rates. Objective weights are `lambda=0.02`, `beta=18.0`, and `mu=0.04`.

**Justification:** Multiple perturbation samples let the study observe both mean objective and variability.

**Source:** Evaluation Table VI and run metadata.

## Slide 16: Nominal Results

**Time:** 1:20

**Visual:** `figures/fig_nominal_baseline_bars.svg`

**On-slide text:**
- Online DQN: 2.209 objective.
- Topology greedy / Keep Current: 2.633.
- Global popularity is worst.

**Speaker notes:** At `p=0.00`, Online DQN improves the deployed topology-aware greedy cache with small relocation. Local and global popularity perform poorly because they ignore cooperation and diversity. The result supports the need for topology-aware and online-refined placement.

**Justification:** Stable-window results isolate the value of refinement before severe perturbation.

**Source:** `tables/nominal_metrics.md`.

## Slide 17: Perturbation Results

**Time:** 1:30

**Visual:** `figures/fig_objective_trends.svg`

**On-slide text:**
- Online DQN beats fixed Greedy+DQN.
- Diversified is a strong static baseline.
- High perturbation changes the update decision.

**Speaker notes:** Online DQN remains much better than fixed Greedy+DQN and topology-aware greedy across perturbation rates. Diversified popularity is competitive in moderate perturbation because broad coverage helps when nodes and edges disappear. The line plot also prepares the key p=0.25 caveat.

**Justification:** The comparison shows why DQN must be transition-aware rather than a separate fixed placement.

**Source:** `tables/perturbation_objectives.md`.

## Slide 18: Critical p=0.25 Caveat

**Time:** 1:15

**Visual:** `figures/fig_p025_closeup.svg`

**On-slide text:**
- Keep Current: 4.342.
- Online DQN: 4.415.
- No-update must be tested.

**Speaker notes:** Lower objective is better. At `p=0.25`, Keep Current is slightly better than Online DQN by 0.073. This does not invalidate Online DQN as a learned refinement method; it shows that severe perturbation needs a stricter acceptance or confidence rule before rewriting.

**Justification:** This is the most important honesty slide. It prevents overclaiming and turns the edge case into a research insight.

**Source:** Table VIII / perturbation objectives.

## Slide 19: Hit-Ratio Breakdown

**Time:** 1:15

**Visual:** `figures/fig_hit_breakdown.svg`

**On-slide text:**
- Local hit is not the whole story.
- Cooperative hits depend on topology.
- Origin misses grow under stress.

**Speaker notes:** The hit decomposition separates local hits, cooperative hits, and origin misses. A policy may have a decent hit ratio but still be worse if cooperative paths are long or if it creates excessive duplication. This motivates the objective rather than a single hit metric.

**Justification:** Multi-metric reporting is necessary for operational caching evaluation.

**Source:** Summary metrics.

## Slide 20: Cost Components

**Time:** 1:15

**Visual:** `figures/fig_weighted_cost_components.svg`

**On-slide text:**
- Replication, redundancy, relocation are guardrails.
- Relocation rises when Online DQN rewrites.
- Access remains dominant.

**Speaker notes:** The weighted non-access terms are intentionally smaller than access cost under the selected weights. Their role is not to dominate the objective; their role is to break ties, discourage wasteful duplication, and make cache churn visible.

**Justification:** The objective weights encode deployment priorities and should be calibrated for real systems.

**Source:** Online DQN metrics and Discussion.

## Slide 21: Strong-Policy Heatmap

**Time:** 1:15

**Visual:** `figures/fig_objective_heatmap.svg`

**On-slide text:**
- Strong methods trade places.
- No single static rule dominates.
- Sequential evaluation is necessary.

**Speaker notes:** The heatmap condenses strong-policy behavior across perturbation rates. It shows why the project avoids a simple "one method wins everything" narrative. The right operational policy should compare candidate refinement against no-update.

**Justification:** This slide bridges raw results to the broader scientific conclusion.

**Source:** Perturbation objective table.

## Slide 22: Research Question Answers

**Time:** 1:30

**Visual:** `figures/fig_rq_answers.svg`

**On-slide text:**
- RQ1: Handoff graph is usable.
- RQ2: Topology helps.
- RQ3: Online learning helps.
- RQ4: Update only when justified.

**Speaker notes:** RQ1 is answered by the 20-node, 190-edge handoff graph. RQ2 is supported by the gap against local/global popularity. RQ3 is supported by Online DQN outperforming fixed Greedy+DQN and topology-aware greedy. RQ4 is answered conservatively: Keep Current at `p=0.25` shows update decisions need no-update comparison.

**Justification:** The conclusions align with measured results and avoid overclaiming.

**Source:** Research Question Outcomes.

## Slide 23: Limitations

**Time:** 1:15

**Visual:** `figures/fig_limitations.svg`

**On-slide text:**
- Trace-compatible mobility, not raw trace.
- Synthetic content demand.
- Scale: 20 APs, 48 objects.
- High-p acceptance needs calibration.

**Speaker notes:** The study validates the pipeline and method under controlled reproducible conditions. Real raw traces and content logs would strengthen claims about deployment magnitude. Larger graphs require faster search and perhaps graph neural encoders. The `p=0.25` result motivates confidence-based no-update rules.

**Justification:** Limitations define the boundary of the claims and make future work precise.

**Source:** Limitations section.

## Slide 24: Future Work

**Time:** 1:10

**Visual:** `figures/fig_future_work.svg`

**On-slide text:**
- Raw traces and temporal windows.
- Acceptance calibration.
- Larger graphs and GNN policies.

**Speaker notes:** The next step is temporal online evaluation: estimate graph and demand from past windows, deploy caches, evaluate future windows. Add confidence intervals and statistical no-update thresholds. Scale to larger graphs with candidate pruning and graph encoders.

**Justification:** Each future direction addresses a specific limitation rather than being a generic extension.

**Source:** Future Work section.

## Slide 25: Takeaway

**Time:** 1:00

**Visual:** `figures/fig_takeaway.svg`

**On-slide text:**
- Mobility-derived topology should shape caching.
- DQN should refine the deployed cache locally.
- Cache rewriting must beat no-update.

**Speaker notes:** The final message is that cooperative edge caching should be evaluated as a sequential control problem. The result is not "DQN solves caching"; the result is that topology-aware structure plus bounded online learning gives a practical and interpretable framework when access, replication, redundancy, and relocation are reported together.

**Source:** Conclusion.
