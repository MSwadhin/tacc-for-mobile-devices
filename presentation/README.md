# Presentation Package

This directory prepares a 30-minute, 25-slide presentation for the project:

**Learning Topology-Aware Cooperative Caching Policies for IoT Edge Networks**

The package is organized so the actual slide deck can be built in PowerPoint, Google Slides, Keynote, or Overleaf Beamer without rethinking the research flow.

## Contents

- `PRESENTATION_PLAN.md`: overall narrative, timing, design rules, and scientific justification map.
- `SLIDE_CONTENTS.md`: slide-by-slide content, visual assignment, speaker notes, and evidence source.
- `ASSET_MAP.md`: mapping from generated figures/tables to slides and paper/result sources.
- `figures/`: presentation-ready PNG and SVG visuals.
- `tables/`: CSV and Markdown tables derived from `results/final/summary_metrics.csv`.
- `scripts/generate_presentation_assets.py`: reproducible asset-generation script.

## Regenerate Visual Assets

Run from the repository root:

```bash
python3 presentation/scripts/generate_presentation_assets.py
```

The script reads:

- `results/final/summary_metrics.csv`
- `results/final/run_metadata.json`

and writes:

- `presentation/figures/*.png`
- `presentation/figures/*.svg`
- `presentation/tables/*.csv`
- `presentation/tables/*.md`

## Slide Design Rules

- Target 25 slides for a 30-minute talk.
- Use font size >= 18 pt everywhere.
- Prefer one large visual plus one takeaway sentence.
- Use at most 3 short bullets per slide.
- Put equations in large block form only when needed.
- Keep detailed explanation in speaker notes, not on the slide.
- Use the generated SVGs when possible for sharp scaling; use PNGs if the slide tool handles raster images better.

## Important Result Nuance

The deck must not claim that Online DQN is best in every perturbation window.

At `p=0.25`, Table VIII / the generated results show:

- Keep Current objective: `4.342`
- Online DQN objective: `4.415`

Since lower objective is better, Keep Current is slightly better in that highest-perturbation window. The correct claim is:

Online DQN is the strongest learned refinement policy and consistently beats fixed Greedy+DQN and topology-aware greedy, but high perturbation requires a no-update acceptance check.
