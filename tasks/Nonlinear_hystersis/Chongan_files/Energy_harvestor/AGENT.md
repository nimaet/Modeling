# AGENT.md

## Project

This is a lightweight research project on implementing the main idea of the dynamics paper in `Old_sources/1-s2.0-S0020746224003470-main.pdf` with a piezoelectric energy harvesting system.

The project currently has four threads:

- Concept: what idea from the dynamics paper is being transferred.
- Theory: equations, assumptions, and reduced-order modeling.
- Numerics: MATLAB simulations, parameter studies, robustness checks.
- Experiment: future collaboration on physical validation.

## Archive Rule

`Old_sources/` is the old non-agentic project archive. It is read-only.

Agents and collaborators may read, cite, summarize, and reproduce material from `Old_sources/`, but must not edit, rename, move, delete, or save new outputs there.

When using old material, mention the source path in the new note or code. Example:

```text
Source: Old_sources/Continuous model/Continuous_harvester.m
```

## Working Structure

Create only what is useful. Suggested folders:

```text
docs/          project notes, archive summaries, meeting notes
theory/        derivations, equations, assumptions
simulations/   cleaned and reproducible MATLAB/Python scripts
results/       new figures, tables, and simulation outputs
experiments/   future lab plans, setup notes, measurement protocols
reports/       collaborator-facing memos, slides, manuscript drafts
```

Keep `Old_sources/` as the historical record. Put all new work outside it.

## Agent Workflow

For each task:

1. Read `AGENT.md` first.
2. Check whether the task depends on `Old_sources/`.
3. Make the smallest useful change or note.
4. Keep provenance clear: cite archive files, paper sections, or new assumptions.
5. For simulations, record enough parameters and script names to reproduce the result.
6. Leave short open questions when a modeling choice is uncertain.

Do not over-organize. Prefer one clear note or runnable script over a large directory system.

## Content Guidelines

For concept notes:

- State the physical idea in plain language.
- Say how it maps from the dynamics paper to the piezoelectric harvester.
- Separate confirmed understanding from speculation.

For theory notes:

- Define variables and units.
- State assumptions near the equations.
- Mark whether equations are from the paper, from old project files, or newly derived.

For simulation code:

- Start from a minimal baseline.
- Keep old scripts in `Old_sources/` untouched.
- If rewriting an old script, say which old file it came from and what changed.
- Save generated outputs in `results/`.

For experimental planning:

- Keep practical details separate from theory.
- Track needed hardware, measurable quantities, and validation targets.
- Connect proposed measurements back to the model or numerical result they test.

## Collaboration Style

Write for two readers:

- a human collaborator who needs the scientific story quickly;
- a future agent who needs enough context to continue without damaging the archive.

Use short files, explicit assumptions, and clear source references. When in doubt, document the uncertainty rather than hiding it.

