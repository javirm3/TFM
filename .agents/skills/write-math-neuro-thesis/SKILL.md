---
name: write-math-neuro-thesis
description: Structure, outline, and draft a master's thesis in mathematics and computational neuroscience. Use when Codex needs to help with thesis planning, chapter architecture, literature-to-outline synthesis, mathematical exposition, model/method sections, results narratives, discussion writing, revision for coherence, or converting rough notes into academically structured prose for `.md`, `.tex`, or document drafts.
---

# Write Math Neuro Thesis

## Overview

Use this skill to turn scattered thesis ideas, papers, notes, equations, and draft text into a defensible thesis structure and readable academic prose. Keep the work concrete, evidence-aware, and aligned with the conventions of mathematics and computational neuroscience.

## Workflow

1. Identify the immediate deliverable.
2. Recover the thesis context from existing artifacts before drafting.
3. Choose the right chapter or section pattern.
4. Produce an outline before long-form prose unless the user explicitly asks for direct drafting.
5. Write with explicit claims, assumptions, and transitions.
6. Revise for logical flow, notation consistency, and evidentiary support.

## Identify the Deliverable

Pin down what the user needs now. Common deliverables:

- thesis title and research question refinement
- full thesis outline
- chapter outline
- literature review synthesis
- methods/model draft
- results narrative
- discussion or conclusion draft
- rewrite for clarity, rigor, or style
- bridge text between sections

If the request is broad, narrow it to one artifact and make progress on that artifact.

## Recover Context

Read only the materials needed for the current task: thesis proposal, supervisor notes, existing draft chapters, bibliographic notes, paper summaries, equations, figure captions, or code/model descriptions.

Extract and keep visible:

- thesis goal
- core mathematical objects or models
- computational neuroscience problem setting
- main claims or hypotheses
- evidence available now
- unresolved gaps or missing citations
- notation that must remain consistent

Do not invent citations, datasets, results, or theorem claims. Mark missing support explicitly.

Also separate:

- what is already supported by current notes, analyses, or figures
- what is still roadmap, aspiration, or future work

Structure the thesis around the first set, not the second.

## Obsidian And Zettelkasten Workflow

When the user works from Obsidian or another note system:

- prefer editing the existing note they pointed to instead of returning a parallel draft
- search for existing note names before creating new concept notes
- reuse the user's own terminology from project notes and meeting notes
- prefer compact reference notes that define one concept, explain why it matters for the thesis, and link to related literature notes or project notes
- when updating an outline, patch the existing outline block instead of appending a second outline

For notes outside the current writable workspace:

- discuss the proposed structure briefly first
- ask for explicit confirmation before editing
- after confirmation, modify the existing note directly if permissions allow

## Default Thesis Spine

If the user has not provided an institutional template, default to a five-chapter structure:

1. Introduction
2. Methods
3. Results
4. Discussion
5. Conclusion

In this format:

- keep background and related work inside the Introduction unless the user wants a separate chapter
- explain behavioral tasks briefly in the Introduction as motivation and problem setting
- put full technical task, dataset, preprocessing, and covariate details in Methods
- organize Methods from descriptive models to mechanistic models when the thesis develops a modeling pipeline

## Choose the Section Pattern

Use [chapter-blueprints.md](references/chapter-blueprints.md) for section-level structure. Select the smallest pattern that matches the task.

Default chapter logic:

1. Motivate the problem.
2. Define the mathematical or scientific objects precisely.
3. Present the method, model, or theorem flow.
4. Report results or implications.
5. Interpret limitations and next steps.

For writing moves specific to mathematics and computational neuroscience, use [writing-patterns.md](references/writing-patterns.md).

## Draft in Two Passes

First pass:

- Produce a compact outline with section goals and bullet claims.
- Order definitions before dependent arguments.
- State where empirical evidence, figures, or citations are needed.

Second pass:

- Expand bullets into prose.
- Keep paragraphs single-purpose.
- Use explicit transitions such as problem -> method, method -> result, result -> interpretation.
- Prefer precise statements over rhetorical filler.

## LaTeX Outline Conventions

When the user is drafting thesis structure in LaTeX:

- prefer giving a chapter and section skeleton that can be pasted directly into the thesis
- write formal thesis-ready prose for chapter titles, section titles, and any non-callout text; do not leave substantive guidance only as loose comments outside callouts
- when placeholder guidance is useful, use callout blocks rather than plain `%` comments when the user's document already uses them
- default to `todo` callouts for drafting prompts unless the user has asked for a different convention
- semantic callouts such as `warning`, `note`, and `tip` are optional and should be used only when they add clear value
- use `warning` for unresolved conceptual risks or claims that need evidentiary support
- use `note` for content that should be explained in that section
- use `tip` for writing or structural guidance
- keep callouts short and specific so they act as drafting prompts, not mini-paragraphs
- avoid naming specific tasks, datasets, or collaborators in headings unless the user has explicitly chosen to foreground them in the thesis structure

## Writing Rules

- Define notation once, then reuse it consistently.
- Distinguish assumptions, definitions, results, and interpretations.
- When presenting equations or models, explain what each object means before discussing consequences.
- When summarizing literature, group papers by idea, method, or limitation instead of listing them one by one.
- When describing simulations or experiments, separate setup, metrics, and findings.
- When uncertainty exists, write the cautious version.
- Prefer direct, formal prose over inflated academic language.

## Revision Checklist

Before finalizing a draft, check:

- does each section answer a clear question?
- are definitions introduced before use?
- does every paragraph connect to the chapter goal?
- are claims matched with citations, derivations, data, or explicit caveats?
- are notation and terminology stable across sections?
- does the discussion distinguish contribution, limitation, and future work?

## Output Pattern

Unless the user asks otherwise, return:

1. a short diagnosis of the current draft state
2. a proposed outline or revised structure
3. the drafted or revised text
4. a short list of unresolved issues such as missing citations, missing derivations, or unclear claims
