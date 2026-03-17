# Writing Patterns

Use these patterns to draft prose that reads like a mathematics and computational neuroscience thesis instead of generic academic text.

## Paragraph Pattern

Default paragraph shape:

1. topic sentence
2. technical development
3. interpretation or consequence
4. transition

Avoid paragraphs that mix definitions, literature review, and interpretation without clear boundaries.

## Definition Pattern

Use when introducing notation, variables, or model objects:

1. name the object
2. define it precisely
3. explain why it matters
4. state any assumptions or domain restrictions

## Theorem or Proposition Pattern

1. state the result precisely
2. give intuition in plain language
3. explain the proof strategy
4. present the proof or sketch
5. state why the result matters for the thesis

## Model Description Pattern

Use when explaining a neural model, dynamical system, or computational framework:

1. describe the biological or mathematical motivation
2. introduce states, inputs, and parameters
3. present the governing equations or update rules
4. explain assumptions
5. note what phenomenon the model is expected to capture

## Methods-to-Results Bridge

Use transitions like:

- "With the model specified, we now evaluate whether..."
- "This construction allows us to test..."
- "Under these assumptions, the relevant comparison is..."

Make the bridge explicit so the reader sees why the result section follows from the method section.

## Literature Synthesis Pattern

Do not stack summaries paper by paper. Instead:

1. name the theme
2. summarize the shared idea
3. separate major variants
4. state the unresolved limitation
5. connect that limitation to the thesis

## Discussion Pattern

Use this order:

1. strongest finding
2. interpretation
3. limitation
4. implication
5. future extension

## Style Constraints

- Prefer "we show", "we consider", "this suggests" over vague passive constructions.
- Avoid claiming novelty unless the user has support for that claim.
- Avoid overstating biological realism when the model is abstract.
- Avoid overstating mathematical generality when results depend on narrow assumptions.
- Prefer short equations-focused sentences around formal statements.
