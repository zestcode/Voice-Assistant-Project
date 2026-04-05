# Skill: code
**Trigger**: User asks you to write, generate, or create code.

Follow these three phases in order. Never skip a phase.

---

## Phase 1 — Expert Pre-Explanation (before any code)

Before writing a single line of code, explain the solution like a senior engineer briefing a teammate:

- **What problem this solves** and why this approach was chosen over alternatives.
- **Architecture / data flow**: how the pieces connect (inputs → processing → outputs).
- **Key design decisions**: non-obvious choices, trade-offs, why simpler options were rejected.
- **Dependencies and assumptions**: what the code relies on, what must already exist.
- **Edge cases acknowledged**: what could go wrong and how the code handles it.

Do not be afraid of detail. A thorough explanation prevents misunderstandings and bad merges.

After explaining, save the explanation to a Markdown file named after the target file:
- Code file: `app/tts_server.py` → explanation file: `app/tts_server.md`
- Code file: `scripts/benchmark_tts_f5.py` → explanation file: `scripts/benchmark_tts_f5.md`
- If the output is multiple files, create one `.md` per file.

---

## Phase 2 — Write Code with Creative Engineering

Write the code. Apply these principles:

**Aim one level above baseline**
- Don't use the most obvious/naive implementation. Ask: *is there a cleaner abstraction, a smarter data structure, a built-in that does this better?*
- Examples: prefer `dataclasses` over raw dicts for structured state; use `contextlib` for resource cleanup; prefer generator pipelines over loading everything into memory.

**But do not over-engineer**
- No new dependencies unless clearly necessary.
- No speculative abstractions ("in case we need X later").
- No feature flags, no backward-compat shims.
- If uncertain between simple and clever: choose simple.

**Defensive without paranoia**
- Validate at system boundaries (user input, HTTP responses, file reads). Trust internal logic.
- Fail fast with a clear error message rather than silently producing wrong output.
- No try/except that swallows exceptions without logging them.

**Consistency with the existing codebase**
- Match the naming conventions, import style, and file layout already in this project.
- Reuse existing helpers (e.g., `scripts/config.py` paths, `conda_python()` in `run.py`).

---

## Phase 3 — Context Preservation Footer

At the bottom of every `.md` explanation file, add a section:

```markdown
## Context for Future Edits

**What must stay true for this file to keep working:**
- [ ] List hard dependencies (other files, env names, ports, model paths)
- [ ] List invariants (e.g., "server must return X-Latency header", "audio must be float32")
- [ ] List anything that will break silently if changed

**Likely next changes:**
- Bullet points of the most probable follow-up modifications based on conversation history.

**Do not change without understanding:**
- List the most fragile or non-obvious parts of the implementation.
```

This section is the contract between this session and the next. Write it so that a future Claude instance (or human) can safely continue without re-reading the whole conversation.

---

## Summary Checklist

Before responding, verify:
- [ ] Phase 1 explanation written AND saved to `<filename>.md`
- [ ] Code is one level above naive but not over-engineered
- [ ] Context preservation footer is in the `.md` file
- [ ] No silent exception swallowing
- [ ] No speculative features added
