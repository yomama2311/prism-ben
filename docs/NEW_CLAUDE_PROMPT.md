# Starting a New Claude Session — Prompt Template

Copy-paste one of these into a fresh Claude Code session in the
`/home/yuhai/project/prism-jepa` directory. Claude will auto-load `CLAUDE.md`
on session start, but a structured first message ensures it engages
correctly with the project state.

---

## Template A: General continuation (most common)

```
I'm continuing PRISM-JEPA work. Please read these in order before doing anything:

1. CLAUDE.md (project context, build commands, lessons learned)
2. docs/00_session_handover.md (current state, open decisions)
3. docs/06_roadmap.md §6.4 (Phase 3 VLM integration plan)

Then summarize back to me in 5 bullets:
- What was finished in the last session
- The single most important empirical finding
- The next open decision I need to make
- The first concrete action you'd take
- Any risks you see in that action

Don't run anything yet — wait for my green light.
```

---

## Template B: Specific task (e.g. starting Phase 3)

```
I'm continuing PRISM-JEPA. Today's goal: start Phase 3 (replace KNN oracle
with VLM-based prior).

Required reading (in order):
1. CLAUDE.md
2. docs/00_session_handover.md §6 (Open decisions)
3. docs/06_roadmap.md §6.4 (Phase 3 detail)
4. /home/yuhai/project/prism/tdmpc2/prism_head.py (PRISM-Newt's
   projection head — we'll adapt this architecture)

Then propose a Phase 3 implementation plan with:
- File layout (what new files to add to prism-jepa/)
- VLM choice (Qwen2.5-VL-3B vs alternatives) with rationale
- Training script outline (BC loss on PushT or Cube expert dataset)
- Integration point in oracle_policy.py (replacing the KNN call)

Don't write code yet. Plan first, I'll review.
```

---

## Template C: Just a question (no task to start)

```
Question about PRISM-JEPA: <your question>

Before answering, please read (only what's relevant):
- CLAUDE.md (project overview)
- The specific docs/ file most relevant to my question (look at docs/README.md
  index)

Then answer concisely. If the question requires running code or testing,
stop and tell me what you'd run instead of running it.
```

---

## Template D: Resume after a long break (>1 week)

```
I'm coming back to PRISM-JEPA after a break. Catch me up:

1. Read CLAUDE.md and docs/00_session_handover.md
2. Run `git log --oneline | head -10` to see recent commits
3. Read the most recent doc in docs/ (probably docs/0X_*.md with the
   highest number) — that's where the latest findings are

Summarize:
- The 3-5 most important things I know about PRISM-JEPA
- What I was working on right before I stepped away
- The first thing to do today

Then wait for my decision on what to work on.
```

---

## Tips for working with new Claude on this project

1. **Always activate venv before running anything:**
   ```bash
   cd /home/yuhai/project/prism-jepa
   source .venv/bin/activate
   export STABLEWM_HOME=/home/yuhai/project/prism-jepa/.stable-wm
   ```
   Tell Claude to do this in every command if it forgets.

2. **The single most surprising thing about this codebase:** the StandardScaler
   bug (CLAUDE.md §"Critical Lessons" #1). Mention this if Claude is
   debugging an unexpected SR=0 result.

3. **Tworoom failing is intentional**, not a regression — see
   `docs/07_phase2_results.md` §7.3. If Claude tries to "fix" Tworoom by
   tuning sigma or KNN parameters, redirect them.

4. **Don't trust auto-generated SR numbers** without verification — early in
   the session, the bash script's grep capture failed silently. The actual
   ground-truth SR is in the per-task `*_results.txt` files under
   `.stable-wm/<task>/`. Check there if numbers look weird.

5. **PRISM-JEPA is a sister project of PRISM-Newt** at
   `/home/yuhai/project/prism`. Don't conflate. Newt is for CoRL 2026 (May 28
   deadline); JEPA is for NeurIPS 2027 / RSS 2027 follow-up.

6. **For VLM work**, Newt-PRISM has a working Qwen2.5-VL pipeline at
   `/home/yuhai/project/prism/tdmpc2/prism_vlm.py` and a projection head at
   `prism_head.py`. Reference these but do NOT copy blindly — JEPA's input
   format and action space differ.

7. **If Claude proposes a big refactor**, push back. The current code is
   working and validated. Phase 3 should be additive (new files), not
   restructuring.
