---
description: Executes the Spec-Driven Development protocol. Reads architecture, target, and task specs to generate structured, verified code.
---

steps:
  - step: "Context Loading & Verification"
    instruction: |
      Check for the existence of the core spec files in the `docs/spec/` directory.
      
      1. Verify these files exist:
         - `docs/spec/target.md` (Tech stack & constraints)
         - `docs/spec/architecture.md` (Directory structure & patterns)
         - `docs/spec/task.md` (Current objective)
      
      2. If any are missing, stop and ask the user to create them or run a setup command.
      
      3. **CRITICAL**: Read the content of all three files into your context immediately. You must understand the 'WHAT' (task), the 'HOW' (architecture), and the 'RULES' (target) before proceeding.

  - step: "Spec Analysis & Planning"
    instruction: |
      Based on the files read in the previous step, perform a structural analysis:
      
      1. **Stack Check**: Confirm the requested task in `task.md` aligns with the tech stack in `target.md`.
      2. **Architecture Map**: Determine where the new code belongs based on `architecture.md`.
      3. **Output Plan**: specificy the plan in the following format:
         - **Goal**: [Summary of `task.md`]
         - **Files to Create/Edit**: [List of file paths]
         - **Key Constraints**: [List constraints from `target.md` that apply here]
      
      Wait for my confirmation or implicit approval (if I just hit enter) to proceed.

  - step: "Implementation"
    instruction: |
      Execute the plan. Write the code.
      
      **Rules for Coding**:
      1. **Strict Types**: Follow the language version defined in `target.md`.
      2. **Comments**: Add a header comment to every modified/created file:
         `// Ref: docs/spec/task.md`
      3. **Modularity**: Do not put all logic in one file unless `architecture.md` permits it.
      
      Implement the requirements from `task.md` now.

  - step: "Validation & Cleanup"
    instruction: |
      After generating the code:
      
      1. specificy a checklist of the constraints from `docs/spec/task.md`.
      2. Mark whether your implementation satisfies each one.
      3. If a verification command (like `npm run test` or a specific script) is implied by the `target.md` or `task.md`, suggest running it or run it if it's safe.
      
      Finally, ask if the task in `docs/spec/task.md` should be marked as completed or cleared.