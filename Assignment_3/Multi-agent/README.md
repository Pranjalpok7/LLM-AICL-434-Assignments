## Project Summary: The AI Content Creation & Polishing Team

This project implements a sophisticated **collaborative multi-agent system** designed to automatically produce a high-quality written article from a single topic. Powered by Google's Gemini API, the system simulates a human editorial team by assigning distinct roles to specialized AI agents: a Brainstormer, a Writer, a Critic, and an Editor.

The core objective is to move beyond simple, one-shot content generation. Instead, it mimics the iterative and critical process that human teams use to refine work, resulting in a more coherent, well-structured, and polished final product.

---

### The Agents (The Team)

The system is composed of four distinct agents, each with a unique "personality" and function defined by its system prompt:

1.  **`BrainstormerAgent` (The Idea Generator):** Receives a topic and generates a structured outline with key points and a logical flow for the article.
2.  **`WriterAgent` (The First Drafter):** Takes the brainstormed ideas and writes a complete first draft of the article, focusing on getting the core concepts and narrative down.
3.  **`CriticAgent` (The Quality Gate):** This is the key innovative agent. It reviews the first draft not for grammatical errors, but for high-level issues: a weak argument, a boring introduction, lack of evidence, or a poor structure. It provides specific, constructive feedback.
4.  **`EditorAgent` (The Final Polisher):** Receives both the original draft and the critic's feedback. Its job is to intelligently synthesize this information, rewriting and revising the text to produce the final, high-quality article.

### The Workflow (How They Collaborate)

The agents work in a sequential, assembly-line fashion, passing their work product to the next specialist in the chain:

1.  **Brainstorming:** The process starts with the `BrainstormerAgent` creating an outline.
2.  **Drafting:** The `WriterAgent` uses this outline to write the first draft.
3.  **Critique:** The `CriticAgent` reviews the draft and produces a list of critiques.
4.  **Editing & Finalizing:** The `EditorAgent` takes the draft and the critiques to produce the final version.

This collaboration is orchestrated using the project's core utilities:
*   A **`MessageBus`** sends tasks and instructions to the appropriate agent.
*   A **`SharedMemory`** system acts as a central workspace where agents store and retrieve the artifacts (ideas, drafts, critiques) at each stage.

### Key Innovations and Highlights

*   **Iterative Refinement:** The project's strength lies in its multi-step process, where the quality of the output is improved at each stage, especially through the critique-and-edit loop.
*   **Specialized Role-Playing:** It demonstrates the power of using carefully crafted prompts to make a single LLM model (Gemini) perform distinct, specialized tasks effectively.
*   **Quality Through Critique:** The inclusion of a dedicated `CriticAgent` forces the system to self-assess and improve its own work, a crucial step towards more reliable and higher-quality AI generation.
*   **Modular Architecture:** The system is easily extensible. One could add a `FactCheckerAgent` after the Critic or a `SEO_OptimizerAgent` after the Editor to further enhance the workflow.
