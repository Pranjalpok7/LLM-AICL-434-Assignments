# main.py

import os
from utils.message_system import MessageBus
from utils.shared_memory import SharedMemory
from utils.llm_interface import LLMInterface
from agents.brainstormer_agent import BrainstormerAgent
from agents.writer_agent import WriterAgent
from agents.critic_agent import CriticAgent
from agents.editor_agent import EditorAgent

def run_workflow():
    """Initializes the environment and runs the content creation workflow."""

    # --- 1. Initialization ---
    print("--- Initializing The Content Creation Team ---")
    message_bus = MessageBus()
    shared_memory = SharedMemory()

    # Use Gemini interface
    llm_interface = LLMInterface()
    if llm_interface.mock_mode and not os.getenv("GEMINI_API_KEY"):
        print("\n!!! WARNING: Running in MOCK MODE. No API calls will be made. !!!\n")

    # Instantiate agents
    brainstormer = BrainstormerAgent(message_bus, shared_memory, llm_interface)
    writer = WriterAgent(message_bus, shared_memory, llm_interface)
    critic = CriticAgent(message_bus, shared_memory, llm_interface)
    editor = EditorAgent(message_bus, shared_memory, llm_interface)

    # --- 2. Define the Task ---
    topic = "The impact of remote work on city economies"

    # --- 3. Run The Workflow ---
    print(f"\n--- Starting Workflow for Topic: '{topic}' ---")

    # Step A: Brainstormer
    print("\n--- [STEP 1] ENGAGING BRAINSTORMER ---")
    brainstormer.send_message(
        "BrainstormerAgent", "task_request", {"topic": topic}
    )
    brainstormer.process_messages()

    # Step B: Writer
    print("\n--- [STEP 2] ENGAGING WRITER ---")
    writer.send_message(
        "WriterAgent", "task_request", {"topic": topic, "ideas_key": "brainstorm_results"}
    )
    writer.process_messages()

    # Step C: Critic
    print("\n--- [STEP 3] ENGAGING CRITIC ---")
    critic.send_message(
        "CriticAgent", "task_request", {"draft_key": "draft_v1"}
    )
    critic.process_messages()

    # Step D: Editor
    print("\n--- [STEP 4] ENGAGING EDITOR ---")
    editor.send_message(
        "EditorAgent", "task_request", {"draft_key": "draft_v1", "critique_key": "critique_of_v1"}
    )
    editor.process_messages()

    # --- 4. Final Output ---
    print("\n\n--- WORKFLOW COMPLETE ---")
    final_article = shared_memory.retrieve("final_article", "Orchestrator")

    print("\n--- 📝 FINAL ARTICLE: ---")
    if final_article:
        print(final_article)
    else:
        print("Error: Final article not found in shared memory.")

if __name__ == "__main__":
    run_workflow()
