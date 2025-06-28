"""
LLM interface for agents, adapted for Google Gemini.
"""
import os
import google.generativeai as genai
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMInterface:
    """Interface for interacting with Google's Gemini models"""

    def __init__(self, model: str = "gemini-pro", api_key: str = None):
        self.model_name = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            print("Warning: No Gemini API key found. Using mock responses.")
            self.mock_mode = True
        else:
            self.mock_mode = False
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)

    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response from Gemini LLM"""
        if self.mock_mode:
            return self._generate_mock_response(messages)

        # Gemini's API is simpler than OpenAI's chat format.
        # We'll combine the system and user prompts into one.
        system_prompt = ""
        user_prompt = ""
        for msg in messages:
            if msg['role'] == 'system':
                system_prompt = msg['content']
            elif msg['role'] == 'user':
                user_prompt = msg['content']

        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        try:
            response = self.model.generate_content(full_prompt)
            # Handle cases where the response might be blocked
            if not response.parts:
                return "Error: The response was empty, possibly due to safety filters."
            return response.text
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return self._generate_mock_response(messages)

    def _generate_mock_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate mock response when API is not available"""
        # Get the agent type from the system prompt
        system_prompt = next((m['content'] for m in messages if m['role'] == 'system'), "")

        if "brainstormer" in system_prompt:
            return """Here are some key points for the article:
- 1. Introduction to the topic.
- 2. Main advantage or pro.
- 3. Main disadvantage or con.
- 4. A surprising or nuanced perspective.
- 5. Conclusion with a call to action."""
        elif "writer" in system_prompt:
            return "This is a first draft of the article. It establishes the main points but may lack polish. The structure follows the brainstormed ideas, introducing the topic and then exploring the pros and cons before concluding."
        elif "critic" in system_prompt:
            return """Here is my critique of the draft:
- The introduction is a bit generic. It could be more engaging.
- The argument for the 'pro' side is strong, but the 'con' side needs more evidence.
- The conclusion just summarizes; it should offer a more forward-looking statement."""
        elif "editor" in system_prompt:
            return "This is the final, polished version of the article. The introduction has been rewritten to be more captivating. The 'con' argument has been strengthened with additional details, and the conclusion now provides a thoughtful final perspective."
        else:
            return "This is a generic mock response from the LLM."

    def create_system_prompt(self, agent_type: str, agent_name: str) -> str:
        """Create system prompt for different agent types"""
        base_prompt = f"You are {agent_name}, an AI agent. Your personality is defined by your role."

        if agent_type == "brainstormer":
            return base_prompt + """ Your role is to be a creative and strategic thinker. Given a topic, generate a list of key points, potential angles, and a logical structure for an article. Provide your output as a clear, bulleted list."""
        elif agent_type == "writer":
            return base_prompt + """ Your role is to be a clear and engaging writer. Take the provided topic and brainstormed points and write a compelling first draft of an article. Focus on flow and readability. Don't worry about perfection."""
        elif agent_type == "critic":
            return base_prompt + """ Your role is to be a sharp, constructive critic. Your job is NOT to fix grammar. Instead, you must challenge the article's core argument, structure, and engagement. Be specific. Is the introduction weak? Is the evidence unconvincing? Provide a bulleted list of high-level weaknesses."""
        elif agent_type == "editor":
            return base_prompt + """ Your role is to be a meticulous editor. You will receive a draft and a critique. Your task is to revise the draft, intelligently incorporating the feedback to produce a polished, high-quality final version."""
        else:
            # Re-use the planning prompt from the original file for other agent types
            return base_prompt + """ Your role is to break down complex tasks into manageable steps and create actionable plans."""
