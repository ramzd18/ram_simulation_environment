from datasets import load_dataset
from typing import Optional, Tuple, List
import aiohttp
import asyncio
import random

class SimulationEnvironment(MyBaseEnv):
    def __init__(self, server_url: str, vllm_server_url: str):
        self.server_url = server_url
        self.vllm_server_url = vllm_server_url
        self.customer_personas = None
        self.character_codex = None
        self.current_batch = []
        self.session = None

    async def setup(self) -> None:
        # Load datasets
        self.customer_personas = load_dataset("CordwainerSmith/CustomerPersonas", split="train").shuffle(seed=42).select(range(6000))
        self.character_codex = load_dataset("NousResearch/CharacterCodex", split="train").shuffle(seed=42).select(range(2000))
        self.cutomer_idx=0   
        self.character_idx=0
        self.session = aiohttp.ClientSession()
        
        async with self.session.post(f"{self.server_url}/setup") as response:
            if response.status != 200:
                raise Exception("Failed to setup server")

    async def get_next_item(self) -> Optional[any]:
        if self.current_batch:
            if self.cutomer_idx >= len(self.customer_personas):
                character = self.character_codex[self.character_idx]
                self.character_idx += 1
                return self._generate_character_scenario(character)
            elif self.character_idx >= len(self.character_codex):
                persona = self.customer_personas[self.cutomer_idx]
                self.cutomer_idx += 1
                return self._generate_customer_scenario(persona)
            
            if random.random() < 0.5:
                persona = self.customer_personas[self.cutomer_idx]
                self.cutomer_idx += 1
                scenario = self._generate_customer_scenario(persona)
            else:
                character = self.character_codex[self.character_idx]
                self.character_idx += 1
                scenario = self._generate_character_scenario(character)
            
            return scenario
        return None

    async def collect_trajectories(self) -> Tuple[any, List[any]]:
        all_conversations = []
        character1 = self.get_next_item()
        character2 = self.get_next_item()
        scenario = self.generate_character_scenario(character1, character2)
        
        # Run 4 separate conversations for GRPO
        for conversation_num in range(4):
            conversation = []
            for _ in range(random.randint(5, 7)):
                char1_prompt = f"""Given the scenario: {scenario}
                And the conversation history: {conversation}
                Generate the next response for character 1: {character1}"""
                
                async with self.session.post(
                    f"{self.vllm_server_url}/generate",
                    json={
                        "prompt": char1_prompt,
                        "max_tokens": 500,
                        "temperature": 0.7
                    }
                ) as response:
                    if response.status == 200:
                        char1_response = await response.json()
                        conversation.append({"speaker": "character1", "text": char1_response["text"]})

                char2_prompt = f"""Given the scenario: {scenario}
                And the conversation history: {conversation}
                Generate the next response for character 2: {character2}"""
                
                async with self.session.post(
                    f"{self.vllm_server_url}/generate", 
                    json={
                        "prompt": char2_prompt,
                        "max_tokens": 500,
                        "temperature": 0.7
                    }
                ) as response:
                    if response.status == 200:
                        char2_response = await response.json()
                        conversation.append({"speaker": "character2", "text": char2_response["text"]})

            # Add completed conversation to list
            all_conversations.append(conversation)

        # Send all conversations to collection server at once
        async with self.session.post(
            f"{self.server_url}/collect",
            json={"scenario": scenario, "conversations": all_conversations}
        ) as response:
            if response.status != 200:
                return scenario, []

        return scenario, all_conversations

    async def evaluate(self) -> float:
        # Get accumulated conversations from server
        async with self.session.post(f"{self.server_url}/dispatch") as response:
            if response.status == 200:
                data = await response.json()
                conversations = data.get("queue", [])
                
                # Simple evaluation metric: average conversation length
                if conversations:
                    return sum(len(conv) for conv in conversations) / len(conversations)
        return 0.0
    

    # TODO: Wrap this in an instructor call to get valid json
    async def generate_character_scenario(self,character1, character2): 
        prompt = f"""Given these two characters, suggest a specific, focused topic for them to have an in-depth discussion about. 
        The topic should be concrete and specific (not broad like "politics" or "science"), and could be something that neither character is an expert in.
        Consider their backgrounds, expertise, and potential points of view.

        Character 1: {character1}
        Character 2: {character2}

        Generate a discussion topic that includes:
        1. The specific topic or question to discuss
        2. Why this topic would create an interesting dynamic between these characters
        3. A few key points or angles they might explore

        Format the response as a JSON with fields:
        - topic: The specific discussion topic
        - rationale: Why this creates an interesting dynamic
        - key_points: List of 3-4 specific aspects to explore

        Return only the JSON."""

        async with self.session.post(
            f"{self.vllm_server_url}/generate",
            json={
                "prompt": prompt,
                "max_tokens": 500,
                "temperature": 0.7,
                "top_p": 0.9
            }
        ) as response:
            if response.status == 200:
                result = await response.json()
                discussion_scenario = result.get("text", "").strip()
                return discussion_scenario
            return None


    def _generate_customer_scenario(self, persona):
        scenario = {
            "name": f"{persona['first_name']} {persona['last_name']}",
            "background": persona["generated_persona"],
            "personality": persona["personality"],
            "age_range": persona["age_range"],
            "job_title": persona["job_title"],
            "location": persona["location"],
            "interests": persona["interests"],
            "state_of_mind": persona["state_of_mind"],
        }
        return  self.enrich_customer_persona(scenario)
    
    async def enrich_customer_persona(self, persona):
        # Prepare the prompt for vLLM to generate a hyperrealistic profile
        prompt = f"""Given the following basic customer information, create an extremely detailed and hyperrealistic profile that makes this person feel like a real human being. Include specific details, quirks, and realistic life experiences that would make sense for someone with this background.

        Basic Information:
        Name: {persona['first_name']} {persona['last_name']}
        Age Range: {persona['age_range']}
        Job Title: {persona['job_title']}
        Location: {persona['location']}
        Current State of Mind: {persona['state_of_mind']}
        Interests: {', '.join(persona['interests'])}

        Please generate a comprehensive profile that includes:
        1. Detailed personal history
        2. Current lifestyle
        3. Relationships and social dynamics
        4. Personal quirks and habits
        5. Emotional state and psychological profile
        6 Social Views and Beliefs

        Make this profile extremely hyperealistic. Based on this profile I should be able to emualte how this personal woudl respond to people and events. Return 
        the profile and nothign else."""

        async with self.session.post(
            f"{self.vllm_server_url}/generate",
            json={
                "prompt": prompt,
                "max_tokens": 1000,
                "temperature": 0.4,
                "top_p": 0.9
            }
        ) as response:
            if response.status == 200:
                result = await response.json()
                enriched_profile = result.get("text", "").strip()
                return enriched_profile

    def _generate_character_scenario(self, character):
        scenario = {
            "name": character.get("character_name", "Yilin"),
            "background": character.get("description", "Yilin is a young nun from the Hengshan Sect in Jin Yong's novel \"The Smiling, Proud Wanderer.\" Known for her innocence and kindness, she becomes friends with the protagonist Linghu Chong. Her gentle nature often puts her at odds with the violent world of martial arts."),
        }
        return self.enrich_character_scenario(scenario)
    async def enrich_character_scenario(self, character):
        # Prepare the prompt for vLLM to generate a detailed character profile
        prompt = f"""Given the following basic character information, create an extremely detailed and rich character profile that makes this character feel like a real person. Include specific details about their personality, motivations, and background that would influence how they interact with others.

        Basic Information:
        Name: {character['name']}
        Background: {character['background']}

        Please generate a comprehensive character profile that includes:
        1. Detailed personality traits and mannerisms
        2. Core motivations and goals
        3. Relationships and social dynamics
        4. Personal history and significant events
        5. Emotional tendencies and psychological characteristics
        6. Beliefs, values and worldview
        7. Typical behavioral patterns and reactions

        Make this profile extremely detailed and realistic. Based on this profile, one should be able to accurately predict and emulate how this character would behave and respond in various situations. Return the profile and nothing else."""

        async with self.session.post(
            f"{self.vllm_server_url}/generate",
            json={
                "prompt": prompt,
                "max_tokens": 1000,
                "temperature": 0.4,
                "top_p": 0.9
            }
        ) as response:
            if response.status == 200:
                result = await response.json()
                enriched_profile = result.get("text", "").strip()
                return enriched_profile

    async def __del__(self):
        if self.session:
            await self.session.close()
