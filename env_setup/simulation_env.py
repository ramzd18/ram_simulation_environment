from datasets import load_dataset
from typing import Optional, Tuple, List
import aiohttp
import asyncio
import random
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
import requests
import time
class MyBaseEnv(ABC):
    @abstractmethod
    async def setup(self) -> None: ...
    @abstractmethod
    async def get_next_item(self) -> Optional[any]: ...
    @abstractmethod
    async def collect_trajectories(self, item) -> List[Tuple[str, str, str, List[dict], dict]]: ...
    @abstractmethod
    async def generate_rewards(self, conversation, character1, character2, scenario) -> dict: ...

class SimulationEnvironment(MyBaseEnv):
    def __init__(self, server_url: str, vllm_server_url: str):
        self.server_url = server_url
        self.vllm_server_url = vllm_server_url
        self.customer_personas = None
        self.character_codex = None
        # List[List[Tuple[str, str, str, List[dict], dict]]]        
        self.current_batch = []
        self.session = None
        self.batch_size = 100
        self.cutomer_idx=None
        self.character_idx=None
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
        if self.cutomer_idx >= len(self.customer_personas) and self.character_idx >= len(self.character_codex):
            return None

        if self.cutomer_idx < len(self.customer_personas) and (self.character_idx >= len(self.character_codex)):
            persona = self.customer_personas[self.cutomer_idx]
            self.cutomer_idx += 1
            return await self._generate_customer_scenario(persona)

        if self.character_idx < len(self.character_codex) and (self.cutomer_idx >= len(self.customer_personas)):
            character = self.character_codex[self.character_idx]
            self.character_idx += 1
            return await self._generate_character_scenario(character)

        if self.cutomer_idx < len(self.customer_personas) and self.character_idx < len(self.character_codex):
            if random.random() < 0.5:
                persona = self.customer_personas[self.cutomer_idx]
                self.cutomer_idx += 1
                return await self._generate_customer_scenario(persona)
            else:
                character = self.character_codex[self.character_idx]
                self.character_idx += 1
                return await self._generate_character_scenario(character)

        return ""

    async def collect_trajectories(self,character1, character2) -> Tuple[any, List[any]]:
        all_conversations = []
        scenario = await self.generate_character_scenario(character1, character2)
        for conversation_num in range(4):
            conversation = []
            samped_number = random.randint(5, 7)
            for _ in range(samped_number):
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

            # Conversation is type List[dict]
            all_conversations.append((scenario, character1, character2, conversation))
        final_output= []
        for scenario,character1,character2, conversation in all_conversations:
            rewards = await self.generate_rewards(conversation, character1, character2, scenario)
            final_output.append((scenario,character1,character2, conversation, rewards))
        # final_output is type List[Tuple[str, str, str, List[dict], List[dict]]]
        self.current_batch.append(final_output)
        await self.check_batch()
        return final_output


    async def check_batch(self): 
        if len(self.current_batch) >= self.batch_size:
            async with self.session.post(
                f"{self.server_url}/collect",
                json={"items": self.current_batch}
            ) as response:
                if response.status != 200:
                    raise Exception("Failed to collect batch")
            self.current_batch = []

    

    async def generate_rewards(self, conversation, character1, character2, scenario):
        # Generate terminal rewards for each character
        terminal_prompt = f"""Given this conversation: {conversation}
        And this scenario: {scenario}
        And these characters:
        Character 1: {character1}
        Character 2: {character2}
        
        Evaluate both characters' performance based on these criteria:
        1. How authentic their responses were to their character (0-40 points)
        2. How interesting and engaging their contributions were (0-30 points)
        3. How accurate their responses were throughout the history of the conversation (0-30 points)

        Return only a JSON with fields:
        - character1_authenticity: Score for character 1's authenticity (0-40)
        - character1_engagement: Score for character 1's engagement (0-30)
        - character1_accuracy: Score for character 1's accuracy (0-30)
        - character2_authenticity: Score for character 2's authenticity (0-40)
        - character2_engagement: Score for character 2's engagement (0-30) 
        - character2_accuracy: Score for character 2's accuracy (0-30)
        - character1_feedback: Brief explanation of character 1's scores
        - character2_feedback: Brief explanation of character 2's scores"""

        async with self.session.post(
            f"{self.vllm_server_url}/generate",
            json={
                "prompt": terminal_prompt,
                "max_tokens": 500,
                "temperature": 0.3
            }
        ) as response:
            terminal_rewards = {
                "character1": {
                    "authenticity": 0,
                    "engagement": 0,
                    "accuracy": 0
                },
                "character2": {
                    "authenticity": 0,
                    "engagement": 0,
                    "accuracy": 0
                }
            }
            if response.status == 200:
                result = await response.json()
                scores = result.get("text", {})
                terminal_rewards["character1"]["authenticity"] = scores.get("character1_authenticity", 0)
                terminal_rewards["character1"]["engagement"] = scores.get("character1_engagement", 0)
                terminal_rewards["character1"]["accuracy"] = scores.get("character1_accuracy", 0)
                terminal_rewards["character2"]["authenticity"] = scores.get("character2_authenticity", 0)
                terminal_rewards["character2"]["engagement"] = scores.get("character2_engagement", 0)
                terminal_rewards["character2"]["accuracy"] = scores.get("character2_accuracy", 0)

        # Generate per-utterance scores
        utterance_scores = []
        for utterance in conversation:
            utterance_prompt = f"""Given this conversation utterance: {utterance}
            And the full conversation context: {conversation}
            
            Score this utterance on a scale of 0-10 based on:
            1. How engaging and interesting the response is (0-5 points)
            2. How directly it addresses previous points in the conversation (0-5 points)

            Return only a JSON with fields:
            - score: The numerical score (0-10)
            - feedback: Brief explanation of score"""

            async with self.session.post(
                f"{self.vllm_server_url}/generate",
                json={
                    "prompt": utterance_prompt,
                    "max_tokens": 200,
                    "temperature": 0.3
                }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    score_data = result.get("text", {})
                    utterance_scores.append({
                        "speaker": utterance["speaker"],
                        "score": score_data.get("score", 0),
                        "feedback": score_data.get("feedback", "")
                    })
                else:
                    utterance_scores.append({
                        "speaker": utterance["speaker"],
                        "score": 0,
                        "feedback": "Failed to generate score"
                    })

        return {
            "terminal_rewards": terminal_rewards,
            "utterance_scores": utterance_scores
        }
        
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


    async def _generate_customer_scenario(self, persona):
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
        return  await self.enrich_customer_persona(scenario)
    
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

    async def _generate_character_scenario(self, character):
        scenario = {
            "name": character.get("character_name", "Yilin"),
            "background": character.get("description", "Yilin is a young nun from the Hengshan Sect in Jin Yong's novel \"The Smiling, Proud Wanderer.\" Known for her innocence and kindness, she becomes friends with the protagonist Linghu Chong. Her gentle nature often puts her at odds with the violent world of martial arts."),
        }
        return await self.enrich_character_scenario(scenario)
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
    async def check_status_env(self):
        async with self.session.get(f"{self.server_url}/status") as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception("Failed to check server status")


if __name__ == "__main__":
    env = SimulationEnvironment("http://localhost:8000", "http://localhost:8001")
    print("Setting up environment")
    asyncio.run(env.setup())
    print("Setup complete")
    while True: 
        can_sample = asyncio.run(env.check_status_env())
        can_sample_value = False 
        if can_sample.status==200:
            can_sample_value = can_sample.get('can_sample', False)
        if not can_sample_value:
            time.sleep(10)
            continue
        none_counter=0
        exit_flag= False
        for _ in range(env.batch_size):
            character_1 = asyncio.run(env.get_next_item())
            character_2 = asyncio.run(env.get_next_item())
            if character_1 or character_2 is '':
                exit_flag= True
                break
            if character_1 and character_2 is None: 
                none_counter+=1
                if none_counter > 10: 
                    exit_flag= True
                    break
                continue
            else: 
                none_counter=0
        if exit_flag:
            break
        asyncio.run(env.collect_trajectories(character_1, character_2))
    requests.post(f"{env.server_url}/notify_teardown", timeout=10)

    print("DONE")

