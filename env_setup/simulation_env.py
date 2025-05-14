from datasets import load_dataset
from typing import Optional, Tuple, List
import aiohttp
import asyncio
import random
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
import requests
import time
import instructor
from openai import OpenAI
from pydantic import BaseModel

class MyBaseEnv(ABC):
    @abstractmethod
    async def setup(self) -> None: ...
    @abstractmethod
    async def get_next_item(self) -> Optional[any]: ...
    @abstractmethod
    async def collect_trajectories(self, item) -> List[Tuple[str, str, str, List[dict], dict]]: ...
    @abstractmethod
    def generate_rewards(self, conversation, character1, character2, scenario) -> dict: ...


class RewardConfig(BaseModel):
        terminal_rewards: dict
        utterance_scores: list

class Reward_Scores(BaseModel):
        character1_authenticity: int
        character1_engagement: int 
        character1_accuracy: int
        character2_authenticity: int
        character2_engagement: int
        character2_accuracy: int

class Utterance_Scores(BaseModel):
        score: int

class Sceanrio(BaseModel):
        topic: str
        key_points: List[str]
class Str_Response(BaseModel):
        response: str

class SimulationEnvironment(MyBaseEnv):
    def __init__(self, server_url: str, vllm_server_url: str, client=None):
        self.server_url = server_url
        self.vllm_server_url = vllm_server_url
        self.client = client
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

    async def collect_trajectories(self, character1, character2) -> Tuple[any, List[any]]:
        all_conversations = []
        scenario = await self.generate_character_scenario(character1, character2)
        print("SCENARIO GENERATED, length: ", len(scenario))
        for conversation_num in range(4):
            conversation = []
            samped_number = random.randint(5, 7)
            for _ in range(samped_number):
                char1_prompt = f"""Given the scenario: {scenario}
                And the conversation history: {conversation}
                Generate the next response for character 1, given their description: {character1}. 
                You are the character 1. Respond with exactly what they should say in this scenario.
                
                Return a JSON with a single field:
                - response: The exact response the character should say"""

                response = self.client.chat.completions.create(
                    model="NousResearch/DeepHermes-3-Llama-3-3B-Preview",
                    messages=[{"role": "user", "content": char1_prompt}],
                    max_tokens=400,
                    temperature=0.7,
                    response_model=Str_Response
                )
                conversation.append({"speaker": "character1", "text": response.response })

                char2_prompt = f"""Given the scenario: {scenario}
                And the conversation history: {conversation}
                Generate the next response for character 2, given their description: {character2}. 
                You are the character 2. Respond with exactly what they should say in this scenario.
                
                Return a JSON with a single field:
                - response: The exact response the character should say"""
                
                response = self.client.chat.completions.create(
                    model="NousResearch/DeepHermes-3-Llama-3-3B-Preview", 
                    messages=[{"role": "user", "content": char2_prompt}],
                    max_tokens=400,
                    temperature=0.7,
                    response_model=Str_Response
                )
                conversation.append({"speaker": "character2", "text": response.response})

            all_conversations.append((scenario, character1, character2, conversation))
        final_output = []
        for scenario, character1, character2, conversation in all_conversations:
            rewards = self.generate_rewards(conversation, character1, character2, scenario)
            final_output.append((scenario, character1, character2, conversation, rewards))
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

    def generate_rewards(self, conversation, character1, character2, scenario):
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
        - character2_accuracy: Score for character 2's accuracy (0-30). 
        Respond with only the JSON."""

        response = self.client.chat.completions.create(
            model="NousResearch/DeepHermes-3-Llama-3-3B-Preview",
            messages=[{"role": "user", "content": terminal_prompt}],
            max_tokens=1500,
            temperature=0.3,
            response_model=Reward_Scores,
        )
        scores = response

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
        
        if scores:
            terminal_rewards["character1"]["authenticity"] = scores.character1_authenticity
            terminal_rewards["character1"]["engagement"] = scores.character1_engagement
            terminal_rewards["character1"]["accuracy"] = scores.character1_accuracy
            terminal_rewards["character2"]["authenticity"] = scores.character2_authenticity
            terminal_rewards["character2"]["engagement"] = scores.character2_engagement
            terminal_rewards["character2"]["accuracy"] = scores.character2_accuracy

        utterance_scores = []
        for utterance in conversation:
            utterance_prompt = f"""Given this conversation utterance: {utterance}
            And the full conversation context: {conversation}
            
            Score this utterance on a scale of 0-10 based on:
            1. How engaging and interesting the response is (0-5 points)
            2. How directly it addresses previous points in the conversation (0-5 points)

            Return only a JSON with fields:
            - score: The numerical score (0-10)
"""
            response = self.client.chat.completions.create(
                model="NousResearch/DeepHermes-3-Llama-3-3B-Preview",
                messages=[{"role": "user", "content": utterance_prompt}],
                max_tokens=400,
                temperature=0.3,
                response_model=Utterance_Scores
            )
            score_data = response
            
            utterance_scores.append({
                "speaker": utterance["speaker"],
                "score": score_data.score,
                "feedback": score_data.feedback
            })

        return {
            "terminal_rewards": terminal_rewards,
            "utterance_scores": utterance_scores
        }
        
    async def generate_character_scenario(self, character1, character2): 
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
        - key_points: List of 3-4 specific aspects to explore

        Return only the JSON."""

        response = self.client.chat.completions.create(
            model="NousResearch/DeepHermes-3-Llama-3-3B-Preview",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.7,
            top_p=0.9,
            response_model=Sceanrio
        )
        response_json = response
        response_topic = response_json.topic
        response_key_points= response_json.key_points
        return f"Scenario: {response_topic}, Key Points To Talk About: {response_key_points}"

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
        prompt = f"""Given the following basic customer information, create an extremely detailed and hyperrealistic profile that makes this person feel like a real human being. Include specific details, quirks, and realistic life experiences that would make sense for someone with this background.

        Basic Information:
        Name: {persona['name']} 
        Age Range: {persona['age_range']}
        Job Title: {persona['job_title']}
        Location: {persona['location']}
        Current State of Mind: {persona['state_of_mind']}
        Interests: {', '.join(persona['interests'])}
        Background: {persona['background']}
        Personality: {persona['personality']}

        Please generate a comprehensive profile that includes:
        1. Detailed personal history
        2. Current lifestyle
        3. Relationships and social dynamics
        4. Personal quirks and habits
        5. Emotional state and psychological profile
        6 Social Views and Beliefs

        Make this profile extremely hyperealistic. Based on this profile I should be able to emualte how this personal woudl respond to people and events.
        
        Return a JSON with a single field:
        - response: The detailed profile text"""

        response = self.client.chat.completions.create(
            model="NousResearch/DeepHermes-3-Llama-3-3B-Preview",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.4,
            top_p=0.9,
            response_model=Str_Response
        )
        return response.response

    async def _generate_character_scenario(self, character):
        scenario = {
            "name": character.get("character_name", "Yilin"),
            "background": character.get("description", "Yilin is a young nun from the Hengshan Sect in Jin Yong's novel \"The Smiling, Proud Wanderer.\" Known for her innocence and kindness, she becomes friends with the protagonist Linghu Chong. Her gentle nature often puts her at odds with the violent world of martial arts."),
        }
        return await self.enrich_character_scenario(scenario)
    async def enrich_character_scenario(self, character):
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

        Make this profile extremely detailed and realistic. Based on this profile, one should be able to accurately predict and emulate how this character would behave and respond in various situations.
        
        Return a JSON with a single field:
        - response: The detailed character profile text"""

        response = self.client.chat.completions.create(
            model="NousResearch/DeepHermes-3-Llama-3-3B-Preview",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.4,
            top_p=0.9,
            response_model=Str_Response
        )
        return response.response

    async def __del__(self):
        if self.session:
            await self.session.close()
    async def check_status_env(self):
        async with self.session.get(f"{self.server_url}/status_check") as response:
            if response.status == 200:
                return await response.json()
            else:
                raise Exception("Failed to check server status")

async def main():
    env = SimulationEnvironment("http://localhost:5000", "http://localhost:8000")
    print("Setting up environment")
    await env.setup()
    print("Setup complete")
    while True:
        can_sample = await env.check_status_env()
        can_sample_value = can_sample.get('can_sample', False)
        if not can_sample_value:
            await asyncio.sleep(10)
            continue
        while True:
            try:
                status_response = requests.get("http://localhost:8000/health", timeout=5)
                if status_response.status_code == 200:
                    print("VLLM server GOOD TO GO")
                    break
                else:
                    print("VLLM server NOT READY")
                    await asyncio.sleep(10)
            except requests.exceptions.RequestException:
                print("Could not connect to vLLM server")
                await asyncio.sleep(10)
                continue
        client = instructor.from_openai(OpenAI(base_url=env.vllm_server_url+"/v1",api_key="ollama"),mode=instructor.Mode.JSON)

        if client:
            env.client = client
        else: 
            print("Failed to setup client")
            await asyncio.sleep(10)
            continue
        none_counter = 0
        exit_flag = False
        for _ in range(env.batch_size):
            print("Getting next item")
            character_1 = await env.get_next_item()
            character_2 = await env.get_next_item()
            print("GOTH BORTH CHARACTER")
            if character_1 or character_2 == '':
                print("True exit flag")
                exit_flag = True
                break
            if character_1 and character_2 is None: 
                none_counter += 1
                if none_counter > 10: 
                    exit_flag = True
                    break
                continue
            else: 
                none_counter = 0
        if exit_flag:
            break
        await env.collect_trajectories(character_1, character_2)
    requests.post(f"{env.server_url}/notify_teardown", timeout=10)
    print("DONE")

if __name__ == "__main__":
    asyncio.run(main())

