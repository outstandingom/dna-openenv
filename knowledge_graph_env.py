import os
import json
import asyncio
import threading
import time
import pickle
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set, Any
import numpy as np
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai

# ============================================================
# Configuration
# ============================================================
DIMS = 16
ALPHABET = [chr(ord('A') + i) for i in range(26)]
POSITION_OFFSET = 0.1
LR_LETTER = 0.005
LR_FEATURE_VEC = 0.01
LR_CONCEPT = 0.01
GRAD_CLIP = 1.0
MAX_CONCEPTS = 10000
BATCH_SIZE = 32
TRAIN_INTERVAL_SEC = 10
PERSIST_DIR = "./brain_data"

# LLM configuration from environment
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN = os.getenv("HF_TOKEN", "")
openai_client = openai.OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN) if HF_TOKEN else None

STOP_WORDS = {"the","and","for","are","but","not","you","all","can","had","her","was","one","our","out","has","have","from","they","been","said","each","which","their","will","other","about","many","then","them","these","some","would","make","like","into","time","very","when","come","could","than","its","also","back","after","two","how","what","where","who","why","this","that","with"}

# ============================================================
# Auto‑expanding Ontology with OpenAI
# ============================================================
class DynamicOntology:
    def __init__(self):
        self.concept_to_features: Dict[str, List[str]] = {}
        self.feature_to_concepts: Dict[str, List[str]] = defaultdict(list)
        self.llm_enabled = openai_client is not None

    async def get_features_llm(self, concept: str) -> List[str]:
        if not self.llm_enabled:
            return [concept]
        try:
            response = openai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts physical and semantic features of concepts. Return a comma-separated list of up to 5 features."},
                    {"role": "user", "content": f"List the most important physical and semantic features of '{concept}'. Return only the list."}
                ],
                temperature=0.3,
                max_tokens=100
            )
            text = response.choices[0].message.content
            features = [f.strip().lower() for f in text.split(",")]
            return features[:5]
        except Exception:
            return [concept]

    async def add_concept(self, concept: str, context: str = ""):
        concept_low = concept.lower()
        if concept_low in self.concept_to_features:
            return
        features = await self.get_features_llm(f"{context} {concept}" if context else concept)
        self.concept_to_features[concept_low] = features
        for f in features:
            self.feature_to_concepts[f].append(concept_low)

    def get_features(self, concept: str, context: str = "") -> List[str]:
        concept_low = concept.lower()
        if concept_low not in self.concept_to_features:
            return [concept_low]
        return self.concept_to_features[concept_low]

    def serialize(self) -> dict:
        return self.concept_to_features

    def restore(self, data: dict):
        self.concept_to_features = data
        self.feature_to_concepts = defaultdict(list)
        for concept, feats in data.items():
            for f in feats:
                self.feature_to_concepts[f].append(concept)

# ============================================================
# Feature Registry (unchanged)
# ============================================================
class FeatureRegistry:
    def __init__(self, ontology: DynamicOntology):
        self.ontology = ontology
        self.feature_to_id: Dict[str, int] = {}
        self.id_to_feature: Dict[int, str] = {}
        self.feature_vectors: Dict[int, np.ndarray] = {}
        self.next_id = 0
        all_features = set()
        for feats in ontology.concept_to_features.values():
            all_features.update(feats)
        for feat in all_features:
            self.register(feat)

    def register(self, feature_name: str) -> int:
        name = feature_name.lower()
        if name in self.feature_to_id:
            return self.feature_to_id[name]
        fid = self.next_id
        self.next_id += 1
        self.feature_to_id[name] = fid
        self.id_to_feature[fid] = name
        self.feature_vectors[fid] = np.random.uniform(-1, 1, DIMS).astype(np.float32)
        return fid

    def get_vector(self, fid: int) -> np.ndarray:
        return self.feature_vectors[fid]

    def update_vector(self, fid: int, delta: np.ndarray):
        delta = np.clip(delta, -GRAD_CLIP, GRAD_CLIP)
        self.feature_vectors[fid] += delta

    def feature_to_letters(self, fid: int, length: int = 5) -> List[str]:
        vec = self.feature_vectors[fid]
        probs = np.exp(vec[:length])
        probs /= np.sum(probs)
        idx = np.argmax(probs)
        letter_idx = idx % 26
        return [ALPHABET[letter_idx]] * length

    def serialize(self) -> dict:
        return {
            "feature_to_id": self.feature_to_id,
            "id_to_feature": {str(k): v for k, v in self.id_to_feature.items()},
            "feature_vectors": {str(k): v.tolist() for k, v in self.feature_vectors.items()},
            "next_id": self.next_id,
            "ontology": self.ontology.serialize()
        }

    def restore(self, data: dict):
        self.feature_to_id = data["feature_to_id"]
        self.id_to_feature = {int(k): v for k, v in data["id_to_feature"].items()}
        self.feature_vectors = {int(k): np.array(v, dtype=np.float32) for k, v in data["feature_vectors"].items()}
        self.next_id = data["next_id"]
        self.ontology.restore(data["ontology"])

# ============================================================
# Letter Vectors
# ============================================================
class LetterVectors:
    def __init__(self):
        self.vec = {ch: np.random.uniform(-1, 1, DIMS).astype(np.float32) for ch in ALPHABET}
    def get(self, letter: str) -> np.ndarray: return self.vec[letter]
    def update(self, letter: str, delta: np.ndarray):
        delta = np.clip(delta, -GRAD_CLIP, GRAD_CLIP)
        self.vec[letter] += delta
    def serialize(self) -> dict: return {ch: self.vec[ch].tolist() for ch in ALPHABET}
    def restore(self, data: dict):
        for ch, arr in data.items():
            self.vec[ch] = np.array(arr, dtype=np.float32)

# ============================================================
# DNA Concept (simplified)
# ============================================================
class DNAConcept:
    def __init__(self, name: str, physical_features: List[int], semantic_features: List[int]):
        self.name = name
        self.physical_features = physical_features
        self.semantic_features = semantic_features
        self.vector: Optional[np.ndarray] = None
        self._update_vector()

    def _encode_feature(self, fid: int, start_pos: int) -> np.ndarray:
        letters = FEATURE_REGISTRY.feature_to_letters(fid, length=5)
        vec = np.zeros(DIMS, dtype=np.float32)
        for i, ch in enumerate(letters):
            base = LETTER_VEC.get(ch)
            vec += np.sin(base + (start_pos + i) * POSITION_OFFSET)
        return vec

    def _update_vector(self):
        vec = np.zeros(DIMS, dtype=np.float32)
        pos = 0
        for fid in self.physical_features:
            vec += self._encode_feature(fid, pos)
            pos += 5
        for fid in self.semantic_features:
            vec += self._encode_feature(fid, pos)
            pos += 5
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        self.vector = vec

    def move_towards(self, other: 'DNAConcept', lr: float = LR_CONCEPT):
        diff = other.vector - self.vector
        self.vector += lr * diff
        other.vector -= lr * diff
        self.vector /= (np.linalg.norm(self.vector) + 1e-8)
        other.vector /= (np.linalg.norm(other.vector) + 1e-8)

    def cosine_similarity(self, other: 'DNAConcept') -> float:
        return np.dot(self.vector, other.vector) / (np.linalg.norm(self.vector) * np.linalg.norm(other.vector) + 1e-8)

    def serialize(self) -> dict:
        return {
            "name": self.name,
            "physical_features": self.physical_features,
            "semantic_features": self.semantic_features,
            "vector": self.vector.tolist()
        }
    @classmethod
    def from_serialized(cls, data: dict):
        obj = cls(data["name"], data["physical_features"], data["semantic_features"])
        obj.vector = np.array(data["vector"], dtype=np.float32)
        return obj

# ============================================================
# Reasoning Engine
# ============================================================
class ReasoningEngine:
    def __init__(self, concept_memory: 'ConceptMemory', feature_registry: FeatureRegistry):
        self.concept_memory = concept_memory
        self.feature_registry = feature_registry

    def multi_hop_reasoning(self, start: str, max_hops: int = 3, decay: float = 0.7) -> Dict[str, float]:
        if start not in self.concept_memory.relationships:
            return {}
        activation = {start: 1.0}
        for hop in range(max_hops):
            new_activation = {}
            for node, score in activation.items():
                neighbors = self.concept_memory.relationships.get(node, set())
                for nb in neighbors:
                    weight = 1.0
                    if node in self.concept_memory.concepts and nb in self.concept_memory.concepts:
                        weight = self.concept_memory.concepts[node].cosine_similarity(self.concept_memory.concepts[nb])
                    new_activation[nb] = new_activation.get(nb, 0) + score * weight * decay
            for k, v in new_activation.items():
                activation[k] = activation.get(k, 0) + v
        if activation:
            max_act = max(activation.values())
            activation = {k: v/max_act for k, v in activation.items()}
        return activation

    def analogical_reasoning(self, a: str, b: str, c: str) -> List[str]:
        if a not in self.concept_memory.concepts or b not in self.concept_memory.concepts:
            return []
        vec_a = self.concept_memory.concepts[a].vector
        vec_b = self.concept_memory.concepts[b].vector
        direction = vec_b - vec_a
        if c not in self.concept_memory.concepts:
            return []
        vec_c = self.concept_memory.concepts[c].vector
        target = vec_c + direction
        target /= (np.linalg.norm(target) + 1e-8)
        results = self.concept_memory.search(target, top_k=5)
        return [r[0] for r in results if r[0] not in (a, b, c)]

# ============================================================
# Concept Memory
# ============================================================
class ConceptMemory:
    def __init__(self, max_concepts=MAX_CONCEPTS):
        self.concepts: Dict[str, DNAConcept] = {}
        self.relationships: Dict[str, Set[str]] = defaultdict(set)
        self.max_concepts = max_concepts
        self.index = faiss.IndexFlatIP(DIMS)
        self.id_to_name: Dict[int, str] = {}
        self.name_to_id: Dict[str, int] = {}
        self.next_id = 0

    def _rebuild_index(self):
        if not self.concepts:
            return
        vectors = [c.vector for c in self.concepts.values()]
        names = list(self.concepts.keys())
        vecs = np.vstack(vectors).astype(np.float32)
        self.index = faiss.IndexFlatIP(DIMS)
        self.index.add(vecs)
        self.id_to_name = {i: name for i, name in enumerate(names)}
        self.name_to_id = {name: i for i, name in enumerate(names)}
        self.next_id = len(self.concepts)

    def register(self, name: str, physical_features: List[int], semantic_features: List[int]) -> DNAConcept:
        name_low = name.lower()
        if name_low in self.concepts:
            return self.concepts[name_low]
        concept = DNAConcept(name_low, physical_features, semantic_features)
        self.concepts[name_low] = concept
        self.index.add(concept.vector.reshape(1, -1))
        self.id_to_name[self.next_id] = name_low
        self.name_to_id[name_low] = self.next_id
        self.next_id += 1
        self._prune()
        return concept

    def add_relationship(self, a: str, b: str):
        a_low, b_low = a.lower(), b.lower()
        if a_low in self.concepts and b_low in self.concepts:
            self.relationships[a_low].add(b_low)
            self.relationships[b_low].add(a_low)
            self.concepts[a_low].move_towards(self.concepts[b_low])

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        if self.index.ntotal == 0:
            return []
        q = query_vector.reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(q, top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:
                name = self.id_to_name[idx]
                sim = float(distances[0][i])
                results.append((name, sim))
        return results

    async def extract_and_link(self, text: str, sector: str = "general") -> List[str]:
        words = [w for w in text.lower().split() if len(w) > 3 and w not in STOP_WORDS]
        unique = list(set(words))[:15]
        concept_list = []
        for kw in unique:
            features = await ONTOLOGY.get_features_llm(kw) if ONTOLOGY.llm_enabled else ONTOLOGY.get_features(kw)
            physical_fids = [FEATURE_REGISTRY.register(f) for f in features]
            semantic_fids = [FEATURE_REGISTRY.register(f) for f in features]
            concept = self.register(kw, physical_fids, semantic_fids)
            concept_list.append(kw)
        for i in range(len(unique)):
            for j in range(i+1, min(i+4, len(unique))):
                self.add_relationship(unique[i], unique[j])
        return concept_list

    def _prune(self):
        if len(self.concepts) > self.max_concepts:
            sorted_concepts = sorted(self.concepts.items(), key=lambda x: len(self.relationships.get(x[0], [])))
            to_remove = [name for name, _ in sorted_concepts[:len(self.concepts)-self.max_concepts]]
            for name in to_remove:
                del self.concepts[name]
                if name in self.relationships:
                    del self.relationships[name]
            self._rebuild_index()

    def serialize(self) -> dict:
        return {
            "concepts": {name: c.serialize() for name, c in self.concepts.items()},
            "relationships": {k: list(v) for k, v in self.relationships.items()}
        }
    def restore(self, data: dict):
        self.concepts = {}
        self.relationships = defaultdict(set)
        for name, cdata in data.get("concepts", {}).items():
            self.concepts[name] = DNAConcept.from_serialized(cdata)
        for k, vlist in data.get("relationships", {}).items():
            self.relationships[k] = set(vlist)
        self._rebuild_index()

# ============================================================
# Persistence
# ============================================================
class PersistenceManager:
    @staticmethod
    def save_all(concept_memory: ConceptMemory, feature_registry: FeatureRegistry, letter_vec: LetterVectors):
        os.makedirs(PERSIST_DIR, exist_ok=True)
        with open(os.path.join(PERSIST_DIR, "concept_memory.pkl"), "wb") as f:
            pickle.dump(concept_memory.serialize(), f)
        with open(os.path.join(PERSIST_DIR, "feature_registry.pkl"), "wb") as f:
            pickle.dump(feature_registry.serialize(), f)
        with open(os.path.join(PERSIST_DIR, "letter_vec.pkl"), "wb") as f:
            pickle.dump(letter_vec.serialize(), f)
        if concept_memory.index.ntotal > 0:
            faiss.write_index(concept_memory.index, os.path.join(PERSIST_DIR, "faiss.index"))
    @staticmethod
    def load_all() -> Tuple[ConceptMemory, FeatureRegistry, LetterVectors]:
        ontology = DynamicOntology()
        feature_registry = FeatureRegistry(ontology)
        letter_vec = LetterVectors()
        concept_memory = ConceptMemory()
        if not os.path.exists(PERSIST_DIR):
            return concept_memory, feature_registry, letter_vec
        concept_file = os.path.join(PERSIST_DIR, "concept_memory.pkl")
        if os.path.exists(concept_file):
            with open(concept_file, "rb") as f:
                concept_memory.restore(pickle.load(f))
        feat_file = os.path.join(PERSIST_DIR, "feature_registry.pkl")
        if os.path.exists(feat_file):
            with open(feat_file, "rb") as f:
                feature_registry.restore(pickle.load(f))
        letter_file = os.path.join(PERSIST_DIR, "letter_vec.pkl")
        if os.path.exists(letter_file):
            with open(letter_file, "rb") as f:
                letter_vec.restore(pickle.load(f))
        index_file = os.path.join(PERSIST_DIR, "faiss.index")
        if os.path.exists(index_file):
            concept_memory.index = faiss.read_index(index_file)
        else:
            concept_memory._rebuild_index()
        return concept_memory, feature_registry, letter_vec

# ============================================================
# Background Trainer (optional)
# ============================================================
class ContinuousTrainer:
    def __init__(self, concept_memory: ConceptMemory, interval_sec: int = TRAIN_INTERVAL_SEC):
        self.concept_memory = concept_memory
        self.interval = interval_sec
        self.running = False
        self.thread = None
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._train_loop, daemon=True)
        self.thread.start()
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
    def _train_loop(self):
        while self.running:
            time.sleep(self.interval)
            rels = [(a, b) for a, s in self.concept_memory.relationships.items() for b in s]
            if not rels:
                continue
            batch = random.sample(rels, min(BATCH_SIZE, len(rels)))
            for a, b in batch:
                if a in self.concept_memory.concepts and b in self.concept_memory.concepts:
                    self.concept_memory.concepts[a].move_towards(self.concept_memory.concepts[b])
            self.concept_memory._rebuild_index()
            if random.random() < 0.1:
                PersistenceManager.save_all(self.concept_memory, FEATURE_REGISTRY, LETTER_VEC)

# ============================================================
# Upgraded Environment (OpenEnv compliant)
# ============================================================
class KnowledgeGraphEnv:
    def __init__(self):
        global LETTER_VEC, FEATURE_REGISTRY, CONCEPT_MEMORY, REASONING_ENGINE, ONTOLOGY, TRAINER
        self.concept_memory, self.feature_registry, self.letter_vec = PersistenceManager.load_all()
        global LETTER_VEC, FEATURE_REGISTRY, CONCEPT_MEMORY, ONTOLOGY, REASONING_ENGINE
        LETTER_VEC = self.letter_vec
        FEATURE_REGISTRY = self.feature_registry
        CONCEPT_MEMORY = self.concept_memory
        ONTOLOGY = self.feature_registry.ontology
        REASONING_ENGINE = ReasoningEngine(CONCEPT_MEMORY, FEATURE_REGISTRY)
        self.trainer = ContinuousTrainer(CONCEPT_MEMORY)
        self.trainer.start()
        self._seed_initial_concepts()
        self.current_task = None
        self.current_step = 0
        self.episode_reward = 0.0
        self.done = False

    def _seed_initial_concepts(self):
        if len(self.concept_memory.concepts) > 0:
            return
        seed_data = [
            ("login issue", ["authentication", "password", "access"]),
            ("billing", ["payment", "invoice", "refund"]),
            ("slow performance", ["latency", "response time", "optimization"]),
            ("crash", ["bug", "error", "unstable"]),
            ("feature request", ["enhancement", "new functionality", "suggestion"]),
            ("account locked", ["security", "blocked", "verification"]),
        ]
        for concept, features in seed_data:
            physical_fids = [self.feature_registry.register(f) for f in features]
            semantic_fids = [self.feature_registry.register(f) for f in features]
            self.concept_memory.register(concept, physical_fids, semantic_fids)
        self.concept_memory.add_relationship("login issue", "account locked")
        self.concept_memory.add_relationship("billing", "refund")
        self.concept_memory.add_relationship("slow performance", "crash")
        self.concept_memory.add_relationship("feature request", "enhancement")

    def reset(self) -> str:
        self.current_task = self._generate_dynamic_task()
        self.current_step = 0
        self.episode_reward = 0.0
        self.done = False
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.concept_memory.extract_and_link(self.current_task["input"]))
            else:
                loop.run_until_complete(self.concept_memory.extract_and_link(self.current_task["input"]))
        except:
            pass
        return self.current_task["input"]

    def step(self, action: str) -> Tuple[str, float, bool, dict]:
        if self.done:
            raise RuntimeError("Episode already done. Call reset() first.")
        step_type = ["identify", "relate", "answer"][self.current_step]
        reward = 0.0
        if step_type == "identify":
            reward = self._grade_identification(action, self.current_task["expected_concept"])
            if reward >= 0.3:
                self.current_step = 1
        elif step_type == "relate":
            reward = self._grade_relation(action, self.current_task["expected_relation"])
            if reward >= 0.3:
                self.current_step = 2
        else:
            reward = self._grade_answer(action, self.current_task["expected_answer"])
            self.done = True
        self.episode_reward += reward
        obs = self.current_task["input"] if not self.done else ""
        info = {
            "task_type": self.current_task["type"],
            "step": step_type,
            "step_reward": reward,
            "total_reward": self.episode_reward
        }
        return obs, reward, self.done, info

    def state(self) -> dict:
        if not self.current_task:
            return {}
        return {
            "task": self.current_task,
            "step": self.current_step,
            "step_name": ["identify", "relate", "answer"][self.current_step] if self.current_step < 3 else "done"
        }

    def _generate_dynamic_task(self) -> dict:
        concepts = list(self.concept_memory.concepts.keys())
        if not concepts:
            return {
                "type": "easy",
                "input": "My phone won't turn on",
                "expected_concept": "hardware failure",
                "expected_relation": "battery issue",
                "expected_answer": "replace battery or check power"
            }
        base_concept = random.choice(concepts)
        related = list(self.concept_memory.relationships.get(base_concept, set()))
        if not related:
            vec = self.concept_memory.concepts[base_concept].vector
            similar = self.concept_memory.search(vec, top_k=3)
            related = [r[0] for r in similar if r[0] != base_concept]
        expected_relation = related[0] if related else "related issue"
        reasoning_result = REASONING_ENGINE.multi_hop_reasoning(base_concept, max_hops=2)
        possible_answers = [c for c in reasoning_result.keys() if c != base_concept and c != expected_relation]
        expected_answer = possible_answers[0] if possible_answers else "contact support"
        templates = [
            f"I'm having trouble with {base_concept}. Can you help?",
            f"User reports {base_concept} persists.",
            f"Ticket: {base_concept} affecting workflow.",
            f"Customer says {base_concept} needs urgent resolution.",
        ]
        input_text = random.choice(templates)
        return {
            "type": "dynamic",
            "input": input_text,
            "expected_concept": base_concept,
            "expected_relation": expected_relation,
            "expected_answer": expected_answer
        }

    def _grade_identification(self, action: str, expected_concept: str) -> float:
        action_lower = action.lower().strip()
        if action_lower in self.concept_memory.concepts:
            pred_vec = self.concept_memory.concepts[action_lower].vector
        else:
            features = ONTOLOGY.get_features(action_lower)
            temp_vec = np.zeros(DIMS)
            for i, f in enumerate(features[:5]):
                if f in self.feature_registry.feature_to_id:
                    fid = self.feature_registry.feature_to_id[f]
                    fv = self.feature_registry.get_vector(fid)
                    temp_vec += fv
            if np.linalg.norm(temp_vec) > 0:
                temp_vec /= np.linalg.norm(temp_vec)
            else:
                temp_vec = np.random.randn(DIMS)
                temp_vec /= np.linalg.norm(temp_vec)
            pred_vec = temp_vec
        expected_vec = self.concept_memory.concepts[expected_concept].vector
        similarity = float(np.dot(pred_vec, expected_vec) / (np.linalg.norm(pred_vec) * np.linalg.norm(expected_vec) + 1e-8))
        reward = max(0.0, min(1.0, (similarity + 1) / 2))
        if action_lower == expected_concept:
            reward = 1.0
        return reward

    def _grade_relation(self, action: str, expected_relation: str) -> float:
        action_lower = action.lower().strip()
        reward = 0.0
        base_concept = self.current_task["expected_concept"]
        if action_lower in self.concept_memory.relationships.get(base_concept, set()):
            reward += 0.5
        if expected_relation in action_lower:
            reward += 0.5
        acts = REASONING_ENGINE.multi_hop_reasoning(base_concept, max_hops=1)
        if action_lower in acts:
            reward += 0.3
        return min(1.0, reward)

    def _grade_answer(self, action: str, expected_answer: str) -> float:
        action_lower = action.lower().strip()
        if action_lower in self.concept_memory.concepts:
            pred_vec = self.concept_memory.concepts[action_lower].vector
        else:
            features = ONTOLOGY.get_features(action_lower)
            temp_vec = np.zeros(DIMS)
            for i, f in enumerate(features[:5]):
                if f in self.feature_registry.feature_to_id:
                    fid = self.feature_registry.feature_to_id[f]
                    fv = self.feature_registry.get_vector(fid)
                    temp_vec += fv
            if np.linalg.norm(temp_vec) > 0:
                temp_vec /= np.linalg.norm(temp_vec)
            else:
                temp_vec = np.random.randn(DIMS)
                temp_vec /= np.linalg.norm(temp_vec)
            pred_vec = temp_vec
        results = self.concept_memory.search(pred_vec, top_k=1)
        if results:
            best_match, sim = results[0]
            expected_vec = self.concept_memory.concepts[expected_answer].vector
            similarity = float(np.dot(pred_vec, expected_vec) / (np.linalg.norm(pred_vec) * np.linalg.norm(expected_vec) + 1e-8))
            reward = max(0.0, min(1.0, (similarity + 1) / 2))
            if best_match == expected_answer:
                reward = 1.0
        else:
            reward = 0.0
        return reward

    def close(self):
        self.trainer.stop()
        PersistenceManager.save_all(self.concept_memory, self.feature_registry, self.letter_vec)

# ============================================================
# FastAPI Server for HF Space
# ============================================================
app = FastAPI()
env = KnowledgeGraphEnv()

class ResetResponse(BaseModel):
    observation: str

class StepRequest(BaseModel):
    action: str

class StepResponse(BaseModel):
    observation: str
    reward: float
    done: bool
    info: dict

class StateResponse(BaseModel):
    state: dict

@app.get("/ping")
async def ping():
    return {"status": "ok"}

@app.post("/reset", response_model=ResetResponse)
async def reset_endpoint():
    obs = env.reset()
    return ResetResponse(observation=obs)

@app.post("/step", response_model=StepResponse)
async def step_endpoint(req: StepRequest):
    obs, reward, done, info = env.step(req.action)
    return StepResponse(observation=obs, reward=reward, done=done, info=info)

@app.get("/state", response_model=StateResponse)
async def state_endpoint():
    return StateResponse(state=env.state())

@app.on_event("shutdown")
def shutdown_event():
    env.close()

# Global references for other modules
LETTER_VEC = None
FEATURE_REGISTRY = None
CONCEPT_MEMORY = None
REASONING_ENGINE = None
ONTOLOGY = None
TRAINER = None