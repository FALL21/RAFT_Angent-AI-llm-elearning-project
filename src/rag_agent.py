"""
ÉTAPE 7 — RAG + Agent IA.
L'agent peut :
  • Rechercher dans la base vectorielle (RAG)
  • Chercher sur le web pour compléter les informations
  • Lire des PDFs supplémentaires
  • Effectuer des calculs
  • Exécuter du code Python

Architecture ReAct (Reasoning + Acting).
"""
import logging
from typing import List, Dict, Optional, Any

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import Tool, StructuredTool
from langchain.schema import Document
from langchain_community.tools import DuckDuckGoSearchRun

import src.config as cfg
from src.config import AGENT_CONFIG
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)

# ── Prompt de l'Agent ───────────────────────────────────────
AGENT_PROMPT = PromptTemplate.from_template(
    """Tu es un assistant expert avec accès à des outils pour répondre aux questions.

Outils disponibles :
{tools}

Format de réponse :
Question: la question de l'utilisateur
Thought: réflexion sur ce qu'il faut faire
Action: le nom de l'outil à utiliser
Action Input: l'entrée de l'outil
Observation: le résultat de l'outil
... (ce cycle peut se répéter)
Thought: j'ai maintenant la réponse finale
Final Answer: la réponse complète et détaillée

RÈGLES :
1. Commence TOUJOURS par chercher dans la base de connaissances.
2. Si la base ne contient pas assez d'info, utilise la recherche web.
3. Synthétise toutes les sources dans ta réponse finale.
4. Cite tes sources.

Noms des outils : {tool_names}

Question: {input}
{agent_scratchpad}"""
)


class RAGAgentTools:
    """Outils de l'agent RAG."""

    def __init__(self, vector_store: VectorStore, llm=None):
        self.vector_store = vector_store
        self.llm = llm

    def rag_search(self, query: str) -> str:
        """Recherche dans la base de connaissances vectorielle."""
        try:
            docs = self.vector_store.similarity_search(query, k=cfg.RAG_TOP_K)
            if not docs:
                return "Aucun document pertinent trouvé dans la base de connaissances."

            results = []
            for i, doc in enumerate(docs):
                source = doc.metadata.get("source_file", "N/A")
                results.append(f"[{i+1}] Source: {source}\n{doc.page_content[:400]}")
            return "\n\n".join(results)
        except Exception as e:
            return f"Erreur de recherche : {e}"

    def web_search(self, query: str) -> str:
        """Recherche sur le web via DuckDuckGo."""
        try:
            search = DuckDuckGoSearchRun()
            return search.run(query)
        except Exception as e:
            return f"Erreur de recherche web : {e}"

    def python_executor(self, code: str) -> str:
        """Exécute du code Python simple (calculs, transformations)."""
        try:
            import io
            import contextlib
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exec(code, {"__builtins__": __builtins__})
            return output.getvalue() or "Code exécuté avec succès (pas de sortie)."
        except Exception as e:
            return f"Erreur d'exécution : {e}"

    def get_tools(self) -> List[Tool]:
        """Retourne la liste des outils de l'agent."""
        tools = [
            Tool(
                name="base_connaissances",
                func=self.rag_search,
                description="Recherche dans la base de connaissances interne (PDFs indexés). "
                            "Utiliser en PREMIER pour toute question.",
            ),
            Tool(
                name="recherche_web",
                func=self.web_search,
                description="Recherche sur Internet pour trouver des informations complémentaires "
                            "quand la base de connaissances ne suffit pas.",
            ),
            Tool(
                name="calculateur_python",
                func=self.python_executor,
                description="Exécute du code Python pour des calculs ou transformations de données.",
            ),
        ]
        return tools


class RAGAgent:
    """
    Agent IA combinant RAG et outils externes.
    Architecture ReAct : Raisonne → Agit → Observe → Répète.
    """

    def __init__(
        self,
        llm,
        vector_store: VectorStore,
        max_iterations: int = AGENT_CONFIG["max_iterations"],
    ):
        self.llm = llm
        self.vector_store = vector_store
        self.max_iterations = max_iterations

        # Initialiser les outils
        self.tools_provider = RAGAgentTools(vector_store, llm)
        self.tools = self.tools_provider.get_tools()

        # Créer l'agent ReAct
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=AGENT_PROMPT,
        )

        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            max_iterations=self.max_iterations,
            verbose=True,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )

    def query(self, question: str) -> Dict:
        """Pose une question à l'agent."""
        logger.info(f"\nAgent RAG — Question : {question[:80]}...")

        try:
            result = self.executor.invoke({"input": question})

            # Extraire les étapes intermédiaires
            steps = []
            for action, observation in result.get("intermediate_steps", []):
                steps.append({
                    "tool": action.tool,
                    "input": action.tool_input,
                    "observation": str(observation)[:300],
                })

            return {
                "question": question,
                "answer": result["output"],
                "steps": steps,
                "num_steps": len(steps),
                "tools_used": list(set(s["tool"] for s in steps)),
                "pipeline": "rag_agent",
            }

        except Exception as e:
            logger.error(f"Erreur agent : {e}")
            return {
                "question": question,
                "answer": f"Erreur lors du traitement : {e}",
                "steps": [],
                "pipeline": "rag_agent",
            }

    def query_with_context(self, question: str, context: str = "") -> Dict:
        """Pose une question avec un contexte additionnel."""
        enriched_q = f"{question}\n\nContexte additionnel : {context}" if context else question
        return self.query(enriched_q)
