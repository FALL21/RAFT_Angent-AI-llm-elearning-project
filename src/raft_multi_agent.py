"""
ÉTAPE 8 — RAFT + Multi-Agent IA.
Système multi-agents avec :
  • Agent Chercheur : recherche dans la base RAG et sur le web
  • Agent Analyste : analyse, compare et vérifie les informations
  • Agent Rédacteur : synthétise et rédige la réponse finale
  • Superviseur : orchestre les agents et décide du workflow

Si le LLM n'a pas suffisamment d'informations pertinentes,
l'agent chercheur va automatiquement sur les sites dédiés
pour améliorer la réponse du RAG.
"""
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document, HumanMessage, SystemMessage
from langchain.schema.output_parser import StrOutputParser

import src.config as cfg
from src.config import MULTI_AGENT_CONFIG
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    sender: str
    content: str
    tool_used: Optional[str] = None
    confidence: float = 0.0


# ════════════════════════════════════════════════════════════
# AGENTS SPÉCIALISÉS
# ════════════════════════════════════════════════════════════

class ResearcherAgent:
    """Agent Chercheur — recherche dans RAG et sur le web."""

    SYSTEM_PROMPT = """Tu es un agent chercheur expert. Ta mission :
1. Chercher d'abord dans la base de connaissances interne.
2. Si l'information est insuffisante (< 70% de confiance), chercher sur le web.
3. Retourner les informations trouvées avec les sources et un score de confiance.

Retourne ta réponse au format :
CONFIANCE: [0.0-1.0]
SOURCES: [liste des sources]
CONTENU: [informations trouvées]"""

    def __init__(self, llm, vector_store: VectorStore):
        self.llm = llm
        self.vector_store = vector_store
        self.name = "chercheur"

    def search(self, query: str) -> AgentMessage:
        """Recherche dans la base RAG, puis web si nécessaire."""
        logger.info(f"[Chercheur] Recherche : {query[:60]}...")

        # 1. Recherche RAG
        rag_docs = self.vector_store.similarity_search(query, k=cfg.RAG_TOP_K)
        rag_context = "\n".join(d.page_content[:300] for d in rag_docs)

        # 2. Évaluer la pertinence
        eval_prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", f"Question : {query}\n\nDocuments RAG trouvés :\n{rag_context}\n\n"
                      f"Évalue la pertinence et complète si nécessaire."),
        ])

        chain = eval_prompt | self.llm | StrOutputParser()
        result = chain.invoke({})

        # 3. Si confiance faible → recherche web
        confidence = self._extract_confidence(result)
        if confidence < 0.7:
            logger.info(f"[Chercheur] Confiance faible ({confidence:.1%}), recherche web...")
            web_result = self._web_search(query)
            result += f"\n\n[COMPLÉMENT WEB]\n{web_result}"
            confidence = min(confidence + 0.2, 1.0)

        return AgentMessage(
            sender=self.name,
            content=result,
            tool_used="rag+web" if confidence < 0.9 else "rag",
            confidence=confidence,
        )

    def _web_search(self, query: str) -> str:
        """Recherche web de secours."""
        try:
            from langchain_community.tools import DuckDuckGoSearchRun
            search = DuckDuckGoSearchRun()
            return search.run(query)
        except Exception as e:
            return f"Recherche web indisponible : {e}"

    def _extract_confidence(self, text: str) -> float:
        """Extrait le score de confiance de la réponse."""
        import re
        match = re.search(r"CONFIANCE:\s*([0-9.]+)", text)
        return float(match.group(1)) if match else 0.5


class AnalystAgent:
    """Agent Analyste — analyse, vérifie et compare les informations."""

    SYSTEM_PROMPT = """Tu es un agent analyste rigoureux. Ta mission :
1. Vérifier la cohérence des informations du chercheur.
2. Identifier les contradictions ou lacunes.
3. Structurer les données pour le rédacteur.
4. Attribuer un score de fiabilité.

Format de réponse :
FIABILITÉ: [0.0-1.0]
POINTS_CLÉS: [liste des points importants]
LACUNES: [informations manquantes]
ANALYSE: [ton analyse détaillée]"""

    def __init__(self, llm):
        self.llm = llm
        self.name = "analyste"

    def analyze(self, question: str, research: AgentMessage) -> AgentMessage:
        """Analyse les résultats de la recherche."""
        logger.info(f"[Analyste] Analyse des résultats du chercheur...")

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", f"Question originale : {question}\n\n"
                      f"Résultats de recherche (confiance: {research.confidence:.1%}) :\n"
                      f"{research.content}\n\nAnalyse ces informations."),
        ])

        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({})

        return AgentMessage(
            sender=self.name,
            content=result,
            confidence=research.confidence,
        )


class WriterAgent:
    """Agent Rédacteur — synthétise et rédige la réponse finale."""

    SYSTEM_PROMPT = """Tu es un agent rédacteur expert. Ta mission :
1. Synthétiser les analyses en une réponse claire et complète.
2. Structurer la réponse avec des sections si nécessaire.
3. Citer les sources.
4. Adapter le ton au domaine (académique, professionnel, etc.).

Rédige une réponse qui :
- Est précise et factuelle
- Cite les sources pertinentes
- Est bien structurée
- Ne contient pas de spéculation"""

    def __init__(self, llm):
        self.llm = llm
        self.name = "rédacteur"

    def write(self, question: str, analysis: AgentMessage) -> AgentMessage:
        """Rédige la réponse finale."""
        logger.info(f"[Rédacteur] Rédaction de la réponse finale...")

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", f"Question : {question}\n\n"
                      f"Analyse de l'expert :\n{analysis.content}\n\n"
                      f"Rédige la réponse finale."),
        ])

        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({})

        return AgentMessage(
            sender=self.name,
            content=result,
            confidence=analysis.confidence,
        )


# ════════════════════════════════════════════════════════════
# SUPERVISEUR / ORCHESTRATEUR
# ════════════════════════════════════════════════════════════

class MultiAgentSupervisor:
    """
    Superviseur qui orchestre le workflow multi-agents.
    Pipeline : Question → Chercheur → Analyste → Rédacteur → Réponse.
    
    Si la confiance est trop basse, le superviseur peut :
    - Demander au chercheur de faire une nouvelle recherche
    - Demander à l'analyste de vérifier des points spécifiques
    """

    def __init__(
        self,
        llm,
        vector_store: VectorStore,
        max_retries: int = 2,
        min_confidence: float = 0.5,
    ):
        self.llm = llm
        self.max_retries = max_retries
        self.min_confidence = min_confidence

        # Initialiser les agents
        self.researcher = ResearcherAgent(llm, vector_store)
        self.analyst = AnalystAgent(llm)
        self.writer = WriterAgent(llm)

        self.conversation_log: List[AgentMessage] = []

    def process(self, question: str) -> Dict:
        """
        Workflow principal du système multi-agents.
        """
        logger.info("=" * 60)
        logger.info(f"MULTI-AGENT — Question : {question[:80]}...")
        logger.info("=" * 60)

        self.conversation_log = []
        retries = 0

        # ── Boucle de recherche ─────────────────────────────
        while retries <= self.max_retries:
            # 1. Agent Chercheur
            research = self.researcher.search(question)
            self.conversation_log.append(research)
            logger.info(f"  [Chercheur] Confiance = {research.confidence:.1%}")

            # 2. Agent Analyste
            analysis = self.analyst.analyze(question, research)
            self.conversation_log.append(analysis)

            # 3. Vérifier la confiance
            if research.confidence >= self.min_confidence:
                break

            retries += 1
            logger.info(f"  ⚠ Confiance insuffisante, retry {retries}/{self.max_retries}")
            question_refined = f"{question} (recherche approfondie, tentative {retries+1})"

        # 4. Agent Rédacteur
        final = self.writer.write(question, analysis)
        self.conversation_log.append(final)

        return {
            "question": question,
            "answer": final.content,
            "confidence": research.confidence,
            "agents_used": [msg.sender for msg in self.conversation_log],
            "num_steps": len(self.conversation_log),
            "retries": retries,
            "conversation_log": [
                {
                    "agent": msg.sender,
                    "content": msg.content[:200],
                    "tool": msg.tool_used,
                    "confidence": msg.confidence,
                }
                for msg in self.conversation_log
            ],
            "pipeline": "raft_multi_agent",
        }


# ════════════════════════════════════════════════════════════
# FACTORY
# ════════════════════════════════════════════════════════════

def create_multi_agent_system(llm, vector_store: VectorStore) -> MultiAgentSupervisor:
    """Crée le système multi-agents complet."""
    return MultiAgentSupervisor(llm=llm, vector_store=vector_store)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    print("Module Multi-Agent prêt.")
    print("Agents : Chercheur, Analyste, Rédacteur")
    print("Orchestrateur : Superviseur")
