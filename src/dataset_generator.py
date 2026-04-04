"""
ÉTAPE 1 — Génération du dataset d'évaluation (300 paires Question-Réponse).
Domaine unique : E-LEARNING (IA en éducation, apprentissage adaptatif, LLMs, NLP éducatif).

Sous-domaines couverts :
  • IA et apprentissage adaptatif
  • LLMs et tuteurs intelligents
  • NLP pour l'éducation
  • MOOC et plateformes en ligne
  • Gamification et engagement
  • Évaluation automatique
  • Learning Analytics
  • Design pédagogique
  • Technologies éducatives émergentes
  • Éthique et IA en éducation
"""
import json
import logging
import random
import re
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

from src.config import EVALUATION_DIR, EVALUATION_CONFIG

logger = logging.getLogger(__name__)

QA_FROM_TEXT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            """Tu es un expert en pédagogie et NLP. À partir du passage suivant (extrait de documents PDF),
rédige UNE question d'évaluation et SA réponse. La réponse doit être entièrement fidèle au texte (aucune invention).

Exigences :
- Question claire, de niveau cours / révision.
- Réponse : 2 à 5 phrases, en t'appuyant uniquement sur le passage.

Réponds UNIQUEMENT avec un JSON valide sur une seule « forme » objet, sans markdown ni texte autour :
{{"question": "...", "answer": "..."}}

Passage :
---
{text}
---""",
        ),
    ]
)


def _parse_json_object(raw: str) -> Dict[str, Any]:
    s = raw.strip()
    if "```" in s:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", s)
        if m:
            s = m.group(1).strip()
    s = s.strip()
    try:
        data = json.loads(s)
        if isinstance(data, dict):
            return data
        raise ValueError("La racine JSON doit être un objet {{question, answer}}")
    except json.JSONDecodeError:
        pass
    start = s.find("{")
    if start == -1:
        raise ValueError("Aucun objet JSON trouvé dans la réponse du modèle")
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(s, start)
    if not isinstance(obj, dict):
        raise ValueError("Le JSON extrait n'est pas un objet")
    return obj


def _is_inference_quota_exhausted(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "402" in msg
        or "payment required" in msg
        or "depleted your monthly" in msg
        or "pre-paid credits" in msg
    )

# ────────────────────────────────────────────────────────────
# 30 Templates de base × 10 variantes = 300 Q/R
# ────────────────────────────────────────────────────────────

ELEARNING_QA_TEMPLATES = [
    # ── IA & Apprentissage adaptatif (1-30) ────────────────
    {"q": "Qu'est-ce que l'apprentissage adaptatif en e-learning ?",
     "a": "L'apprentissage adaptatif utilise des algorithmes d'IA pour personnaliser le parcours pédagogique en temps réel selon le niveau, le rythme et les préférences de chaque apprenant, en ajustant dynamiquement le contenu et la difficulté.",
     "sub_domain": "apprentissage_adaptatif"},

    {"q": "Comment les systèmes de recommandation fonctionnent-ils dans les plateformes e-learning ?",
     "a": "Les systèmes de recommandation en e-learning utilisent le filtrage collaboratif et le filtrage basé sur le contenu pour suggérer des ressources pertinentes. Ils analysent l'historique d'apprentissage, les performances et les préférences de l'apprenant pour proposer du contenu adapté.",
     "sub_domain": "apprentissage_adaptatif"},

    {"q": "Quels sont les principaux algorithmes utilisés pour l'apprentissage adaptatif ?",
     "a": "Les principaux algorithmes incluent les modèles bayésiens (BKT — Bayesian Knowledge Tracing), les réseaux de neurones récurrents (DKT — Deep Knowledge Tracing), les arbres de décision, le Q-learning pour l'optimisation des parcours, et les bandits multi-bras pour l'exploration-exploitation du contenu.",
     "sub_domain": "apprentissage_adaptatif"},

    # ── LLMs et Tuteurs Intelligents (31-60) ────────────────
    {"q": "Comment les LLMs peuvent-ils servir de tuteurs intelligents ?",
     "a": "Les LLMs comme GPT-4 ou Mistral peuvent agir comme tuteurs en générant des explications personnalisées, en posant des questions socratiques, en fournissant du feedback détaillé sur les réponses des apprenants, et en s'adaptant au niveau de compréhension de l'étudiant.",
     "sub_domain": "llm_tuteur"},

    {"q": "Qu'est-ce qu'un ITS (Intelligent Tutoring System) ?",
     "a": "Un ITS est un système informatique qui fournit un enseignement personnalisé sans intervention humaine. Il comprend un modèle du domaine (connaissances), un modèle de l'apprenant (suivi des compétences), un modèle pédagogique (stratégies d'enseignement) et une interface utilisateur.",
     "sub_domain": "llm_tuteur"},

    {"q": "Quels sont les avantages des chatbots éducatifs basés sur les LLMs ?",
     "a": "Les chatbots éducatifs offrent une disponibilité 24h/24, des réponses personnalisées, un feedback instantané, la capacité de s'adapter au rythme de l'apprenant, et une approche socratique pour guider la réflexion plutôt que donner directement les réponses.",
     "sub_domain": "llm_tuteur"},

    {"q": "Comment le RAG (Retrieval-Augmented Generation) améliore-t-il les tuteurs IA ?",
     "a": "Le RAG permet au tuteur IA de consulter une base de connaissances vérifiée (cours, manuels) avant de répondre, réduisant les hallucinations et assurant que les réponses sont alignées avec le programme pédagogique officiel.",
     "sub_domain": "llm_tuteur"},

    # ── NLP pour l'éducation (61-90) ────────────────────────
    {"q": "Comment le NLP est-il utilisé pour l'évaluation automatique des essais ?",
     "a": "Le NLP évalue les essais en analysant la grammaire, la cohérence textuelle, la richesse lexicale, la structure argumentative et la pertinence du contenu. Des modèles comme BERT fine-tuné peuvent attribuer des scores corrélés avec ceux des correcteurs humains.",
     "sub_domain": "nlp_education"},

    {"q": "Qu'est-ce que la génération automatique de questions en NLP éducatif ?",
     "a": "La génération automatique de questions utilise des modèles de langage pour créer des questions à partir d'un texte source. Les techniques incluent la transformation syntaxique, les modèles seq2seq, et les LLMs avec prompting. Cela permet de créer des quiz et examens rapidement.",
     "sub_domain": "nlp_education"},

    {"q": "Comment le traitement du langage naturel détecte-t-il le plagiat ?",
     "a": "Les systèmes anti-plagiat NLP utilisent la comparaison de n-grammes, les embeddings sémantiques pour détecter la paraphrase, l'analyse stylistique (stylométrie) et la comparaison avec des bases de données de documents. Les modèles récents détectent aussi le texte généré par IA.",
     "sub_domain": "nlp_education"},

    {"q": "Expliquez le Knowledge Tracing et son importance en e-learning.",
     "a": "Le Knowledge Tracing modélise l'évolution des connaissances d'un apprenant au fil du temps. Le BKT (Bayesian KT) utilise des modèles de Markov cachés, tandis que le DKT (Deep KT) utilise des LSTM pour prédire la probabilité de réponse correcte à la prochaine question.",
     "sub_domain": "nlp_education"},

    # ── MOOC et Plateformes (91-120) ────────────────────────
    {"q": "Qu'est-ce qu'un MOOC et quels sont ses avantages ?",
     "a": "Un MOOC (Massive Open Online Course) est un cours en ligne ouvert à tous, sans limite de participants. Ses avantages incluent l'accessibilité mondiale, la flexibilité temporelle, la diversité des contenus et le coût réduit. Les plateformes majeures sont Coursera, edX et FutureLearn.",
     "sub_domain": "mooc_plateformes"},

    {"q": "Comment réduire le taux d'abandon dans les formations en ligne ?",
     "a": "Les stratégies incluent la gamification, les rappels automatisés, les communautés d'apprenants, les contenus courts et variés (microlearning), le feedback régulier, les parcours adaptatifs personnalisés, et les interventions prédictives basées sur les learning analytics.",
     "sub_domain": "mooc_plateformes"},

    {"q": "Quelles sont les différences entre apprentissage synchrone et asynchrone ?",
     "a": "L'apprentissage synchrone se déroule en temps réel (visioconférence, chat en direct) avec interaction immédiate. L'asynchrone permet d'apprendre à son rythme (vidéos, forums, exercices). Le blended learning combine les deux pour maximiser les avantages de chaque approche.",
     "sub_domain": "mooc_plateformes"},

    {"q": "Comment les plateformes e-learning utilisent-elles les données des apprenants ?",
     "a": "Les plateformes collectent des données de navigation, temps passé, résultats aux quiz, interactions et patterns d'apprentissage. Ces données alimentent les systèmes de recommandation, le learning analytics, la détection du décrochage et l'amélioration continue du contenu.",
     "sub_domain": "mooc_plateformes"},

    # ── Gamification & Engagement (121-150) ─────────────────
    {"q": "Qu'est-ce que la gamification dans l'éducation ?",
     "a": "La gamification intègre des mécaniques de jeu (points, badges, classements, quêtes, niveaux) dans un contexte pédagogique pour augmenter la motivation et l'engagement des apprenants. Elle s'appuie sur les théories de la motivation intrinsèque et extrinsèque.",
     "sub_domain": "gamification"},

    {"q": "Quels sont les avantages du microlearning ?",
     "a": "Le microlearning propose des contenus courts (3-5 min) faciles à consommer. Il améliore la rétention grâce à l'effet d'espacement, s'adapte au mobile, convient aux emplois du temps chargés et permet un apprentissage juste-à-temps (just-in-time learning).",
     "sub_domain": "gamification"},

    {"q": "Comment mesurer l'engagement des apprenants en ligne ?",
     "a": "L'engagement se mesure via des indicateurs comportementaux (temps passé, clics, complétion), cognitifs (résultats aux quiz, profondeur des réponses), émotionnels (sentiment analysis des commentaires) et sociaux (interactions forum, collaboration).",
     "sub_domain": "gamification"},

    # ── Évaluation automatique (151-180) ────────────────────
    {"q": "Comment l'IA peut-elle générer du feedback personnalisé ?",
     "a": "L'IA génère du feedback en analysant les erreurs spécifiques de l'apprenant, en les classifiant par type, puis en produisant des explications ciblées. Les LLMs permettent un feedback en langage naturel qui explique pourquoi une réponse est incorrecte et guide vers la bonne compréhension.",
     "sub_domain": "evaluation_auto"},

    {"q": "Qu'est-ce que l'évaluation formative automatisée ?",
     "a": "L'évaluation formative automatisée fournit un feedback continu pendant l'apprentissage, sans noter formellement. Elle utilise des quiz adaptatifs, des exercices interactifs et des analyses en temps réel pour identifier les lacunes et orienter l'apprentissage.",
     "sub_domain": "evaluation_auto"},

    {"q": "Comment les modèles de langage évaluent-ils les réponses ouvertes ?",
     "a": "Les modèles de langage évaluent les réponses ouvertes en comparant les embeddings sémantiques avec des réponses de référence, en analysant la complétude, la précision et la structure argumentative. Le fine-tuning sur des données annotées par des enseignants améliore la fiabilité.",
     "sub_domain": "evaluation_auto"},

    # ── Learning Analytics (181-210) ────────────────────────
    {"q": "Qu'est-ce que le Learning Analytics ?",
     "a": "Le Learning Analytics est la mesure, collecte, analyse et reporting des données sur les apprenants et leurs contextes, dans le but de comprendre et optimiser l'apprentissage. Il utilise des techniques de data mining, ML et visualisation de données.",
     "sub_domain": "learning_analytics"},

    {"q": "Comment prédire le décrochage étudiant avec le machine learning ?",
     "a": "La prédiction du décrochage utilise des features comme la fréquence de connexion, les résultats aux évaluations, le temps passé, les interactions forum. Des modèles comme Random Forest, XGBoost ou LSTM sont entraînés sur des données historiques pour identifier les étudiants à risque.",
     "sub_domain": "learning_analytics"},

    {"q": "Qu'est-ce que le xAPI (Experience API) en e-learning ?",
     "a": "Le xAPI est un standard technique qui permet de capturer et stocker toutes les expériences d'apprentissage (formelles et informelles) sous forme de triplets Acteur-Verbe-Objet dans un Learning Record Store (LRS). Il remplace le SCORM pour un suivi plus riche et flexible.",
     "sub_domain": "learning_analytics"},

    # ── Design pédagogique (211-240) ────────────────────────
    {"q": "Quels sont les principes du design pédagogique (Instructional Design) ?",
     "a": "Le design pédagogique s'appuie sur des modèles comme ADDIE (Analyse, Design, Développement, Implémentation, Évaluation) et SAM (Successive Approximation Model) pour concevoir des formations structurées et efficaces, centrées sur les objectifs d'apprentissage.",
     "sub_domain": "design_pedagogique"},

    {"q": "Qu'est-ce que le blended learning et quels sont ses avantages ?",
     "a": "Le blended learning (apprentissage mixte) combine formation en présentiel et formation en ligne. Il tire parti des avantages des deux : l'interaction humaine et le feedback immédiat du présentiel, avec la flexibilité et la personnalisation du numérique.",
     "sub_domain": "design_pedagogique"},

    {"q": "Comment concevoir un parcours d'apprentissage adaptatif ?",
     "a": "Un parcours adaptatif nécessite un modèle de l'apprenant (pré-test, analytics), un graphe de connaissances (prérequis, objectifs), des règles d'adaptation (seuils de maîtrise), et un moteur de recommandation qui sélectionne le contenu optimal pour chaque apprenant.",
     "sub_domain": "design_pedagogique"},

    # ── Technologies émergentes (241-270) ───────────────────
    {"q": "Comment la réalité virtuelle est-elle utilisée en e-learning ?",
     "a": "La VR en éducation permet des simulations immersives (laboratoires virtuels, visites de terrain, chirurgie simulée), l'apprentissage expérientiel sécurisé, et l'entraînement aux compétences pratiques. Elle augmente l'engagement et la rétention en créant des expériences mémorables.",
     "sub_domain": "technologies_emergentes"},

    {"q": "Qu'est-ce que le Social Learning et comment fonctionne-t-il en ligne ?",
     "a": "Le Social Learning repose sur l'apprentissage par les pairs via des forums, groupes de discussion, projets collaboratifs et partage de connaissances. En ligne, il utilise les wikis, les annotations sociales, les vidéos commentées et les communautés de pratique virtuelles.",
     "sub_domain": "technologies_emergentes"},

    # ── Éthique IA en éducation (271-300) ───────────────────
    {"q": "Quels sont les enjeux éthiques de l'IA en éducation ?",
     "a": "Les enjeux incluent la protection des données personnelles des étudiants (RGPD), les biais algorithmiques dans l'évaluation, la transparence des décisions automatisées, le risque de remplacement des enseignants, et l'équité d'accès aux technologies éducatives.",
     "sub_domain": "ethique_ia"},

    {"q": "Comment l'IA transforme-t-elle le rôle de l'enseignant ?",
     "a": "L'IA automatise les tâches répétitives (correction, suivi) et permet à l'enseignant de se concentrer sur le mentorat, la motivation et l'accompagnement individualisé. Le rôle évolue de transmetteur de savoirs vers facilitateur et concepteur d'expériences d'apprentissage.",
     "sub_domain": "ethique_ia"},
]


class DatasetGenerator:
    """Génère un dataset d'évaluation Q/R (templates ou extraits PDF + LLM)."""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.num_questions = EVALUATION_CONFIG["num_questions"]

    def _pick_chunks(self, chunks: list, n: int) -> list:
        """Échantillonne n chunks (avec remise si n > len(chunks))."""
        if not chunks:
            return []
        if n <= len(chunks):
            return random.sample(chunks, n)
        return random.choices(chunks, k=n)

    def generate_from_pdf_chunks(self, chunks: list, llm, max_pairs: Optional[int] = None) -> List[Dict]:
        """
        Génère des paires Q/R alignées sur le contenu des chunks (PDFs indexés).
        `llm` : modèle chat LangChain (ex. ChatOpenAI).
        """
        if not chunks:
            return []
        target = max_pairs if max_pairs is not None else min(
            self.num_questions,
            EVALUATION_CONFIG.get("dataset_pdf_max_pairs", 60),
        )
        target = max(1, min(target, self.num_questions))

        chain = QA_FROM_TEXT_PROMPT | llm | StrOutputParser()
        dataset: List[Dict] = []
        selected = self._pick_chunks(chunks, target)

        for i, chunk in enumerate(selected):
            text = (chunk.page_content or "")[:4000]
            if len(text.strip()) < 80:
                logger.warning("Chunk trop court, ignoré (source=%s)", chunk.metadata.get("source_file"))
                continue
            source_file = chunk.metadata.get("source_file", "unknown")
            sub = Path(str(source_file)).stem if source_file else "general"
            try:
                raw = chain.invoke({"text": text})
                qa = _parse_json_object(raw)
                q = (qa.get("question") or "").strip()
                a = (qa.get("answer") or "").strip()
                if not q or not a:
                    raise ValueError("question ou answer vide")
                dataset.append(
                    {
                        "id": f"pdf_{len(dataset):04d}",
                        "domain": "documents",
                        "sub_domain": sub[:80],
                        "question": q,
                        "answer": a,
                        "source_file": source_file,
                        "source": "pdf_chunk",
                        "difficulty": "moyen",
                        "variant": i,
                    }
                )
            except Exception as e:
                if _is_inference_quota_exhausted(e):
                    logger.error(
                        "Crédits API épuisés (402) — arrêt de la génération Q/R depuis les PDFs "
                        "(HF Inference Providers ou quota similaire). Paires déjà générées : %s",
                        len(dataset),
                    )
                    break
                logger.warning("Échec génération Q/R pour chunk %s : %s", source_file, e)

        logger.info("Dataset depuis PDFs : %s paires générées / %s tentatives", len(dataset), len(selected))
        return dataset

    def generate_from_documents(self, chunks: list, count: int, llm=None) -> List[Dict]:
        """Alias : génère `count` paires à partir des documents (nécessite `llm`)."""
        model = llm or self.llm_client
        if model is None:
            raise ValueError("Un LLM LangChain (chat) est requis.")
        return self.generate_from_pdf_chunks(chunks, model, max_pairs=count)

    def generate_local_dataset(self) -> List[Dict]:
        """
        Génère 300 paires Q/R sur le e-learning à partir des templates.
        Utilise les 30 templates de base + génération de variantes.
        """
        logger.info("Génération du dataset local (mode template) — Domaine : E-LEARNING")
        dataset = []
        templates = ELEARNING_QA_TEMPLATES

        # Phase 1 : Templates originaux (30 questions)
        for i, tpl in enumerate(templates):
            dataset.append({
                "id": f"elearn_{i:04d}",
                "domain": "e-learning",
                "sub_domain": tpl.get("sub_domain", "general"),
                "question": tpl["q"],
                "answer": tpl["a"],
                "source": "template",
                "difficulty": "moyen",
                "variant": 0,
            })

        # Phase 2 : Variantes pour atteindre 300 (270 variantes supplémentaires)
        variant_prefixes = [
            ("Expliquez en détail ", "comment fonctionne "),
            ("Décrivez les principaux aspects de ", ""),
            ("Quels sont les avantages et inconvénients de ", " ?"),
            ("Comment implémenter ", " dans un système e-learning ?"),
            ("Quelle est l'importance de ", " pour l'éducation en ligne ?"),
            ("En quoi ", " améliore-t-il l'apprentissage en ligne ?"),
            ("Comparez ", " avec les approches traditionnelles."),
            ("Donnez un exemple concret de ", " en contexte éducatif."),
            ("Quels défis pose ", " pour les institutions éducatives ?"),
        ]

        target = self.num_questions
        idx = len(dataset)

        while len(dataset) < target:
            tpl = templates[idx % len(templates)]
            variant_num = (idx // len(templates)) + 1

            # Extraire le sujet principal de la question
            subject = self._extract_subject(tpl["q"])
            prefix, suffix = variant_prefixes[variant_num % len(variant_prefixes)]

            variant_q = f"{prefix}{subject}{suffix}"
            variant_a = self._enrich_answer(tpl["a"], variant_num)

            difficulties = ["facile", "moyen", "difficile"]

            dataset.append({
                "id": f"elearn_{idx:04d}",
                "domain": "e-learning",
                "sub_domain": tpl.get("sub_domain", "general"),
                "question": variant_q,
                "answer": variant_a,
                "source": "template_variant",
                "difficulty": difficulties[variant_num % 3],
                "variant": variant_num,
            })
            idx += 1

        random.shuffle(dataset)
        logger.info(f"Dataset généré : {len(dataset)} paires Q/R (e-learning)")
        return dataset[:target]

    def _extract_subject(self, question: str) -> str:
        """Extrait le sujet principal d'une question."""
        q = question.lower()
        removals = [
            "qu'est-ce que ", "qu'est-ce qu'", "comment ", "quels sont ",
            "quelle est ", "quel est ", "expliquez ", "décrivez ",
        ]
        for r in removals:
            if q.startswith(r):
                subject = question[len(r):]
                return subject.rstrip(" ?").rstrip(".")
        return question.rstrip(" ?")

    def _enrich_answer(self, base_answer: str, seed: int) -> str:
        """Enrichit une réponse avec des détails supplémentaires."""
        additions = [
            " Cette approche est de plus en plus adoptée dans les universités et les entreprises.",
            " Les recherches récentes montrent des résultats prometteurs en termes d'efficacité pédagogique.",
            " L'implémentation nécessite une attention particulière à l'expérience utilisateur et à l'accessibilité.",
            " Les plateformes modernes comme Coursera, Moodle et Canvas intègrent progressivement ces fonctionnalités.",
            " Cette technologie est particulièrement pertinente dans le contexte de l'éducation post-pandémie.",
            " Les enseignants jouent un rôle crucial dans la supervision et l'amélioration continue de ces systèmes.",
            " L'évaluation de l'efficacité repose sur le modèle de Kirkpatrick et les learning analytics.",
            " Les défis incluent la scalabilité, la protection des données et l'interopérabilité des systèmes.",
            " L'approche centrée sur l'apprenant est fondamentale pour le succès de cette implémentation.",
        ]
        return base_answer + additions[seed % len(additions)]

    def save_dataset(
        self,
        dataset: List[Dict],
        filename: str = "dataset_evaluation.json",
        metadata_extra: Optional[Dict] = None,
    ) -> Path:
        """Sauvegarde le dataset au format JSON."""
        output_path = EVALUATION_DIR / filename
        sub_domains = {}
        for d in dataset:
            sd = d.get("sub_domain", "general")
            sub_domains[sd] = sub_domains.get(sd, 0) + 1

        pdf_files = sorted(
            {d.get("source_file") for d in dataset if d.get("source_file")}
        )
        generation = (metadata_extra or {}).get("generation", "template")
        primary_domain = "documents" if generation == "pdf_chunks" else "e-learning"

        meta = {
            "version": "1.1",
            "generated_at": datetime.now().isoformat(),
            "total_questions": len(dataset),
            "domain": primary_domain,
            "generation": generation,
            "sub_domains": sub_domains,
        }
        if pdf_files:
            meta["pdf_sources"] = pdf_files
        if metadata_extra:
            for k, v in metadata_extra.items():
                if k != "generation" and k not in meta:
                    meta[k] = v

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"metadata": meta, "data": dataset}, f, ensure_ascii=False, indent=2)
        logger.info(f"Dataset sauvegardé → {output_path}")
        return output_path

    def load_dataset(self, filename: str = "dataset_evaluation.json") -> List[Dict]:
        """Charge un dataset existant."""
        path = EVALUATION_DIR / filename
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["data"]


# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    gen = DatasetGenerator()
    dataset = gen.generate_local_dataset()
    path = gen.save_dataset(dataset)

    print(f"\n✅ Dataset généré avec {len(dataset)} questions")
    print(f"📁 Fichier : {path}")
    print(f"📌 Domaine : E-LEARNING uniquement")
    print(f"\nRépartition par sous-domaine :")
    from collections import Counter
    for sd, count in Counter(d.get("sub_domain", "general") for d in dataset).most_common():
        print(f"  • {sd}: {count}")
