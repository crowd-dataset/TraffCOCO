"""
Knowledge-based ontology candidate reranker.

Redesigned CandidateRanker

The reranker now performs evidence fusion instead of treating every ontology
attribute independently.

Evidence Sources
----------------
1. Visual Evidence
2. Text Evidence
3. Context Evidence
4. Semantic Evidence
5. Knowledge Evidence

Each module returns:

(obtained_score, maximum_possible_score)

The final attribute score is the normalized fusion of all evidence.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from custom_logger import CustomLogger

from annotation_pipeline.configs.settings import PipelineConfig

from annotation_pipeline.models.ontology.ontology_models import (
    CandidateMatch,
    RetrievalQuery,
)

logger = CustomLogger(__name__)


class CandidateRanker:
    """
    Ontology-based evidence fusion reranker.
    """

    def __init__(
        self,
        config: PipelineConfig,
    ) -> None:

        self.config = config

        self.embedding_weight = 0.60
        self.attribute_weight = 0.40

        logger.info(
            "Initialized ontology evidence ranker."
        )

    # ==========================================================
    # Public API
    # ==========================================================

    def rank_candidates(
        self,
        query: RetrievalQuery,
        candidates: list[CandidateMatch],
    ) -> list[CandidateMatch]:

        logger.info(
            "Ranking {} ontology candidate(s).",
            len(candidates),
        )

        for candidate in candidates:

            candidate.matched_attributes = {}
            candidate.penalties = []

            candidate.attribute_score = (
                self._compute_attribute_score(
                    query,
                    candidate,
                )
            )

            candidate.final_score = (

                self.embedding_weight
                * candidate.embedding_score

                +

                self.attribute_weight
                * candidate.attribute_score

            )

            logger.debug(
                "{} | embedding={:.3f} | attribute={:.3f} | final={:.3f}",
                candidate.entry.class_name,
                candidate.embedding_score,
                candidate.attribute_score,
                candidate.final_score,
            )

        candidates.sort(

            key=lambda c: c.final_score,

            reverse=True,

        )

        return candidates

    
    # ==========================================================
    # Evidence Fusion
    # ==========================================================

    def _compute_attribute_score(
        self,
        query: RetrievalQuery,
        candidate: CandidateMatch,
    ) -> float:
        """
        Fuse every ontology evidence source.
        """

        evidence = [
            self._score_visual(
                query,
                candidate,
            ),
            self._score_textual(
                query,
                candidate,
            ),
            self._score_contextual(
                query,
                candidate,
            ),
            self._score_semantic(
                query,
                candidate,
            ),
            self._score_embedding_semantics(
                query,
                candidate,
            ),
            self._score_knowledge(
                query,
                candidate,
            ),
        ]

        obtained = sum(
            score
            for score, _
            in evidence
        )

        maximum = sum(
            weight
            for _, weight
            in evidence
        )

        penalty = (
            self._apply_common_confusions(
                query,
                candidate,
            )
            + self._apply_negative_cues(
                query,
                candidate,
            )
        )

        if maximum > 0:
            penalty = min(
                penalty,
                maximum * 0.35,
            )

        obtained -= penalty

        if maximum <= 0:
            return 0.0

        score = obtained / maximum

        return max(
            0.0,
            min(score, 1.0),
        )

    # ==========================================================
    # Matching Utilities
    # ==========================================================

    @staticmethod
    def _normalize(
        value: str | None,
    ) -> str:

        if not value:
            return ""

        value = value.lower().strip()

        replacements = {

            "circular": "circle",
            "rectangular": "rectangle",
            "triangular": "triangle",
            "square-shaped": "square",

            "conical": "cone",
            "cone-shaped": "cone",

            "human": "human silhouette",

            "metallic": "metal",

            "grey": "gray",

            "lamp post": "street light",

        }

        for old, new in replacements.items():

            value = value.replace(
                old,
                new,
            )

        return value

    @staticmethod
    def _normalize_list(
        values: list[str] | None,
    ) -> list[str]:

        if not values:
            return []

        return [

            item.strip().lower()

            for item in values

            if item

        ]

    @staticmethod
    def _similarity(
        a: str,
        b: str,
    ) -> float:

        a = a.lower().strip()
        b = b.lower().strip()

        return SequenceMatcher(
            None,
            a,
            b,
        ).ratio()

    def _fuzzy_match(
        self,
        observed: str | None,
        ontology_values: list[str],
    ) -> tuple[bool, float]:
        """
        Returns
        -------
        (matched, similarity)
        """

        if not observed:
            return False, 0.0

        observed = self._normalize(observed)
        ontology_values = self._normalize_list(ontology_values)

        best = 0.0

        for value in ontology_values:

            if observed == value:
                return True, 1.0

            similarity = max(
                self._similarity(observed, value),
                self._similarity(value, observed),
            )

            if observed in value:
                similarity = max(similarity, 0.90)

            if value in observed:
                similarity = max(similarity, 0.90)

            best = max(best, similarity)

        return best >= 0.70, best

    @staticmethod
    def _tokenize_text(
        text: str | None,
    ) -> list[str]:
        """
        Tokenize OCR text while preserving:
        - arrows
        - Japanese, Chinese, Korean
        - identifiers such as 21-2 and A-15
        - numbers and Latin text
        """

        if not text:
            return []

        tokens = re.findall(
            r"[↑↓←→↖↗↘↙]|[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*|[\u3040-\u30ff\u3400-\u9fff\uAC00-\uD7AF]+",
            text,
            flags=re.UNICODE,
        )

        return [
            token.strip().lower()
            for token in tokens
            if token.strip()
        ]

    

    def _get_importance(
        self,
        candidate: CandidateMatch,
        attribute: str,
        default: float,
    ) -> float:

        importance = candidate.entry.data.get(
            "attribute_importance",
            {},
        )

        level = importance.get(
            attribute,
            default * 5,
        )

        return float(level) / 5.0

    # ==========================================================
    # Evidence Modules
    # ==========================================================

    # ==========================================================
    # Visual Evidence
    # ==========================================================

    def _score_visual(
        self,
        query: RetrievalQuery,
        candidate: CandidateMatch,
    ) -> tuple[float, float]:
        """
        Compute visual evidence.

        Uses every available visual attribute from the ontology.
        """

        visual = candidate.entry.data.get(
            "visual_attributes",
            {},
        )

        obtained = 0.0
        maximum = 0.0

        scorers = [
            (
                "shape",
                query.shape,
                visual.get("typical_shapes", []),
                "shape",
                0.25,
            ),
            (
                "primary_color",
                query.primary_color,
                visual.get("typical_colors", []),
                "color",
                0.20,
            ),
            (
                "secondary_color",
                query.secondary_color,
                visual.get("typical_colors", []),
                "color",
                0.10,
            ),
            (
                "material",
                query.material,
                visual.get("material", []),
                "material",
                0.15,
            ),
        ]

        for (
            name,
            observed,
            ontology_values,
            importance_name,
            default,
        ) in scorers:
            weight = self._get_importance(
                candidate,
                importance_name,
                default,
            )

            maximum += weight

            matched, similarity = self._fuzzy_match(
                observed,
                ontology_values,
            )

            if matched:
                obtained += weight * similarity
                candidate.matched_attributes[name] = {
                    "value": observed,
                    "similarity": round(similarity, 3),
                }

        visual_characteristics_weight = self._get_importance(
            candidate,
            "visual_characteristics",
            0.15,
        )

        maximum += visual_characteristics_weight

        obtained += self._score_visual_characteristics(
            query,
            candidate,
            visual,
        )

        reflective_weight = self._get_importance(
            candidate,
            "reflective",
            0.05,
        )

        maximum += reflective_weight

        obtained += self._score_reflective(
            query,
            candidate,
            visual,
        )

        return obtained, maximum

    def _score_visual_characteristics(
        self,
        query: RetrievalQuery,
        candidate: CandidateMatch,
        visual: dict,
    ) -> float:

        characteristics = visual.get(
            "visual_characteristics",
            [],
        )

        if not characteristics:
            return 0.0

        description = self._normalize(
            query.description,
        )

        weight = self._get_importance(
            candidate,
            "visual_characteristics",
            0.15,
        )

        for characteristic in characteristics:

            characteristic = self._normalize(
                characteristic,
            )

            if characteristic in description:

                candidate.matched_attributes[
                    "visual_characteristics"
                ] = characteristic

                return weight

            similarity = self._similarity(
                description,
                characteristic,
            )

            if similarity > 0.80:

                candidate.matched_attributes[
                    "visual_characteristics"
                ] = {
                    "value": characteristic,
                    "similarity": round(
                        similarity,
                        3,
                    ),
                }

                return weight * similarity

        return 0.0

    def _score_reflective(
        self,
        query: RetrievalQuery,
        candidate: CandidateMatch,
        visual: dict,
    ) -> float:

        ontology_reflective = visual.get(
            "reflective",
            None,
        )

        if ontology_reflective is None:
            return 0.0

        if not hasattr(
            query,
            "reflective",
        ):
            return 0.0

        if query.reflective is None:
            return 0.0

        if ontology_reflective == query.reflective:

            candidate.matched_attributes[
                "reflective"
            ] = ontology_reflective

            return self._get_importance(
                candidate,
                "reflective",
                0.05,
            )

        return 0.0

    # ==========================================================
    # Textual Evidence
    # ==========================================================

    def _score_textual(
        self,
        query: RetrievalQuery,
        candidate: CandidateMatch,
    ) -> tuple[float, float]:
        """
        Score every textual cue.

        Uses:
        - OCR text
        - Common text
        - Common symbols
        - Languages
        """

        text_info = candidate.entry.data.get(
            "text_information",
            {},
        )

        obtained = 0.0
        maximum = 0.0

        #########################################
        # OCR Parsing
        #########################################

        tokens = self._tokenize_text(
            query.text,
        )

        if hasattr(query, "symbol") and query.symbol:
            tokens.extend(
                self._tokenize_text(
                    query.symbol,
                )
            )

        tokens = list(dict.fromkeys(tokens))

        #########################################
        # Ontology
        #########################################

        common_text = self._normalize_list(
            text_info.get("common_text", []),
        )

        common_symbols = self._normalize_list(
            text_info.get("common_symbols", []),
        )

        languages = self._normalize_list(
            text_info.get("languages", []),
        )

        #########################################
        # Text
        #########################################

        text_weight = self._get_importance(
            candidate,
            "text",
            0.20,
        )

        maximum += text_weight

        matched_text = []
        best_similarity = 0.0

        for token in tokens:

            matched, similarity = self._fuzzy_match(
                token,
                common_text,
            )

            if matched:

                matched_text.append(
                    {
                        "token": token,
                        "similarity": round(similarity, 3),
                    }
                )

                best_similarity = max(
                    best_similarity,
                    similarity,
                )

        if matched_text:

            obtained += (
                text_weight
                * best_similarity
            )

            candidate.matched_attributes["text"] = matched_text

        #########################################
        # Symbols
        #########################################

        symbol_weight = self._get_importance(
            candidate,
            "symbol",
            0.20,
        )

        maximum += symbol_weight

        matched_symbols = []

        for token in tokens:
            matched, similarity = self._fuzzy_match(
                token,
                common_symbols,
            )

            if matched:
                matched_symbols.append(
                    {
                        "token": token,
                        "similarity": round(similarity, 3),
                    }
                )

                obtained += (
                    symbol_weight
                    * similarity
                    / max(len(tokens), 1)
                )

        if matched_symbols:
            candidate.matched_attributes["symbols"] = matched_symbols

        #########################################
        # Languages
        #########################################

        language_weight = self._get_importance(
            candidate,
            "language",
            0.05,
        )

        maximum += language_weight

        description = self._normalize(
            query.description,
        )

        matched_languages = []

        for language in languages:
            if language in description:
                matched_languages.append(language)

        if matched_languages:
            obtained += language_weight
            candidate.matched_attributes["languages"] = matched_languages

        return obtained, maximum

    # ==========================================================
    # Context Evidence
    # ==========================================================

    def _score_contextual(
        self,
        query: RetrievalQuery,
        candidate: CandidateMatch,
    ) -> tuple[float, float]:

        context = candidate.entry.data.get(
            "context",
            {},
        )

        installation = candidate.entry.data.get(
            "installation",
            {},
        )

        obtained = 0.0
        maximum = 0.0

        ##########################################
        # Nearby Objects
        ##########################################

        nearby = self._normalize_list(
            context.get("frequently_nearby_objects", []),
        )

        weight = self._get_importance(
            candidate,
            "context",
            0.25,
        )

        maximum += weight

        matched = []

        for obj in query.nearby_objects:
            matched_flag, similarity = self._fuzzy_match(
                obj,
                nearby,
            )

            if matched_flag:
                matched.append(
                    {
                        "object": obj,
                        "similarity": round(similarity, 3),
                    }
                )

        if matched:
            obtained += weight * (
                len(matched) / max(len(query.nearby_objects), 1)
            )
            candidate.matched_attributes["nearby_objects"] = matched

        ##########################################
        # Mounted On
        ##########################################

        mounted = self._normalize_list(
            installation.get("mounted_on", []),
        )

        weight = self._get_importance(
            candidate,
            "location",
            0.20,
        )

        maximum += weight

        matched_flag, similarity = self._fuzzy_match(
            query.attached_to,
            mounted,
        )

        if matched_flag:
            obtained += weight * similarity
            candidate.matched_attributes["mounted_on"] = {
                "value": query.attached_to,
                "similarity": round(similarity, 3),
            }

        ##########################################
        # Typical Location
        ##########################################

        locations = self._normalize_list(
            context.get("typical_location", []),
        )

        location_weight = self._get_importance(
            candidate,
            "location",
            0.15,
        )

        maximum += location_weight

        description = self._normalize(
            query.description,
        )

        matched_locations = []

        for location in locations:
            if location in description:
                matched_locations.append(location)

        if matched_locations:
            obtained += location_weight
            candidate.matched_attributes["typical_location"] = matched_locations

        ##########################################
        # Road Side
        ##########################################

        if hasattr(query, "road_side") and query.road_side:
            road_side_weight = self._get_importance(
                candidate,
                "road_side",
                0.05,
            )

            maximum += road_side_weight

            if query.road_side.lower() in description:
                obtained += road_side_weight
                candidate.matched_attributes["road_side"] = query.road_side

        return obtained, maximum

    # ==========================================================
    # Semantic Evidence
    # ==========================================================

    def _score_semantic(
        self,
        query: RetrievalQuery,
        candidate: CandidateMatch,
    ) -> tuple[float, float]:

        semantic = candidate.entry.data.get(
            "semantic_information",
            {},
        )

        cues = candidate.entry.data.get(
            "inference_cues",
            {},
        )

        obtained = 0.0
        maximum = 0.0

        ##########################################
        # Keywords / Synonyms / Aliases
        ##########################################

        words = (
            semantic.get("keywords", [])
            + semantic.get("synonyms", [])
            + semantic.get("aliases", [])
        )

        words = self._normalize_list(words)

        semantic_weight = self._get_importance(
            candidate,
            "semantic",
            0.40,
        )

        maximum += semantic_weight

        description = self._normalize(
            query.description,
        )

        observed = self._normalize(
            query.observed_object,
        )

        matched = []
        best_similarity = 0.0

        for word in words:

            matched_flag, similarity = self._fuzzy_match(
                observed,
                [word],
            )

            if matched_flag:
                matched.append(
                    {
                        "word": word,
                        "similarity": round(similarity, 3),
                    }
                )
                best_similarity = max(best_similarity, similarity)

            elif word in description:
                matched.append(
                    {
                        "word": word,
                        "similarity": 1.0,
                    }
                )
                best_similarity = 1.0

        if matched:
            obtained += semantic_weight * best_similarity
            candidate.matched_attributes["semantic"] = matched

        ##########################################
        # Likely Descriptions
        ##########################################

        descriptions = cues.get(
            "likely_descriptions",
            [],
        )

        description_weight = self._get_importance(
            candidate,
            "description",
            0.25,
        )

        maximum += description_weight

        best_similarity = 0.0
        best_match = None

        for cue in descriptions:
            similarity = self._similarity(
                description,
                cue.lower(),
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = cue

        if best_similarity > 0.75:
            obtained += description_weight * best_similarity
            candidate.matched_attributes["description"] = best_match

        ##########################################
        # Purpose
        ##########################################

        purpose = candidate.entry.data.get(
            "purpose",
            "",
        )

        purpose_weight = self._get_importance(
            candidate,
            "purpose",
            0.15,
        )

        maximum += purpose_weight

        if query.possible_function:
            similarity = self._similarity(
                query.possible_function,
                purpose,
            )

            if similarity > 0.70:
                obtained += purpose_weight * similarity
                candidate.matched_attributes["purpose"] = purpose

        return obtained, maximum

    def _score_embedding_semantics(
        self,
        query,
        candidate,
    ) -> tuple[float, float]:

        """
        Semantic similarity between

        Gemma description

        and

        ontology embedding_text
        """

        embedding_text = candidate.entry.data.get(
            "embedding_text",
            "",
        )

        if not embedding_text:
            return 0.0, 0.0

        similarity = self._similarity(

            self._normalize(query.description),

            self._normalize(embedding_text),

        )

        weight = self._get_importance(
            candidate,
            "embedding_semantics",
            0.30,
        )

        candidate.matched_attributes[
            "embedding_semantics"
        ] = round(similarity, 3)

        return (
            similarity * weight,
            weight,
        )

    # ==========================================================
    # Knowledge Evidence
    # ==========================================================

    def _score_knowledge(
        self,
        query: RetrievalQuery,
        candidate: CandidateMatch,
    ) -> tuple[float, float]:
        """
        Score ontology knowledge relationships.

        Uses
        - parent
        - related classes
        - distinctive features
        - object group
        - category
        """

        obtained = 0.0
        maximum = 0.0

        ##############################################
        # Parent Category
        ##############################################

        relationships = candidate.entry.data.get(
            "relationships",
            {},
        )

        parent = relationships.get(
            "parent",
            "",
        )

        weight = self._get_importance(
            candidate,
            "category",
            0.15,
        )

        maximum += weight

        matched_flag, similarity = self._fuzzy_match(
            query.classification_hint,
            [parent],
        )

        if matched_flag:
            obtained += weight * similarity
            candidate.matched_attributes["parent"] = {
                "value": parent,
                "similarity": round(similarity, 3),
            }

        ##############################################
        # Super Category
        ##############################################

        super_category = candidate.entry.data.get(
            "super_category",
            "",
        )

        weight = self._get_importance(
            candidate,
            "object_group",
            0.15,
        )

        maximum += weight

        matched_flag, similarity = self._fuzzy_match(
            query.object_group,
            [super_category],
        )

        if matched_flag:
            obtained += weight * similarity
            candidate.matched_attributes["super_category"] = {
                "value": super_category,
                "similarity": round(similarity, 3),
            }

        ##############################################
        # Related Classes
        ##############################################

        related = self._normalize_list(
            relationships.get("related_classes", []),
        )

        maximum += 0.10

        matched_related = []

        for obj in query.nearby_objects:

            matched_flag, similarity = self._fuzzy_match(
                obj,
                related,
            )

            if matched_flag:

                matched_related.append(
                    {
                        "object": obj,
                        "similarity": round(similarity, 3),
                    }
                )

        if matched_related:

            best_similarity = max(
                item["similarity"]
                for item in matched_related
            )

            obtained += 0.10 * best_similarity

            candidate.matched_attributes[
                "related_classes"
            ] = matched_related
        ##############################################
        # Distinctive Features
        ##############################################

        features = self._normalize_list(
            candidate.entry.data.get("distinctive_features", []),
        )

        weight = self._get_importance(
            candidate,
            "distinctive_features",
            0.30,
        )

        maximum += weight

        description = self._normalize(
            query.description,
        )

        matched = []

        for feature in features:
            similarity = self._similarity(
                description,
                feature,
            )

            if similarity > 0.75:
                matched.append(
                    {
                        "feature": feature,
                        "similarity": round(similarity, 3),
                    }
                )

        if matched:
            average_similarity = (
                sum(
                    item["similarity"]
                    for item in matched
                )
                /
                len(matched)
            )

            coverage = (
                len(matched)
                /
                max(len(features),1)
            )

            score = average_similarity * coverage
            obtained += weight * score
            candidate.matched_attributes["distinctive_features"] = matched

        return obtained, maximum
    # ==========================================================
    def _apply_common_confusions(
        self,
        query: RetrievalQuery,
        candidate: CandidateMatch,
    ) -> float:

        confusions = candidate.entry.data.get(
            "common_confusions",
            [],
        )

        if not confusions:
            return 0.0

        penalty = 0.0

        description = self._normalize(
            query.description,
        )

        observed = self._normalize(
            query.observed_object,
        )

        for confusion in confusions:
            confusion_class = self._normalize(
                confusion.get("class", ""),
            )

            reason = self._normalize(
                confusion.get("reason_for_confusion", ""),
            )

            distinguishing = self._normalize(
                confusion.get("distinguishing_feature", ""),
            )

            if confusion_class in observed:
                penalty += 0.10
                candidate.penalties.append(
                    f"Possible confusion with {confusion_class}"
                )
            elif reason and reason in description:
                penalty += 0.05
            elif distinguishing and distinguishing not in description:
                penalty += 0.05

        return min(penalty, 0.20)

    def _apply_negative_cues(
        self,
        query: RetrievalQuery,
        candidate: CandidateMatch,
    ) -> float:

        cues = candidate.entry.data.get(
            "negative_cues",
            [],
        )

        if not cues:
            return 0.0

        description = self._normalize(
            query.description,
        )

        penalty = 0.0
        matched = []

        for cue in cues:
            cue = self._normalize(cue)

            if cue in description:
                matched.append(cue)
                penalty += 0.10

        if matched:
            candidate.penalties.extend(matched)

        return min(penalty, 0.20)

    @staticmethod
    def best_candidate(
        candidates: list[CandidateMatch],
    ) -> CandidateMatch | None:

        if not candidates:
            return None

        return max(
            candidates,
            key=lambda c: c.final_score,
        )

    def print_candidate_report(
        self,
        candidate: CandidateMatch,
    ) -> None:

        print()

        print("=" * 60)

        print(candidate.entry.class_name)

        print("-" * 60)

        print(

            f"Embedding : {candidate.embedding_score:.3f}"

        )

        print(

            f"Attribute : {candidate.attribute_score:.3f}"

        )

        print(

            f"Final     : {candidate.final_score:.3f}"

        )

        print()

        print("Matched Attributes")

        print("------------------")

        for key, value in candidate.matched_attributes.items():

            print(

                f"{key:<25}: {value}"

            )

        if candidate.penalties:

            print()

            print("Penalties")

            print("---------")

            for penalty in candidate.penalties:

                print(

                    f"- {penalty}"

                )

        print("=" * 60)