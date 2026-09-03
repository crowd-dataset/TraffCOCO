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
6. Class-Specific Discriminative Cue Evidence

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

        self.embedding_weight = 0.7
        self.attribute_weight = 0.3

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
        """Rank ontology candidates using fused visual and semantic evidence.

        Explicit discriminative cues found in the Scene Understanding
        description are treated as evidence, rather than requiring them to
        first appear in a structured attribute field.
        """

        logger.info(
            "Ranking {} ontology candidate(s).",
            len(candidates),
        )

        # ----------------------------------------------------------
        # Generic super-category constraint
        # ----------------------------------------------------------

        query_category = self._normalize(
            query.object_group
        )

        if query_category:

            compatible_candidates = [
                candidate
                for candidate in candidates
                if self._is_category_compatible(
                    query,
                    candidate,
                )
            ]

            if compatible_candidates:

                logger.info(
                    "Category restriction: '{}' -> {}/{} candidates.",
                    query_category,
                    len(compatible_candidates),
                    len(candidates),
                )

                candidates = compatible_candidates

            else:

                logger.warning(
                    "No ontology candidates match object_group='{}'. "
                    "Keeping original candidate set.",
                    query_category,
                )

        # ----------------------------------------------------------
        # Evidence fusion
        # ----------------------------------------------------------

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

            # ------------------------------------------------------
            # Generic class identity
            # ------------------------------------------------------

            identity_score = self._class_identity_score(
                query,
                candidate,
            )

            candidate.final_score += identity_score

            # ------------------------------------------------------
            # Explicit discriminative evidence
            # ------------------------------------------------------
            #
            # Preserve all existing embedding, attribute, and identity
            # scoring, while adding a bounded direct bonus when the
            # observation contains a cue that distinguishes this candidate
            # from closely related ontology classes.
            #
            # Example:
            #   "traffic signal with a red lens visible"
            #
            # must be able to favor traffic_light_red over the generic
            # traffic_light_3_phase class even when both share the generic
            # "traffic signal" identity.
            #
            # This is deliberately applied after normalized attribute
            # fusion so decisive visual state evidence cannot be diluted
            # by unrelated ontology attributes.
            discriminative_bonus = self._score_discriminative_bonus(
                query,
                candidate,
            )

            candidate.final_score += discriminative_bonus

            logger.debug(
                "{} | embedding={:.3f} | attribute={:.3f} "
                "| identity={:.3f} | discriminative={:.3f} | final={:.3f}",
                candidate.entry.class_name,
                candidate.embedding_score,
                candidate.attribute_score,
                identity_score,
                discriminative_bonus,
                candidate.final_score,
            )

        candidates.sort(
            key=lambda c: c.final_score,
            reverse=True,
        )
        return candidates

    def _class_identity_score(
        self,
        query: RetrievalQuery,
        candidate: CandidateMatch,
    ) -> float:
        """
        Score generic semantic identity between the observed object
        and the ontology class.

        Exact matches receive a strong bonus.

        Legitimate compound classes such as:
            bus -> city_bus
            traffic signal -> traffic_light_3_phase

        are not penalized merely because the ontology class contains
        additional specificity.

        Cross-domain compounds are not rewarded.
        """

        observed = self._normalize(
            query.observed_object
        )

        if not observed:
            return 0.0

        semantic = candidate.entry.data.get(
            "semantic_information",
            {},
        )

        class_name = self._normalize(
            candidate.entry.class_name
        )

        aliases = [
            self._normalize(value)
            for value in semantic.get(
                "aliases",
                [],
            )
            if value
        ]

        synonyms = [
            self._normalize(value)
            for value in semantic.get(
                "synonyms",
                [],
            )
            if value
        ]

        identity_terms = {
            class_name,
            *aliases,
            *synonyms,
        }

        # ----------------------------------------------------------
        # Exact semantic identity
        # ----------------------------------------------------------

        if observed in identity_terms:

            candidate.matched_attributes[
                "class_identity"
            ] = {
                "type": "exact",
                "observed": observed,
            }

            return 0.15

        # ----------------------------------------------------------
        # Token-level identity
        # ----------------------------------------------------------

        observed_tokens = set(
            re.findall(
                r"[a-z0-9]+",
                observed,
            )
        )

        best_similarity = 0.0
        best_term = None

        for term in identity_terms:

            if not term:
                continue

            term_tokens = set(
                re.findall(
                    r"[a-z0-9]+",
                    term,
                )
            )

            if (
                observed_tokens
                and observed_tokens.issubset(term_tokens)
            ):
                similarity = 0.75
            else:
                similarity = self._similarity(
                    observed,
                    term,
                )

            if similarity > best_similarity:
                best_similarity = similarity
                best_term = term

        # ----------------------------------------------------------
        # Strong lexical identity
        # ----------------------------------------------------------

        if best_similarity >= 0.85:

            candidate.matched_attributes[
                "class_identity"
            ] = {
                "type": "similar",
                "observed": observed,
                "candidate": best_term,
                "similarity": round(
                    best_similarity,
                    3,
                ),
            }

            return 0.10

        return 0.0

    def _is_category_compatible(
        self,
        query: RetrievalQuery,
        candidate: CandidateMatch,
    ) -> bool:
        """
        Check whether the ontology candidate belongs to the same
        broad semantic category as the observed scene object.

        Scene Understanding provides:
            query.object_group

        Ontology provides:
            candidate.entry.data["super_category"]

        These two fields use the same controlled vocabulary.
        """

        query_category = self._normalize(
            query.object_group
        )

        candidate_category = self._normalize(
            candidate.entry.data.get(
                "super_category",
                "",
            )
        )

        if not query_category or not candidate_category:
            return True

        return query_category == candidate_category

    def _score_class_identity(
        self,
        query: RetrievalQuery,
        candidate: CandidateMatch,
    ) -> tuple[float, float]:
        """
        Strong class-identity signal.

        Exact object identity is intentionally stronger than fuzzy
        semantic similarity. This prevents compound ontology classes
        such as 'bus_lane_sign' from beating the actual 'bus' class
        merely because they contain the same token.
        """

        observed = self._normalize(
            query.observed_object
        )

        class_name = self._normalize(
            candidate.entry.class_name
        )

        if not observed or not class_name:
            return 0.0, 0.0

        weight = 1.0

        # Exact class identity.
        if observed == class_name:
            candidate.matched_attributes[
                "class_identity"
            ] = {
                "type": "exact",
                "observed": observed,
                "candidate": class_name,
            }

            return weight, weight

        # Token-level comparison.
        observed_tokens = set(
            re.findall(
                r"[a-z0-9]+",
                observed,
            )
        )

        candidate_tokens = set(
            re.findall(
                r"[a-z0-9]+",
                class_name,
            )
        )

        # The observed class is only one component of a more specific
        # compound class, e.g. bus -> bus_lane_sign.
        if (
            observed_tokens
            and observed_tokens.issubset(candidate_tokens)
            and observed != class_name
        ):
            candidate.matched_attributes[
                "class_identity"
            ] = {
                "type": "compound_mismatch",
                "observed": observed,
                "candidate": class_name,
            }

            return 0.0, weight

        # Otherwise use a weak lexical similarity.
        similarity = self._similarity(
            observed,
            class_name,
        )

        if similarity >= 0.85:
            candidate.matched_attributes[
                "class_identity"
            ] = {
                "type": "similar",
                "observed": observed,
                "candidate": class_name,
                "similarity": round(
                    similarity,
                    3,
                ),
            }

            return similarity * 0.25, weight

        return 0.0, weight

    
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
            self._score_class_identity(
                query,
                candidate,
            ),

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

            self._score_observed_evidence(
                query,
                candidate,
            ),

            self._score_class_specific_cues(
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

            observed_tokens = set(
                re.findall(
                    r"[a-z0-9]+",
                    observed,
                )
            )

            value_tokens = set(
                re.findall(
                    r"[a-z0-9]+",
                    value,
                )
            )

            if observed_tokens == value_tokens:
                similarity = max(
                    similarity,
                    1.0,
                )

            elif (
                observed_tokens
                and observed_tokens.issubset(value_tokens)
            ):
                # Do NOT treat a compound class as an exact match.
                similarity = max(
                    similarity,
                    0.55,
                )

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
        Score textual and language-related evidence.

        Uses:
        - observed text
        - observed symbols
        - language information
        - semantic text descriptions
        - ontology keywords/synonyms/aliases

        The scorer is generic and does not contain class-specific rules.
        """

        text_info = candidate.entry.data.get(
            "text_information",
            {},
        )

        semantic = candidate.entry.data.get(
            "semantic_information",
            {},
        )

        obtained = 0.0
        maximum = 0.0

        description = self._normalize(
            query.description
        )

        observed_text = self._normalize(
            query.text
        )

        observed_symbol = self._normalize(
            query.symbol
        )

        # ----------------------------------------------------------
        # Observed text tokens
        # ----------------------------------------------------------

        tokens = self._tokenize_text(
            query.text
        )

        if observed_symbol:
            tokens.extend(
                self._tokenize_text(
                    query.symbol
                )
            )

        tokens = list(
            dict.fromkeys(tokens)
        )

        # ----------------------------------------------------------
        # Ontology text knowledge
        # ----------------------------------------------------------

        common_text = self._normalize_list(
            text_info.get(
                "common_text",
                [],
            )
        )

        common_symbols = self._normalize_list(
            text_info.get(
                "common_symbols",
                [],
            )
        )

        languages = self._normalize_list(
            text_info.get(
                "languages",
                [],
            )
        )

        semantic_terms = self._normalize_list(
            semantic.get(
                "keywords",
                []
            )
            +
            semantic.get(
                "synonyms",
                []
            )
            +
            semantic.get(
                "aliases",
                []
            )
        )

        # ==========================================================
        # Text
        # ==========================================================

        text_weight = self._get_importance(
            candidate,
            "text",
            0.20,
        )

        maximum += text_weight

        best_text_similarity = 0.0
        matched_text = []

        for token in tokens:

            matched, similarity = self._fuzzy_match(
                token,
                common_text + semantic_terms,
            )

            if matched:

                best_text_similarity = max(
                    best_text_similarity,
                    similarity,
                )

                matched_text.append(
                    {
                        "token": token,
                        "similarity": round(
                            similarity,
                            3,
                        ),
                    }
                )

        if matched_text:

            obtained += (
                text_weight
                * best_text_similarity
            )

            candidate.matched_attributes[
                "text"
            ] = matched_text

        # ==========================================================
        # Symbol
        # ==========================================================

        symbol_weight = self._get_importance(
            candidate,
            "symbol",
            0.20,
        )

        maximum += symbol_weight

        if observed_symbol:

            matched_symbols = []

            for symbol in common_symbols:

                similarity = self._similarity(
                    observed_symbol,
                    symbol,
                )

                if similarity >= 0.70:

                    matched_symbols.append(
                        {
                            "ontology": symbol,
                            "similarity": round(
                                similarity,
                                3,
                            ),
                        }
                    )

            if matched_symbols:

                best_similarity = max(
                    item["similarity"]
                    for item in matched_symbols
                )

                obtained += (
                    symbol_weight
                    * best_similarity
                )

                candidate.matched_attributes[
                    "symbols"
                ] = matched_symbols

        # ==========================================================
        # Language / writing-system evidence
        # ==========================================================

        language_weight = self._get_importance(
            candidate,
            "language",
            0.15,
        )

        maximum += language_weight

        if languages:

            matched_languages = []

            for language in languages:

                language_normalized = self._normalize(
                    language
                )

                if (
                    language_normalized
                    and language_normalized in description
                ):
                    matched_languages.append(
                        language_normalized
                    )

            if matched_languages:

                coverage = (
                    len(matched_languages)
                    /
                    len(languages)
                )

                obtained += (
                    language_weight
                    * coverage
                )

                candidate.matched_attributes[
                    "languages"
                ] = matched_languages

        # ==========================================================
        # Generic textual characteristics
        # ==========================================================

        textual_terms = (
            common_text
            + common_symbols
            + languages
            + semantic_terms
        )

        textual_terms = list(
            dict.fromkeys(textual_terms)
        )

        if textual_terms:

            characteristic_weight = self._get_importance(
                candidate,
                "textual_characteristics",
                0.15,
            )

            maximum += characteristic_weight

            matches = []

            for term in textual_terms:

                if (
                    term
                    and term in description
                ):

                    matches.append(term)

            if matches:

                coverage = (
                    len(matches)
                    /
                    max(len(textual_terms), 1)
                )

                obtained += (
                    characteristic_weight
                    * min(
                        coverage * 2.0,
                        1.0,
                    )
                )

                candidate.matched_attributes[
                    "textual_characteristics"
                ] = matches

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

            word_normalized = self._normalize(word)

            if observed == word_normalized:

                matched.append(
                    {
                        "word": word,
                        "similarity": 1.0,
                    }
                )

                best_similarity = 1.0

            elif (
                observed
                and word_normalized
                and observed in word_normalized
            ):

                # Compound semantic term. Keep this weak.
                similarity = 0.35

                matched.append(
                    {
                        "word": word,
                        "similarity": similarity,
                    }
                )

                best_similarity = max(
                    best_similarity,
                    similarity,
                )

            elif word_normalized in description:

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

    def _score_discriminative_bonus(
        self,
        query: RetrievalQuery,
        candidate: CandidateMatch,
    ) -> float:
        """Add a bounded direct bonus for explicit discriminative cues.

        This supplements, rather than replaces, the existing evidence
        fusion. The normalized attribute score can dilute a decisive
        observation because class-specific cues are only one evidence
        source among many.

        Crucially, discriminative cues are derived from the candidate class
        name, not from generic semantic metadata. A class such as
        ``traffic_light_3_phase`` may mention red/green/yellow in its
        metadata, but those colors do not make it a color-specific class.

        A small direct bonus allows an explicitly observed state such as
        "red lens visible" to influence the final ranking without
        overwhelming the existing embedding and attribute evidence.

        Returns
        -------
        float
            Direct bounded bonus in the range [0.0, 0.15].
        """

        class_name = self._normalize(
            candidate.entry.class_name,
        )

        observed_evidence = " ".join(
            part
            for part in [
                self._normalize(query.observed_object),
                self._normalize(query.description),
                self._normalize(query.text),
                self._normalize(query.symbol),
                *[
                    self._normalize(feature)
                    for feature in query.distinguishing_features
                ],
            ]
            if part
        )

        if not observed_evidence:
            return 0.0

        observed_tokens = set(
            re.findall(
                r"[a-z0-9]+",
                observed_evidence,
            )
        )

        class_tokens = set(
            re.findall(
                r"[a-z0-9]+",
                class_name,
            )
        )

        # Generic parent terms are not discriminative because they occur
        # across many related ontology classes.
        generic_tokens = {
            "traffic",
            "light",
            "signal",
            "sign",
            "road",
            "marking",
            "vehicle",
            "object",
            "phase",
            "class",
        }

        class_cues = {
            token
            for token in class_tokens
            if token not in generic_tokens
            and len(token) > 1
        }

        # IMPORTANT:
        # Do not derive discriminative class-selection cues from ontology
        # semantic metadata. Metadata often describes properties shared by
        # the whole class family. For example, a 3-phase traffic light may
        # legitimately list "red", "yellow", and "green" in its metadata,
        # but those colors do NOT define the 3-phase class.
        #
        # The direct bonus therefore comes only from the ontology class name.
        # Generic descriptive knowledge remains available to the normal
        # attribute/evidence scorers above.
        if not class_cues:
            return 0.0

        matched = sorted(
            class_cues & observed_tokens,
        )

        if not matched:
            return 0.0

        # Keep the direct bonus limited to cues that commonly distinguish
        # sibling ontology classes. Generic physical descriptors such as
        # "head", "lens", "circular", and "mounted" must not become strong
        # class-selection signals merely because an ontology entry mentions
        # them.
        discriminative_tokens = {
            "red",
            "green",
            "yellow",
            "amber",
            "orange",
            "blue",
            "white",
            "black",
            "left",
            "right",
            "straight",
            "turn",
            "arrow",
            "bicycle",
            "pedestrian",
            "bus",
            "truck",
            "car",
            "van",
            "motorcycle",
            "stop",
            "yield",
            "priority",
            "parking",
            "prohibition",
            "warning",
            "mandatory",
        }

        strong_matches = [
            token
            for token in matched
            if token in discriminative_tokens
        ]

        if not strong_matches:
            return 0.0

        # One explicit discriminative cue is sufficient for the full
        # bounded bonus. Multiple matching cues do not stack without limit.
        bonus = 0.15

        candidate.matched_attributes[
            "discriminative_bonus"
        ] = {
            "matched": strong_matches,
            "bonus": bonus,
        }

        return bonus

    def _score_class_specific_cues(
        self,
        query: RetrievalQuery,
        candidate: CandidateMatch,
    ) -> tuple[float, float]:
        """Score explicit discriminative cues stated in the observation.

        The original ranker gives most of its weight to the embedding and
        only weakly uses modifiers that distinguish closely related ontology
        classes. This is a problem for classes such as ``traffic_light_red``,
        ``traffic_light_green``, ``traffic_light_yellow`` and ``traffic_light_off``:
        the scene description may explicitly state the decisive cue even when
        the structured visual attribute is ``unknown``.

        This scorer therefore compares discriminative tokens from the
        ontology class name and semantic metadata against the complete
        observed evidence, including the Scene Understanding description.

        A cue is rewarded strongly only when it is explicitly present in the
        observation. Generic parent tokens such as ``traffic``, ``light`` or
        ``signal`` are ignored so that the scorer does not reward every
        traffic-light candidate equally.

        Returns
        -------
        tuple[float, float]
            Obtained evidence and maximum possible evidence.
        """

        class_name = self._normalize(
            candidate.entry.class_name,
        )

        semantic = candidate.entry.data.get(
            "semantic_information",
            {},
        )

        semantic_terms = [
            self._normalize(term)
            for term in (
                semantic.get("keywords", [])
                + semantic.get("synonyms", [])
                + semantic.get("aliases", [])
            )
            if term
        ]

        # The complete description is important here. In the failing case,
        # "red lens visible" exists in the description while query.primary_color
        # is "unknown", so relying only on structured visual attributes loses
        # the strongest available evidence.
        observed_evidence = " ".join(
            part
            for part in [
                self._normalize(query.observed_object),
                self._normalize(query.description),
                self._normalize(query.text),
                self._normalize(query.symbol),
                *[
                    self._normalize(feature)
                    for feature in query.distinguishing_features
                ],
            ]
            if part
        )

        if not observed_evidence:
            return 0.0, 0.0

        observed_tokens = set(
            re.findall(
                r"[a-z0-9]+",
                observed_evidence,
            )
        )

        class_tokens = set(
            re.findall(
                r"[a-z0-9]+",
                class_name,
            )
        )

        # Generic ontology tokens carry little discriminative information.
        # They are already handled by class identity / semantic evidence.
        generic_tokens = {
            "traffic",
            "light",
            "signal",
            "sign",
            "road",
            "marking",
            "vehicle",
            "object",
            "phase",
            "class",
        }

        class_cues = {
            token
            for token in class_tokens
            if token not in generic_tokens
            and len(token) > 1
        }

        # Semantic metadata can contain the same decisive cue even when it is
        # not literally present in the class name.
        for term in semantic_terms:
            term_tokens = set(
                re.findall(
                    r"[a-z0-9]+",
                    term,
                )
            )

            for token in term_tokens:
                if (
                    token not in generic_tokens
                    and len(token) > 1
                ):
                    class_cues.add(token)

        if not class_cues:
            return 0.0, 0.0

        # One strong cue should be enough to materially change the ranking.
        # This is deliberately larger than the old visual-attribute signal,
        # because an explicit textual state such as "red lens" is decisive.
        weight = self._get_importance(
            candidate,
            "class_specific_cues",
            1.0,
        )

        matched = sorted(
            class_cues & observed_tokens,
        )

        # Avoid rewarding generic words accidentally introduced by semantic
        # metadata. A cue must also be present as a meaningful token in the
        # observation.
        if not matched:
            return 0.0, weight

        coverage = min(
            len(matched) / max(len(class_cues), 1),
            1.0,
        )

        candidate.matched_attributes[
            "class_specific_cues"
        ] = {
            "matched": matched,
            "coverage": round(
                coverage,
                3,
            ),
        }

        return (
            weight * coverage,
            weight,
        )

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

    def _score_observed_evidence(
        self,
        query: RetrievalQuery,
        candidate: CandidateMatch,
    ) -> tuple[float, float]:
        """
        Compare the complete observed scene evidence against the
        ontology's descriptive knowledge.

        This is intentionally generic.

        It allows the ontology to contribute evidence from:
            - description
            - inference cues
            - distinctive features
            - semantic information
            - text information

        No ontology class names are hardcoded here.
        """

        data = candidate.entry.data

        description = self._normalize(
            query.description
        )

        observed_evidence = " ".join(
            part
            for part in [
                description,
                self._normalize(query.observed_object),
                self._normalize(query.text),
                self._normalize(query.symbol),
                *[
                    self._normalize(feature)
                    for feature
                    in query.distinguishing_features
                ],
            ]
            if part
        )

        obtained = 0.0
        maximum = 0.0

        # ----------------------------------------------------------
        # Ontology evidence sources
        # ----------------------------------------------------------

        evidence_sources = []

        ontology_description = data.get(
            "description",
            "",
        )

        if ontology_description:
            evidence_sources.append(
                (
                    "ontology_description",
                    ontology_description,
                    0.20,
                )
            )

        inference = data.get(
            "inference_cues",
            {},
        )

        for cue in inference.get(
            "likely_descriptions",
            [],
        ):

            evidence_sources.append(
                (
                    "inference_cue",
                    cue,
                    0.25,
                )
            )

        distinctive = data.get(
            "distinctive_features",
            [],
        )

        for feature in distinctive:

            evidence_sources.append(
                (
                    "distinctive_feature",
                    feature,
                    0.30,
                )
            )

        semantic = data.get(
            "semantic_information",
            {},
        )

        for term in (
            semantic.get("keywords", [])
            +
            semantic.get("synonyms", [])
            +
            semantic.get("aliases", [])
        ):

            evidence_sources.append(
                (
                    "semantic_term",
                    term,
                    0.15,
                )
            )

        if not evidence_sources:
            return 0.0, 0.0

        # ----------------------------------------------------------
        # Evidence matching
        # ----------------------------------------------------------

        for source_type, evidence, weight in evidence_sources:

            evidence = self._normalize(
                evidence
            )

            if not evidence:
                continue

            maximum += weight

            # Exact phrase occurrence is strong evidence.
            if evidence in observed_evidence:

                obtained += weight

                candidate.matched_attributes.setdefault(
                    "observed_evidence",
                    [],
                ).append(
                    {
                        "type": source_type,
                        "evidence": evidence,
                        "similarity": 1.0,
                    }
                )

                continue

            # Otherwise use token overlap rather than comparing
            # two complete sentences with SequenceMatcher.
            evidence_tokens = set(
                re.findall(
                    r"[a-z0-9]+",
                    evidence,
                )
            )

            observed_tokens = set(
                re.findall(
                    r"[a-z0-9]+",
                    observed_evidence,
                )
            )

            if not evidence_tokens:
                continue

            overlap = (
                len(
                    evidence_tokens
                    & observed_tokens
                )
                /
                len(evidence_tokens)
            )

            if overlap >= 0.60:

                obtained += (
                    weight
                    * min(overlap, 1.0)
                )

                candidate.matched_attributes.setdefault(
                    "observed_evidence",
                    [],
                ).append(
                    {
                        "type": source_type,
                        "evidence": evidence,
                        "similarity": round(
                            overlap,
                            3,
                        ),
                    }
                )

        return obtained, maximum

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
            candidate.entry.data.get(
                "distinctive_features",
                [],
            ),
        )

        weight = self._get_importance(
            candidate,
            "distinctive_features",
            0.30,
        )

        maximum += weight

        observed_text = self._normalize(
            " ".join(
                [
                    query.description,
                    *query.distinguishing_features,
                    query.text or "",
                    query.symbol or "",
                ]
            )
        )

        matched = []

        for feature in features:

            feature_tokens = set(
                re.findall(
                    r"[a-z0-9]+",
                    feature,
                )
            )

            observed_tokens = set(
                re.findall(
                    r"[a-z0-9]+",
                    observed_text,
                )
            )

            if not feature_tokens:
                continue

            overlap = (
                len(
                    feature_tokens
                    &
                    observed_tokens
                )
                /
                len(feature_tokens)
            )

            if overlap >= 0.50:

                matched.append(
                    {
                        "feature": feature,
                        "similarity": round(
                            overlap,
                            3,
                        ),
                    }
                )

        if matched:

            best_similarity = max(
                item["similarity"]
                for item in matched
            )

            # Reward strong feature matches without requiring
            # every ontology feature to be visible.
            obtained += (
                weight
                * best_similarity
            )

            candidate.matched_attributes[
                "distinctive_features"
            ] = matched

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