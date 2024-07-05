import ssl
import numpy as np
from collections.abc import Mapping

import nltk
from nltk.tokenize import (
    NLTKWordTokenizer,
    PunktSentenceTokenizer,
)

from lynxius_evals.models.eval_model import EvalModel


class BertScoreEval:
    def __init__(
        self, model: EvalModel, level: str, presence_threshold: float = 0.65
    ) -> None:
        assert level in [
            "sentence",
            "word",
        ], "Only sentence and word level BertScore is supported."
        self.level = level
        self.presence_threshold = presence_threshold
        self.model = model

        # NLTK tokenizer needs some resources to be downloaded.
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            # This workaround is needed to avoid certificate verification issues when
            # downloading resources.
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context
            nltk.download("punkt", raise_on_error=True)

    def evaluate(self, data: list[Mapping[str, str]]) -> object:
        tokenize_fn = (
            PunktSentenceTokenizer().tokenize
            if self.level == "sentence"
            else NLTKWordTokenizer().tokenize
        )
        span_tokenize_fn = (
            PunktSentenceTokenizer().span_tokenize
            if self.level == "sentence"
            else NLTKWordTokenizer().span_tokenize
        )

        # We batch all tokens to a single embed request
        batched_tokens = []
        # This map stores a list of tuples (ref_len, cnd_len).
        # The length of this list is the batch size.
        batched_token_map = []
        # This list contains tuples (ref_spans, cnd_spans) of lists of spans
        batched_token_spans = []
        for item in data:
            reference_tokens = tokenize_fn(item["reference"])
            candidate_tokens = tokenize_fn(item["output"])
            batched_tokens += reference_tokens
            batched_tokens += candidate_tokens
            batched_token_map.append((len(reference_tokens), len(candidate_tokens)))

            # Add spans
            reference_token_spans = list(span_tokenize_fn(item["reference"]))
            candidate_token_spans = list(span_tokenize_fn(item["output"]))
            batched_token_spans.append((reference_token_spans, candidate_token_spans))

        # Embed all text in a batch
        embeddings = self.model.embed(batched_tokens)

        # This result will be returned
        result = []
        consumed = 0
        for batch, token_map in enumerate(batched_token_map):
            ref_len = token_map[0]
            cnd_len = token_map[1]
            ref_start, ref_end = consumed, consumed + ref_len
            cnd_start, cnd_end = consumed + ref_len, consumed + ref_len + cnd_len
            consumed += ref_len + cnd_len

            ref_vecs = np.array(
                [vec for vec in embeddings[ref_start:ref_end]]
            )  # (ref_len, 1536)
            cnd_vecs = np.array(
                [vec for vec in embeddings[cnd_start:cnd_end]]
            )  # (cnd_len, 1536)

            # Normalize to have unit length (no need for this when using OpenAI)
            ref_vecs = ref_vecs / np.linalg.norm(ref_vecs, axis=1)[:, None]
            cnd_vecs = cnd_vecs / np.linalg.norm(cnd_vecs, axis=1)[:, None]

            # Pairwise similarity is a batch dot product of normalized vectors
            similarity = np.matmul(
                ref_vecs, cnd_vecs.transpose(1, 0)
            )  # (ref_len, cnd_len)

            # In theory, cosine similarity is in [-1, 1]. In practice, negative values
            # rarely occur. Just to be safe our metric is in [0, 1], we clamp it:
            similarity = np.clip(similarity, 0.0, 1.0)

            # Just a reminder:
            # ==========================================================================
            # Precision:
            # What proportion of posit. identifications was actually correct? TP/(TP+FP)
            #
            # Recall:
            # What proportion of actual positives was identified correctly? TP/(TP+FN)
            # ==========================================================================
            recall = np.sum(np.max(similarity, axis=1)) / ref_len
            precision = np.sum(np.max(similarity, axis=0)) / cnd_len
            f1 = 2 * precision * recall / (precision + recall)
            recall, precision, f1

            # Now, lets find what exact tokens are missing in the candidate
            sim_best_ref = np.max(similarity, axis=1)  # (ref_len)
            missing_indices = np.where(sim_best_ref < self.presence_threshold)[0]

            missing_tokens = []
            for idx in missing_indices:
                missing_tokens.append(batched_token_spans[batch][0][idx])

            result.append(
                {
                    "reference": data[batch]["reference"],
                    "output": data[batch]["output"],
                    "contexts": data[batch]["contexts"],
                    "trace_uuid": data[batch]["trace_uuid"],
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "missing_tokens": missing_tokens,
                }
            )

        return result
