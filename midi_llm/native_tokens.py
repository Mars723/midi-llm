"""Constants and normalization helpers for the upstream Anticipation MIDI vocabulary."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


AMT_GPT2_BOS_ID = 55_026
LLAMA_VOCAB_SIZE = 128_256
NATIVE_MIDI_BOS_MODEL_TOKEN_ID = LLAMA_VOCAB_SIZE + AMT_GPT2_BOS_ID
ALLOWED_NATIVE_MODEL_TOKEN_IDS = range(LLAMA_VOCAB_SIZE, NATIVE_MIDI_BOS_MODEL_TOKEN_ID)


def normalize_native_model_tokens(model_token_ids: Iterable[int]) -> Tuple[List[int], Dict[str, Any]]:
    """Shift generated model tokens into Anticipation event IDs and retain complete triples.

    The upstream MIDI representation is emitted as time, duration, and note triples.
    Generation may end with EOS or another non-MIDI token. That suffix is not part of
    the MIDI event stream and must not be passed to Anticipation.
    """

    event_tokens: List[int] = []
    stop_model_token_id = None
    for raw_token_id in model_token_ids:
        token_id = int(raw_token_id)
        event_token_id = token_id - LLAMA_VOCAB_SIZE
        if not 0 <= event_token_id < AMT_GPT2_BOS_ID:
            stop_model_token_id = token_id
            break
        event_tokens.append(event_token_id)
    incomplete_tokens = len(event_tokens) % 3
    if incomplete_tokens:
        event_tokens = event_tokens[:-incomplete_tokens]
    return event_tokens, {
        "accepted_event_tokens": len(event_tokens),
        "dropped_incomplete_event_tokens": incomplete_tokens,
        "stop_model_token_id": stop_model_token_id,
    }
