"""Constants and normalization helpers for the upstream Anticipation MIDI vocabulary."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


AMT_GPT2_BOS_ID = 55_026
LLAMA_VOCAB_SIZE = 128_256
NATIVE_MIDI_BOS_MODEL_TOKEN_ID = LLAMA_VOCAB_SIZE + AMT_GPT2_BOS_ID
ALLOWED_NATIVE_MODEL_TOKEN_IDS = range(LLAMA_VOCAB_SIZE, NATIVE_MIDI_BOS_MODEL_TOKEN_ID)
NATIVE_TIME_TOKEN_MIN = 0
NATIVE_TIME_TOKEN_MAX_EXCLUSIVE = 10_000
NATIVE_DURATION_TOKEN_MIN = 10_000
NATIVE_DURATION_TOKEN_MAX_EXCLUSIVE = 11_000
NATIVE_NOTE_TOKEN_MIN = 11_000
NATIVE_NOTE_TOKEN_MAX_EXCLUSIVE = 27_512


def native_event_tokens_to_model_tokens(event_tokens: Iterable[int], *, include_bos: bool = True) -> List[int]:
    """Shift Anticipation event IDs into the MIDI-LLM extended vocabulary."""

    normalized = [int(token_id) for token_id in event_tokens]
    validate_native_event_tokens(normalized)
    prefix = [NATIVE_MIDI_BOS_MODEL_TOKEN_ID] if include_bos else []
    return prefix + [LLAMA_VOCAB_SIZE + token_id for token_id in normalized]


def validate_native_event_tokens(event_tokens: Iterable[int], *, allow_absolute_time_overflow: bool = False) -> None:
    """Validate Anticipation time, duration, and note token triples."""

    tokens = [int(token_id) for token_id in event_tokens]
    if len(tokens) % 3:
        raise ValueError("Native Anticipation event sequence must contain complete triples")
    max_time = None if allow_absolute_time_overflow else NATIVE_TIME_TOKEN_MAX_EXCLUSIVE
    if any(token_id < NATIVE_TIME_TOKEN_MIN or (max_time is not None and token_id >= max_time) for token_id in tokens[0::3]):
        raise ValueError("Native Anticipation time token falls outside the upstream vocabulary")
    if any(not NATIVE_DURATION_TOKEN_MIN <= token_id < NATIVE_DURATION_TOKEN_MAX_EXCLUSIVE for token_id in tokens[1::3]):
        raise ValueError("Native Anticipation duration token falls outside the upstream vocabulary")
    if any(not NATIVE_NOTE_TOKEN_MIN <= token_id < NATIVE_NOTE_TOKEN_MAX_EXCLUSIVE for token_id in tokens[2::3]):
        raise ValueError("Native Anticipation note token falls outside the upstream vocabulary")


def segment_native_event_tokens(event_tokens: Iterable[int]) -> List[Dict[str, Any]]:
    """Split a complete source stream into vocabulary-safe 100-second windows."""

    tokens = [int(token_id) for token_id in event_tokens]
    validate_native_event_tokens(tokens, allow_absolute_time_overflow=True)
    segments: List[Dict[str, Any]] = []
    for index in range(0, len(tokens), 3):
        time_token, duration_token, note_token = tokens[index : index + 3]
        start = time_token // NATIVE_TIME_TOKEN_MAX_EXCLUSIVE * NATIVE_TIME_TOKEN_MAX_EXCLUSIVE
        if not segments or segments[-1]["start_time_token"] != start:
            segments.append({"start_time_token": start, "event_tokens": []})
        segments[-1]["event_tokens"].extend([time_token - start, duration_token, note_token])
    for segment in segments:
        validate_native_event_tokens(segment["event_tokens"])
    return segments


def rebase_native_event_tokens(event_tokens: Iterable[int]) -> List[int]:
    """Move a native event stream to time zero without changing durations or notes."""

    tokens = [int(token_id) for token_id in event_tokens]
    validate_native_event_tokens(tokens, allow_absolute_time_overflow=True)
    if not tokens:
        return []
    start = min(tokens[0::3])
    rebased = [
        token_id - start if index % 3 == 0 else token_id
        for index, token_id in enumerate(tokens)
    ]
    validate_native_event_tokens(rebased)
    return rebased


def tail_native_event_tokens(event_tokens: Iterable[int], seconds: float) -> List[int]:
    """Keep and rebase a bounded tail for native continuation conditioning."""

    if seconds <= 0:
        raise ValueError("seconds must be positive")
    tokens = [int(token_id) for token_id in event_tokens]
    validate_native_event_tokens(tokens, allow_absolute_time_overflow=True)
    if not tokens:
        return []
    final_time = max(tokens[0::3])
    start = max(0, final_time - round(seconds * 100))
    selected = [
        token_id
        for index in range(0, len(tokens), 3)
        if tokens[index] >= start
        for token_id in tokens[index : index + 3]
    ]
    return rebase_native_event_tokens(selected)


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
