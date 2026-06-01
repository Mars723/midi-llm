from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from midi_llm.compiler import find_musescore, render_musescore, write_musicxml, write_performance_midi
from midi_llm.compose import compose
from midi_llm.curriculum import prepare_curriculum
from midi_llm.evaluate import evaluate_score
from midi_llm.fetch_pdmx import (
    _extract_tar,
    _missing_ranges,
    _normalize_existing_parts,
    _part_path,
    _split_range,
    _validate_coverage,
)
from midi_llm.gallery import write_gallery
from midi_llm.generate_checkpoint import (
    _CandidateFileStreamer,
    _ending_fragment_has_tonic,
    _fragment_merge_errors,
    _fragment_quality_score,
    _measure_event_budget_exceeded,
    _normalize_model_note_event,
    _resume_hierarchical_score,
    _rewrite_final_tonic_cadence,
    _sanitize_model_directions,
    _section_expansion_ranges,
    _section_model_input,
    _score_fragment_from_continuation,
    _score_from_continuation,
    _whole_piece_model_input,
    build_parser as build_checkpoint_parser,
)
from midi_llm.import_midi import MidiNote, ParsedMidi, TempoPoint, draft_from_parsed, parse_midi, select_structural_tempos
from midi_llm.materialize_training import materialize_training_dataset
from midi_llm.model_scoredsl import decode_model_score, encode_model_score, model_score_generation_prefix
from midi_llm.musicxml_score import import_musicxml_score
from midi_llm.native_backbone import (
    NativeBackboneConfig,
    generate_native_backbone,
    native_backbone_manifest,
    native_midi_stats,
    upstream_generation_prompt,
)
from midi_llm.native_tokens import (
    LLAMA_VOCAB_SIZE,
    NATIVE_MIDI_BOS_MODEL_TOKEN_ID,
    normalize_native_model_tokens,
)
from midi_llm.notation_analysis import analyze_notation
from midi_llm.package_cloud import extract_cloud_bundle, package_cloud_dataset, verify_cloud_bundle
from midi_llm.planner import ComposeControls, create_piece_plan
from midi_llm.prepare_pdmx import _quality_label, _tasks_for_quality, prepare_manifest
from midi_llm.rules import create_motif_bank, generate_score_candidate, render_performance
from midi_llm.release_gate import run_release_gate
from midi_llm.research_symupe import (
    _candidate_midis,
    _normalize_candidate_name,
    _require_research_license_ack,
)
from midi_llm.scoredsl import decode_score, encode_score
from midi_llm.score_ir import NoteEvent, PianoScoreIR, ScoreDirection, read_score, validate_score, write_json
from midi_llm.training import create_run_plan
from midi_llm.train_scoredsl import TrainConfig, _target_loss_weights, _validate_token_lengths, build_training_spec


class ScoreFirstTest(unittest.TestCase):
    def build_score(self):
        plan = create_piece_plan(
            "A complete lyrical nocturne with a contrasting middle section.",
            ComposeControls(duration_minutes=3.0),
        )
        motifs = create_motif_bank(plan.key, plan.genre)
        score = generate_score_candidate(plan, motifs, seed=23)
        performance = render_performance(score, seed=24)
        return score, performance

    def test_whole_piece_plan_covers_complete_output(self):
        score, _ = self.build_score()
        self.assertGreaterEqual(score.plan.measure_count, 48)
        self.assertLessEqual(score.plan.measure_count, 192)
        self.assertEqual(score.plan.sections[0].start_measure, 1)
        self.assertEqual(score.plan.sections[-1].end_measure, score.plan.measure_count)
        self.assertEqual(validate_score(score), [])

    def test_native_backbone_matches_upstream_prompt_framing(self):
        self.assertEqual(
            upstream_generation_prompt("A solo piano nocturne."),
            "You are a world-class composer. Please compose some music according to the following description: "
            "A solo piano nocturne. ",
        )
        self.assertEqual(NATIVE_MIDI_BOS_MODEL_TOKEN_ID, LLAMA_VOCAB_SIZE + 55_026)

    def test_native_backbone_normalizes_only_complete_anticipation_events(self):
        tokens, report = normalize_native_model_tokens(
            [
                LLAMA_VOCAB_SIZE + 1,
                LLAMA_VOCAB_SIZE + 2,
                LLAMA_VOCAB_SIZE + 3,
                LLAMA_VOCAB_SIZE + 4,
                LLAMA_VOCAB_SIZE + 5,
                128_009,
            ]
        )
        self.assertEqual(tokens, [1, 2, 3])
        self.assertEqual(report["dropped_incomplete_event_tokens"], 2)
        self.assertEqual(report["stop_model_token_id"], 128_009)

    def test_native_backbone_manifest_keeps_native_midi_authoritative(self):
        config = NativeBackboneConfig(prompt="A solo piano nocturne.", output_dir="generated_native_backbone/test")
        manifest = native_backbone_manifest(
            config,
            [{"candidate": 1, "native_midi": "candidate_1/native.mid"}],
            [],
        )
        self.assertEqual(manifest["adapter"], None)
        self.assertTrue(manifest["native_midi_is_authoritative_musical_content"])
        self.assertTrue(manifest["score_conversion_is_draft_only"])

    def test_native_backbone_stats_measure_native_content_without_score_claims(self):
        parsed = ParsedMidi(
            ticks_per_beat=480,
            notes=[
                MidiNote(tick=0, duration=480, pitch=48, velocity=72),
                MidiNote(tick=1920, duration=480, pitch=72, velocity=80),
            ],
            tempos=[TempoPoint(tick=0, bpm=120), TempoPoint(tick=1920, bpm=60)],
            pedal=[],
            meter=ComposeControls().meter,
        )
        self.assertEqual(
            native_midi_stats(parsed),
            {
                "notes": 2,
                "duration_seconds": 3.0,
                "estimated_measures": 2,
                "pitch_range": [48, 72],
                "distinct_pitches": 2,
                "tempo_events": 2,
                "pedal_events": 0,
                "notes_per_second": 0.667,
                "max_notes_at_onset": 1,
                "max_active_notes": 1,
                "potential_density_drift": False,
            },
        )

    def test_native_backbone_keeps_valid_candidate_when_another_native_stream_is_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "native"
            model_tokens = [LLAMA_VOCAB_SIZE + 1, LLAMA_VOCAB_SIZE + 2, LLAMA_VOCAB_SIZE + 3]
            with (
                patch("midi_llm.native_backbone._load_upstream_model", return_value=(None, None, None)),
                patch("midi_llm.native_backbone._sample_native_model_tokens", return_value=model_tokens),
                patch(
                    "midi_llm.native_backbone._write_native_midi",
                    side_effect=[output_dir / "candidate_1" / "native.mid", AssertionError("bad native stream")],
                ),
                patch("midi_llm.native_backbone.parse_midi", return_value=None),
                patch(
                    "midi_llm.native_backbone.native_midi_stats",
                    return_value={"notes": 1, "duration_seconds": 1.0},
                ),
            ):
                generate_native_backbone(
                    NativeBackboneConfig(
                        prompt="A solo piano nocturne.",
                        output_dir=str(output_dir),
                        n_outputs=2,
                        max_tokens=3,
                        score_draft=False,
                    )
                )
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["candidates"]), 1)
            self.assertEqual(manifest["failures"], [{"candidate": 2, "seed": 24, "reason": "bad native stream"}])

    def test_symupe_research_helpers_require_explicit_license_ack(self):
        with self.assertRaisesRegex(SystemExit, "CC-BY-NC-SA-4.0"):
            _require_research_license_ack(False)
        _require_research_license_ack(True)
        self.assertEqual(_normalize_candidate_name("2"), "candidate_2")
        self.assertEqual(_normalize_candidate_name("candidate_4"), "candidate_4")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "candidate_2").mkdir()
            (root / "candidate_2" / "native.mid").touch()
            self.assertEqual(_candidate_midis(root), [root / "candidate_2" / "native.mid"])

    def test_scoredsl_round_trip(self):
        score, _ = self.build_score()
        decoded = decode_score(encode_score(score))
        self.assertEqual(decoded.plan, score.plan)
        self.assertEqual(decoded.motif_bank, score.motif_bank)
        self.assertEqual(decoded.notes, score.notes)
        self.assertEqual(decoded.directions, score.directions)

    def test_model_scoredsl_round_trip_uses_shared_blueprint(self):
        score, _ = self.build_score()
        decoded = decode_model_score(encode_model_score(score), score.plan, score.motif_bank)
        self.assertEqual(decoded.plan, score.plan)
        self.assertEqual(decoded.motif_bank, score.motif_bank)
        self.assertEqual(
            [asdict(note) | {"id": None} for note in decoded.notes],
            [asdict(note) | {"id": None} for note in score.notes],
        )
        self.assertEqual(
            sorted(tuple(asdict(direction).items()) for direction in decoded.directions),
            sorted(tuple(asdict(direction).items()) for direction in score.directions),
        )

    def test_model_generation_prefix_fixes_header_and_opening_measure(self):
        self.assertEqual(
            model_score_generation_prefix(),
            'SCORE ["compact-measure-interleaved-v3"]\nMEASURE [1]\n',
        )

    def test_checkpoint_continuation_uses_requested_whole_piece_plan(self):
        score, _ = self.build_score()
        model_input = _whole_piece_model_input(score.plan, score.motif_bank)
        self.assertEqual(model_input["target_range"], [1, score.plan.measure_count])
        self.assertEqual(model_input["future_ending_target"]["measure"], score.plan.measure_count)
        self.assertEqual(model_input["notation_constraints"]["maximum_identical_measure_run"], 8)
        self.assertEqual(model_input["notation_constraints"]["minimum_lower_staff_measure_coverage"], 0.75)
        decoded = _score_from_continuation(
            encode_model_score(score) + "ignored trailing output",
            score.plan,
            score.motif_bank,
            "training_runs/test/adapter",
        )
        self.assertEqual(decoded.plan, score.plan)
        self.assertEqual(
            [asdict(note) | {"id": None} for note in decoded.notes],
            [asdict(note) | {"id": None} for note in score.notes],
        )
        self.assertEqual(decoded.metadata["score_source"], "trained-scoredsl-adapter")

    def test_checkpoint_continuation_rejects_truncated_piece(self):
        score, _ = self.build_score()
        truncated = "\n".join(
            (
                'SCORE ["compact-measure-interleaved-v3"]',
                "MEASURE [1]",
                "NOTE [1,0.0,1.0,60,1,1,null,null,false,false]",
                "END_SCORE",
            )
        )
        with self.assertRaisesRegex(ValueError, "planned final measure"):
            _score_from_continuation(truncated, score.plan, score.motif_bank, "training_runs/test/adapter")

    def test_checkpoint_continuation_completes_at_planned_measure_boundary(self):
        score, _ = self.build_score()
        continuation = encode_model_score(score).replace(
            "END_SCORE\n",
            f"NOTE [{score.plan.measure_count + 1},0.0,1.0,60,1,1,null,null,false,false]\n",
        )
        decoded = _score_from_continuation(continuation, score.plan, score.motif_bank, "training_runs/test/adapter")
        self.assertEqual(decoded.metadata["decode_completion"], "planned-measure-boundary")
        self.assertEqual(max(note.measure for note in decoded.notes), score.plan.measure_count)

    def test_checkpoint_continuation_skips_only_malformed_optional_rows(self):
        score, _ = self.build_score()
        continuation = encode_model_score(score).replace(
            "MEASURE [1]\n",
            (
                'MEASURE [1]\n'
                'DIRECTION [1,0.0,"words","Alto", slightly,null,null]\n'
                'LAYOUT [1,"system-break","unexpected"]\n'
            ),
            1,
        )
        decoded = _score_from_continuation(continuation, score.plan, score.motif_bank, "training_runs/test/adapter")
        self.assertEqual(decoded.metadata["skipped_malformed_optional_rows"], 2)
        self.assertEqual(len(decoded.notes), len(score.notes))

    def test_checkpoint_note_normalizer_fills_only_deterministic_defaults(self):
        normalized, count = _normalize_model_note_event("NOTE [1,0.0,0.5,48,2,null,null,false,false]")
        self.assertEqual(normalized, "NOTE [1,0.0,0.5,48,2,1,null,null,false,false]")
        self.assertEqual(count, 1)
        normalized, count = _normalize_model_note_event("NOTE [1,0.0,0.5,48,2,1,null,null]")
        self.assertEqual(normalized, "NOTE [1,0.0,0.5,48,2,1,null,null,false,false]")
        self.assertEqual(count, 1)
        unchanged, count = _normalize_model_note_event("NOTE [1,0.0,broken]")
        self.assertEqual(unchanged, "NOTE [1,0.0,broken]")
        self.assertEqual(count, 0)

    def test_checkpoint_continuation_trims_one_empty_terminal_measure(self):
        score, _ = self.build_score()
        planned_measures = score.plan.measure_count
        continuation = "\n".join(
            line
            for line in encode_model_score(score).splitlines()
            if not line.startswith(f"NOTE [{planned_measures},")
        )
        decoded = _score_from_continuation(continuation, score.plan, score.motif_bank, "training_runs/test/adapter")
        self.assertEqual(decoded.metadata["trimmed_empty_trailing_measures"], 1)
        self.assertEqual(decoded.plan.measure_count, planned_measures - 1)
        self.assertEqual(score.plan.measure_count, planned_measures)

    def test_checkpoint_continuation_rejects_measure_loop(self):
        score, _ = self.build_score()
        rows = ['SCORE ["compact-measure-interleaved-v3"]']
        for measure in range(1, score.plan.measure_count + 1):
            rows.append(f"MEASURE [{measure}]")
            rows.append(f"NOTE [{measure},0.0,1.0,60,1,1,null,null,false,false]")
        rows.append("END_SCORE")
        with self.assertRaisesRegex(ValueError, "identical measure content"):
            _score_from_continuation("\n".join(rows), score.plan, score.motif_bank, "training_runs/test/adapter")

    def test_short_checkpoint_fragment_rejects_measure_loop(self):
        score, _ = self.build_score()
        rows = ['SCORE ["compact-measure-interleaved-v3"]']
        for measure in range(1, 9):
            rows.append(f"MEASURE [{measure}]")
            rows.append(f"NOTE [{measure},0.0,1.0,60,1,1,null,null,false,false]")
            rows.append(f"NOTE [{measure},0.0,1.0,48,2,1,null,null,false,false]")
        rows.append("END_SCORE")
        with self.assertRaisesRegex(ValueError, "enough unique measure content"):
            _score_fragment_from_continuation(
                "\n".join(rows),
                score.plan,
                score.motif_bank,
                "training_runs/test/adapter",
                [1, 8],
                allow_terminal_empty=False,
            )

    def test_short_checkpoint_fragment_rejects_invalid_note_fields(self):
        score, _ = self.build_score()
        rows = ['SCORE ["compact-measure-interleaved-v3"]']
        for measure in range(1, 5):
            rows.append(f"MEASURE [{measure}]")
            staff = 45 if measure == 2 else 1
            rows.append(f"NOTE [{measure},0.0,1.0,{60 + measure},{staff},1,null,null,false,false]")
            rows.append(f"NOTE [{measure},0.0,1.0,{48 + measure},2,1,null,null,false,false]")
        rows.append("END_SCORE")
        with self.assertRaisesRegex(ValueError, "invalid staff 45"):
            _score_fragment_from_continuation(
                "\n".join(rows),
                score.plan,
                score.motif_bank,
                "training_runs/test/adapter",
                [1, 4],
                allow_terminal_empty=False,
            )

    def test_checkpoint_fragment_merge_rejects_cross_window_measure_loop(self):
        score, _ = self.build_score()
        partial = PianoScoreIR(
            plan=score.plan,
            motif_bank=score.motif_bank,
            notes=[
                NoteEvent(id=f"upper-{measure}", measure=measure, beat=0.0, duration=1.0, pitch=60, staff=1)
                for measure in range(1, 5)
            ]
            + [
                NoteEvent(id=f"lower-{measure}", measure=measure, beat=0.0, duration=1.0, pitch=48, staff=2)
                for measure in range(1, 5)
            ],
            directions=[],
        )
        fragment = PianoScoreIR(
            plan=score.plan,
            motif_bank=score.motif_bank,
            notes=[
                NoteEvent(id=f"upper-{measure}", measure=measure, beat=0.0, duration=1.0, pitch=60, staff=1)
                for measure in range(5, 9)
            ]
            + [
                NoteEvent(id=f"lower-{measure}", measure=measure, beat=0.0, duration=1.0, pitch=48, staff=2)
                for measure in range(5, 9)
            ],
            directions=[],
        )
        self.assertEqual(
            _fragment_merge_errors(partial, fragment),
            ["Generated fragment extends identical measure content across window boundaries"],
        )

    def test_checkpoint_continuation_rejects_short_periodic_measure_loop(self):
        score, _ = self.build_score()
        rows = ['SCORE ["compact-measure-interleaved-v3"]']
        for measure in range(1, score.plan.measure_count + 1):
            rows.append(f"MEASURE [{measure}]")
            rows.append(f"NOTE [{measure},0.0,1.0,{60 + measure % 2},1,1,null,null,false,false]")
        rows.append("END_SCORE")
        with self.assertRaisesRegex(ValueError, "short measure pattern"):
            _score_from_continuation("\n".join(rows), score.plan, score.motif_bank, "training_runs/test/adapter")

    def test_evaluator_rejects_low_measure_diversity(self):
        score, _ = self.build_score()
        rows = ['SCORE ["compact-measure-interleaved-v3"]']
        for measure in range(1, score.plan.measure_count + 1):
            rows.append(f"MEASURE [{measure}]")
            rows.append(f"NOTE [{measure},0.0,1.0,{60 + measure % 13},1,1,null,null,false,false]")
        rows.append("END_SCORE")
        decoded = _score_from_continuation("\n".join(rows), score.plan, score.motif_bank, "training_runs/test/adapter")
        metrics = evaluate_score(decoded, render_performance(decoded, seed=24))
        self.assertFalse(metrics["measure_diversity_acceptable"])
        self.assertFalse(metrics["valid"])

    def test_checkpoint_sampling_defaults_to_validated_temperature(self):
        args = build_checkpoint_parser().parse_args(("--adapter-dir", "adapter", "--prompt", "prompt"))
        self.assertEqual(args.temperature, 0.8)
        self.assertEqual(args.repetition_penalty, 1.01)
        self.assertEqual(args.strategy, "hierarchical")
        self.assertEqual(args.section_candidates, 2)
        self.assertIsNone(args.min_valid_section_alternatives)
        self.assertEqual(args.max_section_new_tokens, 8192)
        self.assertEqual(args.max_expansion_measures, 16)

    def test_section_model_input_carries_global_plan_and_left_neighbor(self):
        score, _ = self.build_score()
        first, second = score.plan.sections[:2]
        first_input = _section_model_input(score.plan, score.motif_bank, first, score)
        second_input = _section_model_input(score.plan, score.motif_bank, second, score)
        self.assertEqual(first_input["target_range"], [first.start_measure, first.end_measure])
        self.assertIsNone(first_input["left_neighbor_scoredsl"])
        self.assertEqual(second_input["piece_plan"]["measure_count"], score.plan.measure_count)
        self.assertIn("MEASURE", second_input["left_neighbor_scoredsl"])
        self.assertEqual(second_input["active_section"]["label"], second.label)

    def test_long_sections_split_into_bounded_expansion_ranges(self):
        score, _ = self.build_score()
        section = score.plan.sections[0]
        self.assertEqual(
            _section_expansion_ranges(section, 16),
            [[section.start_measure, section.start_measure + 15], [section.start_measure + 16, section.end_measure]],
        )

    def test_hierarchical_resume_reuses_completed_section_prefix(self):
        score, _ = self.build_score()
        first = score.plan.sections[0]
        partial = PianoScoreIR(
            plan=score.plan,
            motif_bank=score.motif_bank,
            notes=[note for note in score.notes if note.measure <= first.end_measure],
            directions=[direction for direction in score.directions if direction.measure <= first.end_measure],
            layout_hints=[hint for hint in score.layout_hints if hint.measure <= first.end_measure],
            metadata={"completed_section_labels": [first.label]},
        )
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "partial.score.ir.json"
            write_json(path, partial)
            resumed = _resume_hierarchical_score(str(path), score.plan, score.motif_bank)
        self.assertEqual(resumed.metadata["completed_section_labels"], [first.label])
        self.assertEqual(resumed.notes, partial.notes)

    def test_hierarchical_generation_starts_from_an_empty_score(self):
        score, _ = self.build_score()
        initial = _resume_hierarchical_score(None, score.plan, score.motif_bank)
        self.assertEqual(initial.notes, [])
        self.assertEqual(initial.directions, [])

    def test_ending_fragment_requires_planned_tonic(self):
        score, _ = self.build_score()
        self.assertTrue(_ending_fragment_has_tonic(score))
        final_measure = max(note.measure for note in score.notes)
        for note in score.notes:
            if note.measure == final_measure and note.staff == 1:
                note.pitch += 1
        self.assertFalse(_ending_fragment_has_tonic(score))

    def test_cadence_repair_rewrites_only_final_measure(self):
        score, _ = self.build_score()
        score.plan.key = "D minor"
        final_measure = max(note.measure for note in score.notes)
        earlier_notes = [note for note in score.notes if note.measure != final_measure]
        _rewrite_final_tonic_cadence(score)
        self.assertEqual(
            [note for note in score.notes if note.measure != final_measure],
            earlier_notes,
        )
        self.assertTrue(_ending_fragment_has_tonic(score))
        final_pitch_classes = {note.pitch % 12 for note in score.notes if note.measure == final_measure}
        self.assertEqual(final_pitch_classes, {2, 5, 9})
        self.assertEqual(score.metadata["cadence_repair"]["measure"], final_measure)

    def test_fragment_quality_rewards_requested_two_staff_texture(self):
        score, _ = self.build_score()
        one_staff = PianoScoreIR(
            plan=score.plan,
            motif_bank=score.motif_bank,
            notes=[note for note in score.notes if note.staff == 1],
            directions=[],
        )
        self.assertGreater(_fragment_quality_score(score), _fragment_quality_score(one_staff))

    def test_evaluator_rejects_missing_accompaniment_staff(self):
        score, performance = self.build_score()
        score.notes = [note for note in score.notes if note.staff == 1]
        metrics = evaluate_score(score, performance)
        self.assertEqual(metrics["lower_staff_measure_coverage"], 0.0)
        self.assertFalse(metrics["two_staff_texture_acceptable"])
        self.assertFalse(metrics["valid"])

    def test_model_direction_sanitizer_drops_conflicting_tempo_words(self):
        score, _ = self.build_score()
        score.directions = [
            ScoreDirection(measure=1, beat=0.0, kind="words", value="Allegro"),
            ScoreDirection(measure=1, beat=0.0, kind="words", value="dolce"),
            ScoreDirection(measure=1, beat=0.0, kind="tempo", value="120|Allegro"),
            ScoreDirection(measure=1, beat=0.0, kind="dynamic", value="p"),
        ]
        self.assertEqual(_sanitize_model_directions(score), 2)
        self.assertEqual([(direction.kind, direction.value) for direction in score.directions], [("words", "dolce"), ("dynamic", "p")])

    def test_checkpoint_streamer_skips_prompt_and_persists_incremental_tokens(self):
        class FakeTensor:
            def __init__(self, values):
                self.values = values

            def detach(self):
                return self

            def cpu(self):
                return self

            def reshape(self, *_shape):
                return self

            def tolist(self):
                return self.values

        class FakeTokenizer:
            def decode(self, values, skip_special_tokens=False):
                return "".join(str(value) for value in values)

        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "candidate.raw.dsl"
            streamer = _CandidateFileStreamer(FakeTokenizer(), path, report_every=2)
            streamer.put(FakeTensor([90, 91]))
            streamer.put(FakeTensor([1]))
            self.assertFalse(path.exists())
            streamer.put(FakeTensor([2]))
            self.assertEqual(path.read_text(encoding="utf-8"), "12")
            streamer.put(FakeTensor([3]))
            streamer.end()
            self.assertEqual(path.read_text(encoding="utf-8"), "123")

    def test_checkpoint_stops_measure_event_overflow(self):
        rows = [f"NOTE [1,{index}.0,0.5,60,1,1,null,null,false,false]" for index in range(4)]
        self.assertFalse(_measure_event_budget_exceeded("\n".join(rows), max_event_rows=4))
        self.assertTrue(_measure_event_budget_exceeded("\n".join(rows + [rows[0]]), max_event_rows=4))

    def test_performance_tempo_curve_does_not_leak_into_musicxml(self):
        score, performance = self.build_score()
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "score.musicxml"
            write_musicxml(score, path)
            root = ET.parse(path).getroot()
            self.assertEqual(len(root.findall(".//sound[@tempo]")), len(score.plan.tempo_marks))
            self.assertGreater(len(performance.tempo_curve), len(score.plan.tempo_marks))
            metrics = evaluate_score(score, performance, path)
            self.assertTrue(metrics["tempo_overlay_isolated"])

    def test_evaluation_reports_measure_repetition(self):
        score, performance = self.build_score()
        metrics = evaluate_score(score, performance)
        self.assertGreater(metrics["unique_measure_signatures"], 0)
        self.assertGreater(metrics["unique_measure_signature_ratio"], 0)
        self.assertGreater(metrics["longest_identical_measure_run"], 0)
        self.assertGreaterEqual(
            metrics["repetition_score_penalty"],
            min(15, max(0, metrics["longest_identical_measure_run"] - 8)),
        )
        self.assertIn("periodic_measure_loop_span", metrics)

    def test_evaluation_invalidates_short_periodic_measure_loop(self):
        score, _ = self.build_score()
        score.notes = [
            NoteEvent(id=f"loop-{measure}", measure=measure, beat=0.0, duration=1.0, pitch=60 + measure % 2, staff=1)
            for measure in range(1, score.plan.measure_count + 1)
        ]
        metrics = evaluate_score(score, render_performance(score, seed=24))
        self.assertFalse(metrics["periodic_loop_acceptable"])
        self.assertFalse(metrics["valid"])

    def test_musicxml_compiler_restores_blueprint_tempos_without_model_directions(self):
        score, performance = self.build_score()
        score.directions = []
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "score.musicxml"
            write_musicxml(score, path)
            root = ET.parse(path).getroot()
            self.assertEqual(len(root.findall(".//sound[@tempo]")), len(score.plan.tempo_marks))
            self.assertTrue(evaluate_score(score, performance, path)["tempo_overlay_isolated"])

    def test_musicxml_compiler_filters_model_tempos_outside_blueprint(self):
        score, performance = self.build_score()
        score.directions.append(ScoreDirection(measure=2, beat=0.0, kind="tempo", value="200|microtempo"))
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "score.musicxml"
            write_musicxml(score, path)
            root = ET.parse(path).getroot()
            self.assertEqual(len(root.findall(".//sound[@tempo]")), len(score.plan.tempo_marks))
            self.assertTrue(evaluate_score(score, performance, path)["tempo_overlay_isolated"])

    def test_musicxml_compiler_separates_multiple_voices_on_one_staff(self):
        score, _ = self.build_score()
        score.notes = [
            NoteEvent(id="right", measure=1, beat=0.0, duration=4.0, pitch=60, staff=1, voice=1),
            NoteEvent(id="left-a", measure=1, beat=0.0, duration=4.0, pitch=48, staff=2, voice=5),
            NoteEvent(id="left-b", measure=1, beat=0.0, duration=4.0, pitch=55, staff=2, voice=6),
        ]
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "score.musicxml"
            write_musicxml(score, path)
            measure = ET.parse(path).getroot().find(".//measure[@number='1']")
            self.assertEqual(len(measure.findall("./backup")), 2)
            voice_six = next(note for note in measure.findall("./note") if note.findtext("./voice") == "6")
            self.assertIsNone(voice_six.find("./chord"))

    def test_external_midi_import_keeps_micro_tempos_in_overlay(self):
        score, performance = self.build_score()
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir)
            midi = path / "performance.mid"
            write_performance_midi(score, performance, midi)
            draft, imported_performance = draft_from_parsed(parse_midi(midi), "Imported")
            self.assertGreater(len(imported_performance.tempo_curve), len(draft.plan.tempo_marks))
            self.assertEqual(len(draft.plan.tempo_marks), 1)
            self.assertTrue(draft.metadata["draft"])

    def test_structural_tempo_filter_promotes_stable_platforms(self):
        from midi_llm.import_midi import TempoPoint

        tempos = [
            TempoPoint(0, 100),
            TempoPoint(480, 98),
            TempoPoint(960, 101),
            TempoPoint(3840, 80),
        ]
        filtered = select_structural_tempos(tempos, 480, 4, final_tick=8000)
        self.assertEqual([round(point.bpm) for point in filtered], [100, 80])

    def test_compose_writes_complete_gallery(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            output = Path(raw_dir) / "run"
            args = argparse.Namespace(
                prompt="A complete prelude with a clear return.",
                controls=None,
                output_dir=str(output),
                genre="prelude",
                form="ABA",
                duration_minutes=3.0,
                measure_range="48-96",
                key="C major",
                meter="4/4",
                tempo=84,
                difficulty="intermediate",
                markings="dynamics,pedal,articulations",
                title=None,
                whole_piece=True,
                strategy="hierarchical",
                candidates=2,
                seed=23,
                skip_musescore=True,
                musescore_bin=None,
            )
            compose(args)
            for filename in (
                "score.ir.json",
                "score.dsl",
                "score.musicxml",
                "score.mid",
                "performance.ir.json",
                "performance.mid",
                "manifest.json",
                "gallery.html",
            ):
                self.assertTrue((output / filename).exists(), filename)
            self.assertIn("Section Timeline", (output / "gallery.html").read_text(encoding="utf-8"))
            self.assertIn("max repeated-measure run", (output / "gallery.html").read_text(encoding="utf-8"))
            self.assertIn("periodic loop span", (output / "gallery.html").read_text(encoding="utf-8"))

    def test_pdmx_manifest_and_cloud_run_plan(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            metadata = root / "pdmx.csv"
            with metadata.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("composer", "title", "path", "instrumentation", "all_valid", "no_license_conflict"),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "composer": "Composer",
                        "title": "Solo",
                        "path": "solo.mxl",
                        "instrumentation": "piano",
                        "all_valid": "1",
                        "no_license_conflict": "1",
                    }
                )
                writer.writerow(
                    {
                        "composer": "Composer",
                        "title": "Duo",
                        "path": "duo.mxl",
                        "instrumentation": "piano violin",
                        "all_valid": "1",
                        "no_license_conflict": "1",
                    }
                )
                writer.writerow(
                    {
                        "composer": "Composer",
                        "title": "Solo",
                        "path": "duplicate-solo.mxl",
                        "instrumentation": "piano",
                        "all_valid": "1",
                        "no_license_conflict": "1",
                    }
                )
            summary = prepare_manifest(metadata, root / "manifest")
            self.assertEqual(summary["accepted_unique_solo_piano_works"], 1)
            manifest = root / "manifest" / "pdmx_score_first_manifest.jsonl"
            run_plan = create_run_plan(manifest, root / "run_plan.json")
            self.assertEqual(
                run_plan["representation"],
                "ModelScoreDSL compact measure-interleaved v3 with rich ScoreDSL artifacts",
            )
            self.assertIn("whole-piece-generate", [phase["name"] for phase in run_plan["phases"]])

    def test_official_pdmx_manifest_labels_intermediate_solo_piano(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            metadata = root / "PDMX.csv"
            fieldnames = (
                "mxl",
                "title",
                "composer_name",
                "tracks",
                "subset:all_valid",
                "subset:no_license_conflict",
                "is_best_unique_arrangement",
                "best_unique_arrangement",
                "complexity",
                "notes_per_bar",
                "n_notes",
                "song_length.bars",
                "genres",
                "tags",
            )
            with metadata.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for title, complexity, density, bars in (
                    ("Simple Waltz", 0, 2, 64),
                    ("Concert Etude", 1, 8, 80),
                    ("Night Nocturne", 2, 14, 96),
                    ("Grand Prelude", 3, 24, 120),
                ):
                    writer.writerow(
                        {
                            "mxl": f"./mxl/{title}.mxl",
                            "title": title,
                            "composer_name": "Composer",
                            "tracks": "0",
                            "subset:all_valid": "True",
                            "subset:no_license_conflict": "True",
                            "is_best_unique_arrangement": "True",
                            "best_unique_arrangement": f"./data/{title}.json",
                            "complexity": complexity,
                            "notes_per_bar": density,
                            "n_notes": density * bars,
                            "song_length.bars": bars,
                            "genres": "classical",
                            "tags": "piano",
                        }
                    )
                writer.writerow(
                    {
                        "mxl": "./mxl/Duet Waltz.mxl",
                        "title": "Duet Waltz",
                        "composer_name": "Composer",
                        "tracks": "0-0",
                        "subset:all_valid": "True",
                        "subset:no_license_conflict": "True",
                        "is_best_unique_arrangement": "True",
                        "best_unique_arrangement": "./data/Duet Waltz.json",
                        "complexity": 1,
                        "notes_per_bar": 8,
                        "n_notes": 640,
                        "song_length.bars": 80,
                        "genres": "classical",
                        "tags": "piano duet",
                    }
                )
            summary = prepare_manifest(
                metadata,
                root / "manifest",
                difficulty="intermediate",
                quality="metadata-curated",
                min_measures=48,
                max_measures=192,
            )
            self.assertEqual(summary["eligible_deduplicated_no_license_conflict_solo_piano_works"], 4)
            self.assertEqual(summary["accepted_unique_solo_piano_works"], 2)
            manifest = root / "manifest" / "pdmx_score_first_manifest.jsonl"
            rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
            self.assertEqual({row["genre"] for row in rows}, {"etude", "nocturne"})
            self.assertEqual({row["form"] for row in rows}, {"free-sectional", "ABA"})
            self.assertTrue(all(row["difficulty"] == "intermediate" for row in rows))
            self.assertTrue(all(row["quality_tier"] == "metadata-curated" for row in rows))
            self.assertTrue(all(row["path"].startswith("./mxl/") for row in rows))
            self.assertTrue(all(row["genre_evidence"]["field"] == "title" for row in rows))
            self.assertTrue(all("whole-piece-generate" not in row["tasks"] for row in rows))

    def test_pdmx_canonical_core_rejects_explicit_arrangements(self):
        row = {
            "title": "Minuet in D major",
            "composer_name": "W. A. Mozart",
            "genres": "classical",
            "is_user_pro": "True",
        }
        tier, _, evidence = _quality_label(row, "minuet")
        self.assertEqual(tier, "canonical-core")
        self.assertTrue(evidence["canonical_classical_composer"])
        self.assertIn("whole-piece-generate", _tasks_for_quality(tier))
        arranged = {**row, "title": "Violin concerto arranged for solo piano"}
        tier, _, evidence = _quality_label(arranged, "classical-piano")
        self.assertNotEqual(tier, "canonical-core")
        self.assertFalse(evidence["piano_native_title"])

    def test_pdmx_fetch_selective_extract_rejects_escaping_members(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            archive = root / "mxl.tar.gz"
            with tarfile.open(archive, "w:gz") as handle:
                for name in ("mxl/a.mxl", "mxl/b.mxl"):
                    payload = name.encode("utf-8")
                    member = tarfile.TarInfo(name)
                    member.size = len(payload)
                    handle.addfile(member, io.BytesIO(payload))
            output = root / "output"
            output.mkdir()
            count = _extract_tar(archive, output, {"mxl/a.mxl"})
            self.assertEqual(count, 1)
            self.assertTrue((output / "mxl" / "a.mxl").exists())
            self.assertFalse((output / "mxl" / "b.mxl").exists())

            escaping = root / "escaping.tar.gz"
            with tarfile.open(escaping, "w:gz") as handle:
                payload = b"escape"
                member = tarfile.TarInfo("../escape.txt")
                member.size = len(payload)
                handle.addfile(member, io.BytesIO(payload))
            with self.assertRaises(ValueError):
                _extract_tar(escaping, output)

    def test_pdmx_parallel_resume_preserves_prefixes_and_fills_gaps(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            _part_path(root, 0, 99).write_bytes(b"a" * 40)
            _part_path(root, 100, 199).write_bytes(b"b" * 50)
            existing = _normalize_existing_parts(root, 0, 300)
            gaps = _missing_ranges(0, 299, [(start, end) for start, end, _ in existing])
            pending = [
                (start, end, _part_path(root, start, end))
                for gap_start, gap_end in gaps
                for start, end in _split_range(gap_start, gap_end, 32)
            ]
            ranges = sorted(existing + pending)
            _validate_coverage(ranges, 0, 300)
            self.assertEqual([(start, end) for start, end, _ in existing], [(0, 39), (100, 149)])
            self.assertEqual(gaps, [(40, 99), (150, 299)])

    def test_curriculum_keeps_full_piece_and_adds_contextual_windows(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            score_path = root / "source.musicxml"
            score_path.write_text(
                """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1">
      <attributes><key><fifths>-3</fifths><mode>minor</mode></key><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
      <direction><sound tempo="76"/></direction>
    </measure>
    %s
  </part>
</score-partwise>
"""
                % "\n".join(f'    <measure number="{number}"/>' for number in range(2, 81)),
                encoding="utf-8",
            )
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "work_id": "work-1",
                        "split": "train",
                        "path": score_path.name,
                        "title": "Source",
                        "composer": "Composer",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            summary = prepare_curriculum(manifest, root / "curriculum")
            self.assertEqual(summary["blueprints_written"], 1)
            examples_path = root / "curriculum" / "curriculum_examples.jsonl"
            examples = [
                json.loads(line)
                for line in examples_path.read_text(encoding="utf-8").splitlines()
            ]
            whole = next(row for row in examples if row["task"] == "whole-piece-generate")
            self.assertEqual(whole["target_range"], [1, 80])
            windows = [row for row in examples if row["task"] == "section-expand-16-64"]
            self.assertEqual({row["target_range"][1] - row["target_range"][0] + 1 for row in windows}, {16, 32, 64})
            self.assertTrue(all(row["context"]["piece_blueprint"] == "blueprint-work-1" for row in windows))
            self.assertTrue(all(row["context"]["future_ending_target"]["measure"] == 80 for row in windows))
            run_plan = create_run_plan(manifest, root / "run_plan.json", examples_path)
            self.assertEqual(run_plan["curriculum_examples"]["status"], "prepared")

    def test_curriculum_respects_quality_task_gates(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "work_id": work_id,
                            "split": "train",
                            "path": f"{work_id}.mxl",
                            "measure_count": 80,
                            "quality_tier": tier,
                            "tasks": _tasks_for_quality(tier),
                        }
                    )
                    for work_id, tier in (("broad-work", "broad"), ("core-work", "canonical-core"))
                )
                + "\n",
                encoding="utf-8",
            )
            summary = prepare_curriculum(manifest, root / "curriculum", dataset_root=root)
            examples = [
                json.loads(line)
                for line in (root / "curriculum" / "curriculum_examples.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            broad_tasks = {row["task"] for row in examples if row["work_id"] == "broad-work"}
            core_tasks = {row["task"] for row in examples if row["work_id"] == "core-work"}
            self.assertEqual(broad_tasks, {"score-dsl-autoencode"})
            self.assertIn("whole-piece-generate", core_tasks)
            self.assertEqual(summary["quality_tier_counts"], {"broad": 1, "canonical-core": 1})
            run_plan = create_run_plan(
                manifest,
                root / "run_plan.json",
                root / "curriculum" / "curriculum_examples.jsonl",
            )
            self.assertEqual(run_plan["quality_tier_counts"], {"broad": 1, "canonical-core": 1})
            self.assertEqual(run_plan["curriculum_examples"]["quality_task_counts"]["broad:score-dsl-autoencode"], 3)

    def test_musicxml_score_import_preserves_professional_notation(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            source = Path(raw_dir) / "notation.musicxml"
            source.write_text(_notation_musicxml(), encoding="utf-8")
            score = import_musicxml_score(source)
            self.assertEqual(score.plan.title, "Training Etude")
            self.assertEqual(score.metadata["composer"], "Test Composer")
            self.assertEqual(score.plan.key, "C minor")
            self.assertEqual(score.plan.tempo_bpm, 76)
            self.assertEqual(score.plan.meter.beats, 4)
            self.assertEqual(score.plan.meter.beat_type, 4)
            self.assertEqual(score.notes[0].pitch, 60)
            self.assertEqual(score.notes[0].articulation, "staccato")
            self.assertEqual(score.notes[0].fingering, "1")
            self.assertTrue(score.notes[0].tie_start)
            self.assertTrue(any(note.staff == 2 for note in score.notes))
            self.assertEqual(
                {"tempo", "dynamic", "wedge-start", "wedge-stop", "pedal-start", "pedal-stop"},
                {direction.kind for direction in score.directions},
            )
            self.assertEqual(
                {"articulations", "dynamics", "fingering", "pedal", "ties", "wedges"},
                set(score.plan.markings),
            )
            profile = analyze_notation(score)
            self.assertEqual(profile["texture"], score.plan.texture)
            self.assertGreater(profile["lower_staff_measure_coverage"], 0)
            self.assertIn(profile["variation_tier"], {"low", "moderate", "high"})
            self.assertEqual(score.layout_hints[0].measure, 2)
            self.assertEqual(validate_score(score), [])

    def test_materialized_training_examples_include_full_plan_and_local_context(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            (root / "notation.musicxml").write_text(_notation_musicxml(), encoding="utf-8")
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "work_id": "work-notation",
                        "split": "train",
                        "path": "notation.musicxml",
                        "title": "Training Etude",
                        "composer": "Test Composer",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            curriculum_dir = root / "curriculum"
            prepare_curriculum(manifest, curriculum_dir, dataset_root=root)
            dataset_dir = root / "model_dataset"
            summary = materialize_training_dataset(curriculum_dir, dataset_dir, dataset_root=root)
            self.assertEqual(summary["source_scores_imported"], 1)
            self.assertEqual(summary["source_scores_failed"], 0)
            self.assertGreater(summary["examples_written"], 0)
            self.assertEqual(summary["score_marking_counts"]["pedal"], 1)
            self.assertEqual(summary["model_representation"], "compact-measure-interleaved-v3")
            rows = [
                json.loads(line)
                for line in (dataset_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            whole = next(row for row in rows if row["task"] == "whole-piece-generate")
            self.assertEqual(whole["target_range"], [1, 80])
            imported_score = read_score(dataset_dir / "scores" / "work-notation" / "score.ir.json")
            decoded_whole = decode_model_score(
                whole["target_scoredsl"],
                imported_score.plan,
                imported_score.motif_bank,
            )
            self.assertEqual(decoded_whole.plan.measure_count, 80)
            self.assertEqual(decoded_whole.plan.genre, "unclassified-piano")
            local = next(
                row
                for row in rows
                if row["task"] == "section-expand-16-64"
                and row["target_range"][0] > 1
                and row["target_range"][1] < 80
            )
            self.assertEqual(local["target_scope"], "fragment")
            self.assertEqual(local["model_input"]["piece_plan"]["measure_count"], 80)
            self.assertIsNotNone(local["model_input"]["left_neighbor_scoredsl"])
            self.assertIsNotNone(local["model_input"]["right_neighbor_scoredsl"])
            self.assertEqual(local["model_input"]["future_ending_target"]["measure"], 80)
            self.assertEqual(local["model_input"]["notation_constraints"]["texture"], imported_score.plan.texture)
            fragment = decode_model_score(
                local["target_scoredsl"],
                imported_score.plan,
                imported_score.motif_bank,
            )
            self.assertTrue(all(local["target_range"][0] <= note.measure <= local["target_range"][1] for note in fragment.notes))
            spec = build_training_spec(
                TrainConfig(
                    dataset_dir=str(dataset_dir),
                    output_dir=str(root / "training_run"),
                    tasks=("whole-piece-generate",),
                )
            )
            self.assertEqual(spec["dataset"]["selected_examples"], 1)
            self.assertEqual(spec["model"]["complete_piece_truncation_policy"], "forbidden")
            self.assertTrue((root / "training_run" / "training_spec.json").exists())

    def test_materializer_records_malformed_musicxml_without_aborting_batch(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            (root / "broken.musicxml").write_text("<score-partwise><broken>", encoding="utf-8")
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "work_id": "broken-work",
                        "split": "train",
                        "path": "broken.musicxml",
                        "measure_count": 48,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            curriculum = root / "curriculum"
            prepare_curriculum(manifest, curriculum, dataset_root=root)
            summary = materialize_training_dataset(curriculum, root / "dataset", dataset_root=root)
            self.assertEqual(summary["source_scores_imported"], 0)
            self.assertEqual(summary["source_scores_failed"], 1)
            self.assertGreater(summary["examples_skipped"], 0)

    def test_materializer_rejects_stale_blueprint_measure_count(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "work_id": "stale-work",
                        "split": "train",
                        "path": "later.musicxml",
                        "measure_count": 81,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            curriculum = root / "curriculum"
            prepare_curriculum(manifest, curriculum, dataset_root=root)
            (root / "later.musicxml").write_text(_notation_musicxml(), encoding="utf-8")
            dataset = root / "dataset"
            summary = materialize_training_dataset(curriculum, dataset, dataset_root=root)
            self.assertEqual(summary["source_scores_imported"], 0)
            self.assertEqual(summary["source_scores_failed"], 1)
            errors = [
                json.loads(line)
                for line in (dataset / "materialization_errors.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertIn("regenerate curriculum", errors[0]["reason"])

    def test_materializer_excludes_oversized_target_without_truncation(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            (root / "notation.musicxml").write_text(_notation_musicxml(), encoding="utf-8")
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "work_id": "oversized-work",
                        "split": "train",
                        "path": "notation.musicxml",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            curriculum = root / "curriculum"
            prepare_curriculum(manifest, curriculum, dataset_root=root)
            dataset = root / "dataset"
            summary = materialize_training_dataset(
                curriculum,
                dataset,
                dataset_root=root,
                max_example_characters=1,
            )
            self.assertEqual(summary["examples_written"], 0)
            self.assertGreater(summary["examples_excluded_oversized"], 0)
            oversized = [
                json.loads(line)
                for line in (dataset / "oversized_examples.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(all("without truncation" in row["reason"] for row in oversized))

    def test_tokenizer_preflight_reports_oversized_target_before_training(self):
        class CharacterTokenizer:
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": list(text)}

        row = {
            "example_id": "too-long",
            "task": "whole-piece-generate",
            "target_scope": "complete-piece",
            "model_input": {"instruction": "Generate score."},
            "target_scoredsl": "END_SCORE\n",
        }
        result = _validate_token_lengths([row], CharacterTokenizer(), max_seq_length=1)
        self.assertEqual(result["oversized_example_count"], 1)
        self.assertGreater(result["maximum_tokens"], 1)

    def test_training_spec_can_resume_a_previous_adapter(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "train.jsonl").write_text(
                json.dumps(
                    {
                        "example_id": "resume-example",
                        "task": "whole-piece-generate",
                        "model_input": {"instruction": "Generate score."},
                        "target_scoredsl": "END_SCORE\n",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            spec = build_training_spec(
                TrainConfig(
                    dataset_dir=str(dataset),
                    output_dir=str(root / "run"),
                    resume_adapter_dir="training_runs/01_grammar/adapter",
                    resume_checkpoint_dir="training_runs/09_two_staff_refinement/checkpoint-25",
                )
            )
            self.assertEqual(spec["model"]["resume_adapter_dir"], "training_runs/01_grammar/adapter")
            self.assertEqual(
                spec["model"]["resume_checkpoint_dir"],
                "training_runs/09_two_staff_refinement/checkpoint-25",
            )
            self.assertIn("--resume-adapter-dir training_runs/01_grammar/adapter", spec["launch_command"])
            self.assertIn(
                "--resume-checkpoint-dir training_runs/09_two_staff_refinement/checkpoint-25",
                spec["launch_command"],
            )
            self.assertEqual(spec["model"]["loss_strategy"], "checkpointed chunked LM-head cross-entropy")
            self.assertEqual(spec["model"]["loss_chunk_tokens"], 256)
            self.assertEqual(
                spec["model"]["attention_fallback_policy"],
                "disable quadratic math SDPA fallback on CUDA",
            )
            self.assertEqual(spec["model"]["supervision_weighting"]["structural_token_weight"], 8.0)
            self.assertIn("--loss-chunk-tokens 256", spec["launch_command"])

    def test_training_spec_can_bound_a_real_sample_smoke_run(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            dataset = root / "dataset"
            dataset.mkdir()
            rows = [
                {
                    "example_id": f"example-{index}",
                    "task": "score-dsl-autoencode",
                    "model_input": {"instruction": "Reconstruct score."},
                    "target_scoredsl": "END_SCORE\n" * index,
                }
                for index in range(1, 5)
            ]
            (dataset / "train.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            spec = build_training_spec(
                TrainConfig(
                    dataset_dir=str(dataset),
                    output_dir=str(root / "run"),
                    max_examples=2,
                    max_example_characters=1000,
                    max_steps=1,
                    loss_chunk_tokens=128,
                    gradient_accumulation_steps=1,
                )
            )
            self.assertEqual(spec["dataset"]["selected_examples"], 2)
            self.assertIn("--max-examples 2", spec["launch_command"])
            self.assertIn("--max-example-characters 1000", spec["launch_command"])
            self.assertIn("--max-steps 1", spec["launch_command"])
            self.assertIn("--loss-chunk-tokens 128", spec["launch_command"])

    def test_training_spec_can_select_notation_diverse_sources(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            dataset = root / "dataset"
            dataset.mkdir()
            rows = [
                {
                    "example_id": f"example-{variation}",
                    "task": "section-variation-revise",
                    "notation_profile": {
                        "variation_score": variation,
                        "lower_staff_measure_coverage": lower_staff_coverage,
                    },
                    "model_input": {"instruction": "Revise contrast."},
                    "target_scoredsl": "END_SCORE\n",
                }
                for variation, lower_staff_coverage in ((0.4, 0.9), (0.7, 0.4), (0.8, 0.9))
            ]
            (dataset / "train.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            spec = build_training_spec(
                TrainConfig(
                    dataset_dir=str(dataset),
                    output_dir=str(root / "run"),
                    min_variation_score=0.55,
                    min_lower_staff_measure_coverage=0.75,
                )
            )
            self.assertEqual(spec["dataset"]["selected_examples"], 1)
            self.assertIn("--min-variation-score 0.55", spec["launch_command"])
            self.assertIn("--min-lower-staff-measure-coverage 0.75", spec["launch_command"])

    def test_training_loss_weights_score_structure_and_target_prefix(self):
        class CharacterTokenizer:
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": list(text)}

        target_ids = list("x SCORE payload\nNOTE payload\n")
        weights = _target_loss_weights(
            target_ids,
            CharacterTokenizer(),
            structural_token_weight=8.0,
            target_prefix_tokens=1,
            target_prefix_weight=4.0,
        )
        self.assertEqual(weights[0], 4.0)
        score_start = target_ids.index("S")
        note_start = "".join(target_ids).index("NOTE")
        self.assertTrue(all(weight == 8.0 for weight in weights[score_start : score_start + len("SCORE")]))
        self.assertTrue(all(weight == 8.0 for weight in weights[note_start : note_start + len("NOTE")]))

    def test_cloud_bundle_excludes_score_cache_and_verifies_before_extract(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            dataset = root / "model_dataset"
            dataset.mkdir()
            for name in (
                "train.jsonl",
                "valid.jsonl",
                "test.jsonl",
                "summary.json",
                "materialization_errors.jsonl",
                "oversized_examples.jsonl",
            ):
                (dataset / name).write_text(f"{name}\n", encoding="utf-8")
            score_cache = dataset / "scores" / "work-1"
            score_cache.mkdir(parents=True)
            (score_cache / "score.dsl").write_text("do not upload\n", encoding="utf-8")
            bundle = root / "bundle.tar.gz"
            packaged = package_cloud_dataset(dataset, bundle)
            self.assertEqual(packaged["excluded_local_cache"], "scores/")
            verification = verify_cloud_bundle(bundle)
            self.assertEqual(len(verification["verified_files"]), 6)
            extracted = extract_cloud_bundle(bundle, root / "remote")
            self.assertTrue(Path(extracted["dataset_dir"], "train.jsonl").exists())
            self.assertFalse(Path(extracted["dataset_dir"], "scores").exists())

    def test_release_gate_dry_run_writes_review_artifacts(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            summary = run_release_gate(
                root,
                piece_count=2,
                candidates=1,
                skip_musescore=True,
                include_sonata=True,
            )
            self.assertEqual(summary["generated_piece_count"], 2)
            self.assertEqual(summary["rates"]["tempo_overlay_leakage_count"], 0)
            self.assertEqual(summary["rates"]["two_staff_texture_rate"], 1.0)
            self.assertFalse(summary["checks"]["piece_count"])
            self.assertFalse(summary["checks"]["musescore_render_rate"])
            self.assertFalse(summary["release_ready"])
            self.assertEqual(len(summary["experimental_sonata_allegro"]), 1)
            self.assertTrue((root / "automatic_results.csv").exists())
            self.assertTrue((root / "human_review.csv").exists())
            self.assertTrue((root / "release_gate.json").exists())
            self.assertIn("two_staff_texture", (root / "human_review.csv").read_text(encoding="utf-8"))
            self.assertIn("piece_001/gallery.html", (root / "review_gallery.html").read_text(encoding="utf-8"))

    def test_release_gate_can_route_generation_through_checkpoint_adapter(self):
        with tempfile.TemporaryDirectory() as raw_dir:
            captured = []

            def generate(args):
                captured.append(args)
                compose(args)

            with patch("midi_llm.release_gate.generate_from_checkpoint", side_effect=generate):
                run_release_gate(
                    raw_dir,
                    piece_count=1,
                    candidates=1,
                    skip_musescore=True,
                    adapter_dir="training_runs/test/adapter",
                    section_candidates=3,
                )
            self.assertEqual(captured[0].adapter_dir, "training_runs/test/adapter")
            self.assertEqual(captured[0].section_candidates, 3)

    @unittest.skipUnless(find_musescore(), "MuseScore is not installed")
    def test_musescore_exports_pdf_and_png(self):
        score, _ = self.build_score()
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            musicxml = write_musicxml(score, root / "score.musicxml")
            result = render_musescore(musicxml, root)
            self.assertTrue(result["available"])
            self.assertTrue((root / "score.pdf").exists())
            self.assertGreater(len(result["pages"]), 0)


def _notation_musicxml(measure_count=80):
    trailing = "\n".join(
        f"""
    <measure number="{number}">
      <note>
        <pitch><step>C</step><octave>4</octave></pitch>
        <duration>16</duration><voice>1</voice><staff>1</staff>
      </note>
    </measure>"""
        for number in range(3, measure_count + 1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <work><work-title>Training Etude</work-title></work>
  <identification><creator type="composer">Test Composer</creator></identification>
  <part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>4</divisions>
        <key><fifths>-3</fifths><mode>minor</mode></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <staves>2</staves>
      </attributes>
      <direction>
        <direction-type><words>Andante</words></direction-type>
        <sound tempo="76"/>
      </direction>
      <direction><direction-type><dynamics><p/></dynamics></direction-type><staff>1</staff></direction>
      <direction><direction-type><wedge type="crescendo"/></direction-type><staff>1</staff></direction>
      <direction><direction-type><pedal type="start"/></direction-type><staff>2</staff></direction>
      <note>
        <pitch><step>C</step><octave>4</octave></pitch>
        <duration>4</duration><voice>1</voice><staff>1</staff>
        <tie type="start"/>
        <notations>
          <tied type="start"/>
          <articulations><staccato/></articulations>
          <technical><fingering>1</fingering></technical>
        </notations>
      </note>
      <note>
        <chord/>
        <pitch><step>E</step><alter>-1</alter><octave>4</octave></pitch>
        <duration>4</duration><voice>1</voice><staff>1</staff>
      </note>
      <backup><duration>4</duration></backup>
      <note>
        <pitch><step>C</step><octave>3</octave></pitch>
        <duration>4</duration><voice>2</voice><staff>2</staff>
      </note>
      <forward><duration>12</duration></forward>
    </measure>
    <measure number="2">
      <print new-system="yes"/>
      <direction><direction-type><wedge type="stop"/></direction-type><staff>1</staff></direction>
      <direction><direction-type><pedal type="stop"/></direction-type><staff>2</staff></direction>
      <note>
        <pitch><step>C</step><octave>4</octave></pitch>
        <duration>16</duration><voice>1</voice><staff>1</staff>
        <tie type="stop"/><notations><tied type="stop"/></notations>
      </note>
    </measure>
    {trailing}
  </part>
</score-partwise>
"""


if __name__ == "__main__":
    unittest.main()
