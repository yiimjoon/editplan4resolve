"""Week 2 integration tests for multicam modules."""

from VideoForge.multicam.angle_selector import AngleSelector
from VideoForge.multicam.llm_tagger import LLMTagger
from VideoForge.multicam.plan_generator import MulticamPlanGenerator


def test_angle_selection() -> None:
    selector = AngleSelector()

    segments = [
        {
            "start_sec": 0.0,
            "end_sec": 5.0,
            "text": "Hello everyone",
            "scores": {
                0: {"sharpness": 0.8, "motion": 0.3, "stability": 0.7, "face_score": 0.6},
                1: {"sharpness": 0.9, "motion": 0.2, "stability": 0.8, "face_score": 0.9},
                2: {"sharpness": 0.7, "motion": 0.5, "stability": 0.6, "face_score": 0.3},
            },
        },
        {
            "start_sec": 5.0,
            "end_sec": 10.0,
            "text": "Watch this!",
            "scores": {
                0: {"sharpness": 0.7, "motion": 0.8, "stability": 0.6, "face_score": 0.4},
                1: {"sharpness": 0.8, "motion": 0.9, "stability": 0.7, "face_score": 0.5},
                2: {"sharpness": 0.9, "motion": 0.6, "stability": 0.8, "face_score": 0.3},
            },
        },
    ]

    selections = selector.select_angles(segments)

    assert len(selections) == 2
    assert selections[0]["angle_index"] == 1
    print("OK: angle selection")


def test_plan_generation() -> None:
    generator = MulticamPlanGenerator()

    angle_selections = [
        {
            "start_sec": 0.0,
            "end_sec": 5.0,
            "angle_index": 1,
            "angle_score": 0.87,
            "reason": "Best sharpness",
        }
    ]

    timeline_info = {"fps": 24.0, "name": "Test Timeline"}
    source_clips = ["cam1", "cam2", "cam3"]

    plan = generator.generate_plan(angle_selections, timeline_info, source_clips)

    assert len(plan.actions) == 1
    assert plan.actions[0].action_type == "cut_and_switch_angle"
    assert plan.actions[0].target == "V2"
    assert plan.actions[0].params["cut_frame"] == 0
    print("OK: plan generation")


def test_llm_tagger() -> None:
    tagger = LLMTagger()

    segments = [
        {"start_sec": 0.0, "end_sec": 5.0, "text": "Hello everyone"},
        {"start_sec": 5.0, "end_sec": 10.0, "text": "Watch this move!"},
    ]

    tags = tagger.tag_segments(segments)

    assert len(tags) == 2
    assert all(tag in ["speaking", "action", "emphasis", "neutral"] for tag in tags)
    print("OK: llm tagger")


if __name__ == "__main__":
    test_angle_selection()
    test_plan_generation()
    test_llm_tagger()
    print("\nAll Week 2 tests passed")
