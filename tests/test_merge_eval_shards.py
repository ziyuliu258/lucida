from scripts.merge_eval_shards import full_loop_success


def test_full_loop_success_requires_pose_and_protocol_tolerances():
    row = {
        "stopped": "1.0",
        "valid_action_rate": "1.0",
        "actions": "3",
        "add_sb_fraction": "0.049",
        "rotation_error_deg": "4.99",
    }
    assert full_loop_success(row) == 1.0

    row["rotation_error_deg"] = "5.0"
    assert full_loop_success(row) == 0.0
    row["rotation_error_deg"] = "4.0"
    row["stopped"] = "0.0"
    assert full_loop_success(row) == 0.0
