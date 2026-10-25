# S3 map 11 — acceptance matrix độc lập, 24/10/2026

Nguồn yêu cầu duy nhất: `2026-10-24-s3-map11-runner-step1.md`. Tác giả test chỉ đọc packet này, `AGENT.md`/`CLAUDE.md` và source/test hiện có; không đọc implementation plan hay source runner mới.

Lệnh chung (cmd):

```bat
E:\Bomberman\venv\Scripts\python.exe -m unittest test_run_s3_map11 -v
```

Các fixture dùng thư mục tạm và seed giả `1900000000` trở lên. Không đánh giá seed test thật. Tên JSON/NPZ và cột `decision_*`/`anchor_*` theo quy ước SD9 hiện có; kiểm tra qua API công khai của packet. `parallel_map` được chạy đồng bộ trong test để quan sát được mạng/env thật và giữ mock search trong cùng process; không thay env, mạng, RuleBot hoặc toán tối ưu.

| Yêu cầu / rủi ro | Mức | Scenario / test | Kết quả quan sát phải có |
|---|---|---|---|
| Public API | Contract | `test_public_stage_api` | Đủ 9 callable trong packet; signature cho phép các lời gọi được kiểm tra tiếp theo. |
| Import và help an toàn | Process integration | `test_import_is_side_effect_free`, `test_help_lists_stages_without_work` | Import mới không gọi `env.step`/optimizer; help liệt kê 6 stage và không gọi stage nào. |
| Stage CLI | CLI contract | `test_cli_dispatches_only_requested_stage`, `test_init_cli_requires_collect_and_test_seed` | Chỉ gọi stage đã chỉ định; thiếu seed bắt buộc bị argparse từ chối. |
| Khóa cấu hình full | Unit contract | `test_full_configuration` | 11×11/13 kênh/2 agent, pilot 100 bàn×2 góc, baseline100/validation400/test400, max500/E475-25-40, pE .75, neo .05, Adam1e-4/batch256/16 epoch/candidates4-8-12-16/KL1/heldout .1. |
| Smoke riêng | Unit contract | `test_smoke_configuration` | 4 bàn/8 game collect, max12, mỗi phép đo4, batch16/2epoch/candidate2, cờ smoke. |
| Đăng ký không làm việc dài | Integration | `test_initialize_records_provenance_without_games_or_gradients` | Lưu config, SHA S3, source hashes/runtime; không game/gradient/checkpoint mới. |
| Không ghi đè run | Filesystem integration | `test_initialize_never_overwrites_existing_output` | Từ chối output tồn tại, giữ nguyên sentinel. |
| Seed hợp lệ | Boundary contract | `test_invalid_seed_ranges_are_rejected` | Từ chối âm, collect0/1 và giao nhau của mọi cặp collect/validation/test (kiểm tra cả phần cuối dải). |
| Seed test cũ | Safety integration | `test_initialize_refuses_known_used_test_seed` | Full init seed57000 bị từ chối trước game/marker test. |
| Mạng chuyển map từ S3 | Real torch integration | `test_load_net_transfers_real_s3_to_11x11` | State bằng `smart_load_weights` trên mạng11, forward `(N,13,11,11)` cho logits6/value; SHA gốc không đổi. |
| Lỗi load | Error integration | `test_load_net_rejects_missing_and_invalid_checkpoint` | File thiếu và format hỏng bị từ chối. |
| Stage cần đăng ký | Error contract | `test_stages_refuse_unregistered_output` | baseline/collect/train/validate/test không tự khởi tạo run hoặc chạy game. |
| Nguồn, checkpoint, config bị đổi | Provenance integration | `test_registered_source_hash_tampering_is_refused`, `test_registered_checkpoint_hash_tampering_is_refused`, `test_registered_configuration_tampering_is_refused` | Stage từ chối provenance không khớp trước game. |
| Worker1–5 | Boundary contract | `test_worker_limits_are_enforced` | Stage có workers từ chối0/6, không gọi game. |
| Collection map/teacher/neo | Real env/network integration | `test_collection_uses_real_11x11_teacher_and_new_anchor_logits` | 8game4bàn×2góc, env11/2agent/max12/E, tensor13×11×11, search3 và RuleBot được gọi; logits neo bằng S3 chuyển map trên chính obs đó. |
| Shard rỗng vẫn11 | Boundary integration | `test_empty_decision_shards_keep_11x11_shape`, `test_empty_anchor_shards_keep_11x11_shape` | `decision_obs` hoặc `anchor_obs` rỗng có shape `(0,13,11,11)`. |
| Shard hỏng | Error integration | `test_corrupt_shard_is_refused_without_overwrite` | Không ghi đè shard không phải NPZ; resume từ chối. |
| Resume chỉ phần thiếu | Recovery integration | `test_resume_collects_only_missing_shards` | Giữ byte các shard có sẵn; chỉ chạy seed của shard thiếu. |
| Artifact hoàn tất bất biến | Safety integration | `test_completed_collection_is_not_overwritten`, `test_baseline_is_not_overwritten` | Gọi lại bị từ chối; nội dung hoàn tất không đổi. |
| Baseline11 | Real env/network integration | `test_baseline_uses_real_11x11_env_and_registered_validation_seed` | Theo làm rõ packet: S3 và S3+search3, mỗi arm4game smoke (tổng8), RuleBot, seed validation đăng ký, E đúng, tất cả network input11. |
| Train tích hợp ngắn | Real optimizer/network integration | `test_train_synthetic_11x11_data_updates_actor_preserves_critic_and_board_split` | Synthetic11 đi qua shard→train thật2epoch, actor đổi/critic nguyên vẹn, candidate2 load11, số liệu hữu hạn, không chung bàn train/heldout, không augmentation9. |
| Từ chối dữ liệu cũ | Negative integration | `test_train_rejects_9x9_decision_or_anchor_observations`, `test_train_rejects_wrong_action_shape` | Dữ liệu9×9 hoặc mask/logits sai số action bị ValueError có ngữ cảnh; không có candidate. |
| Chọn bằng ActiveWR, tự nổ, tie | Stage integration + bounded evaluation double | `test_validation_selects_highest_safe_activewr_and_earliest_tie` | Candidate tự nổ >baseline+1pp bị loại, bằng ngưỡng được xét, hòa ActiveWR chọn epoch sớm; chỉ dùng seed validation. |
| Không hơn baseline thì khóa test | Safety integration + bounded evaluation double | `test_validation_without_improvement_keeps_test_closed` | Validation có thể ghi STOP/chọn để báo cáo nhưng test từ chối và không tạo marker/evaluate test. |
| Test một lần | Safety integration + bounded evaluation double | `test_smoke_test_is_labelled_and_cannot_run_twice` | Smoke so candidate với baseline11, không dùng seed test full, đánh dấu smoke; lần2 từ chối trước evaluation. |
| Selection đóng băng | Safety integration | `test_validation_cannot_be_rerun_after_selection_or_test_open` | Không ghi đè selection và không chọn lại sau marker test. |

## Giới hạn kiểm chứng

- Test không chạy search thật, pilot100bàn,16epoch hay400trận; không có kết luận về chất lượng S3 hoặc model11. Các lệnh đó thuộc bước sau và ngoài phạm vi packet bước1.
- Real collection/baseline và train synthetic là tích hợp tương đương hành trình cốt lõi cho bước chuẩn bị; evaluation double chỉ tạo bảng kết quả để kiểm luật chọn/khóa seed. ProcessPool/worker đa tiến trình không được chạy; thay bằng scheduler đồng bộ giữ nguyên công việc thật.
- Chỉ forward/gradient11 và byte checkpoint được kiểm; các khẳng định “không RL/không sửa runner cũ”, tài liệu hướng dẫn, progress và nhật ký/tracker còn cần parent review diff. Gate CI95 cụ thể được giữ trong packet và reviewer phải xác nhận khi đọc source; API packet không công khai helper gate nên test không bắt buộc một tên helper riêng.
- RED baseline gốc: `2026-10-24-s3-map11-acceptance-red.log`, 33test/33error do module còn thiếu; runtime numpy2.4.6/torch2.11.0+cu128. Các sửa fixture trước implementation và RED đóng băng được ghi trong `2026-10-24-s3-map11-acceptance-red-notes.md` và `2026-10-24-s3-map11-acceptance-red-final.log`. Lỗi dependency hoặc syntax không được tính là RED hợp lệ.
