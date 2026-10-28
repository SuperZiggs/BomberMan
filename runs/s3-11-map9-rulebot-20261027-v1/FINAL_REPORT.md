# S3-11 trên9×9 vớiRuleBot — kết quả cuối,27/10/2026

**Kết luận:** S3-11 vẫn chơi tốt trướcRuleBot9×9 trong phép đo này (WR94.75%, ActiveWR87.50%), nhưng kém S3 gốc (WR98.25%, ActiveWR93.25%). ActiveWR giảm5.75điểm phần trăm, CI95[-9.75,-1.50] hoàn toànâm; WR giảm3.50ppCI[-6,-1], hòa tăng2.75ppCI[.75,5]. Có bằng chứng suy giảm hiệu năng9×9 sau nhánh thích nghi11×11; không kết luận giữ nguyên kỹ năng hoặc tốt hơn trên9×9. Không đặt gate sau kết quả và không chạy thêm test để cứu số liệu.

**Thông số:** Codex/Windows `E:\Bomberman`, Python3.11.9/torch2.11.0+cu128/NumPy2.4.6; S3-11 `runs/s3-map11-20261024-auto-v1/train/epoch_16.pt` SHA `17748471…a95f`, S3 gốc `runs/sd9-20261023/train/epoch_12.pt` SHA `4185375f…69ed7`; cùng đấuRuleBot trên9×9/13kênh/2agent/max500/E475-25-40. CNN CPU lấy mẫu+mask, khôngsearch/train;2arm song song/mỗiarm1torchthread. Seed840000–840199/200bàn×2góc=400trận/model/800tổng, mộtlầnlaunch, khôngresume/replay. Bootstrap ghép cặp theo200bàn/10000lần/seed20261014, chỉ đánh giá mô tả, khônggate/chọncheckpoint. Bắt đầu00:47:20ngày27/10/2026, artifact cuối completed_utc00:49:49; launch→completed_utc cuối148.699s (~2.48phút), không có exit-time native riêng. Thời gian evaluator S3-11 145.781s/S3 120.609s. Run `runs/s3-11-map9-rulebot-20261027-v1`; S3-11 được loader nội suy11→9 trongRAM, không sửa filecheckpoint.

## Kết quả ghép cặp

| Chỉ số (%) | S3-11 trên9×9 | S3 gốc | Delta S3-11−S3 (pp) | CI95 delta (pp) |
|---|---:|---:|---:|---|
| WR | 94.75 | 98.25 | -3.50 | [-6.00, -1.00] |
| ActiveWR | 87.50 | 93.25 | -5.75 | [-9.75, -1.50] |
| Hòa | 3.75 | 1.00 | +2.75 | [0.75, 5.00] |
| Hòa hết bước | 3.50 | 1.00 | +2.50 | [0.50, 4.75] |
| Thua | 1.50 | 0.75 | +0.75 | [-0.50, 2.00] |
| Tự nổ | 0.25 | 0.00 | +0.25 | [0.00, 0.75] |
| Thắng thụ động | 6.25 | 4.50 | +1.75 | [-1.25, 5.00] |
| Thắng nhờ vòng | 1.00 | 0.50 | +0.50 | [-0.75, 1.75] |
| Chết vì vòng | 1.50 | 0.25 | +1.25 | [0.00, 2.50] |

S3-11:379thắng/350thắngchủđộng/15hòa/6thua/1tựnổ; S3:393thắng/373thắngchủđộng/4hòa/3thua/0tựnổ. Thắng chủ động là tập con của thắng, tự nổ là nguyên nhân chết nên không cộng các cột này thành tổng800.

## Từng góc

| Model | Góc | Trận | WR% | ActiveWR% | Hòa% | Tự nổ% |
|---|---:|---:|---:|---:|---:|---:|
| S3_11 | 0 | 200 | 96.00 | 86.50 | 2.00 | 0.50 |
| S3_11 | 1 | 200 | 93.50 | 88.50 | 5.50 | 0.00 |
| S3 | 0 | 200 | 98.00 | 90.50 | 1.50 | 0.00 |
| S3 | 1 | 200 | 98.50 | 96.00 | 0.50 | 0.00 |

## Các chỉ số phụ

S3-11:165.345bước/16.08bom/.875kill mỗi trận; S3:127.5425bước/12.0825bom/.9325kill. Đây là các trung bình mô tả, không có CI cho chênh lệch sốbước/bom trong protocol.

## Xác minh

Đủ400rawgame/model/200bàn×2góc,800tổng, marker mộtlần; metadata9×9/E/max500 đúng registration. Raw outcome/counter/seed-góc và thống kê gốc đều khớp khi tính lại;19source/live+archive,2checkpoint/config/runtime đúng provenance. Xác nhận native root/interpreter/children không còn, không cólock. Harvest chỉ runpy khônggọi main và các helper _summarize/_paired; evaluate_seeded/parallel_map được chặn trong bước xác thực,0game mới, không ghi đè evaluation. Summary SHA256 `7b7e76ffd613f2cf10108569fc8571360219fae00ab0ea0854d619afc9167e09`; rawSHA và proof ở `completion-verification.json`.

## Giới hạn và quyết định

Giới hạn9×9/1v1RuleBot/lịchE/mộtcheckpoint và mộtbộ200bàn. Kết quả11×11 vớiRuleBot trước đó ActiveWR87.25% thuộc bộseed820000 riêng; có bằng chứng S3-11 thắngRuleBot ở cảhai map theo từngphépđo, nhưng không suy delta ngangmap từ hai tập khácseed. Không suy CNN/FFA hoặc mọi luật/map; CNN11×11 đã cho hòa cao ở benchmark riêng. Giữ model phục vụ/checkpoint gốc, không train/promote/Gitcommit/push hoặc can thiệpSD10.

State cuối completed; heartbeat45phút đã được xóa, automation_update xác nhận deleteStatus=deleted. Xác nhận lưu ở autorun-state.json và automation-delete-confirmation.json. Không còn bước theo dõi hoặc đánh giá kế tiếp được tự mở.
