"""Render the audited B1 result; no new evaluation or model selection."""
from pathlib import Path
import json,sys
out=Path(sys.argv[1]).resolve()
a=json.loads((out/'analysis_results.json').read_text())
passed=all(a['gates'].values())
lines=['# T4-B1 — actor GAE(0,95), critic MC', '',
    '**Đủ protocol; '+('đủ tín hiệu để lặp trước khi thay B.' if passed else 'chưa chứng minh cải thiện, giữ checkpoint B.')+'**', '',
    'Đã train lại cả hai nhánh từ B, 200 episode/200 cập nhật Adam mỗi nhánh, rồi chấm 400 trận/model trên seeds35000–35199, cả A0/A1. Teacher local, normalization tắt, critic MC, reward/mask/12 kênh giữ nguyên. Chỉ cách tính advantage của actor khác nhau; train seed 2026101504, gamma0,99, lambda0,95, LR1e-4, PhaseC/9×9/2/max500. CNN lấy mẫu qua mask khi test, không gọi planner.', '',
    '## Kết quả test đã khóa checkpoint', '',
    '| Model | ActiveWR | Win | Hòa | Tự nổ | Kills/trận | Thắng thụ động |',
    '|---|---:|---:|---:|---:|---:|---:|']
for name,r in a['models'].items():
    lines.append(f"| {name} | {r['active_win_rate']:.2f}% | {r['win_rate']:.2f}% | {r['draw_rate']:.2f}% | {r['suicide_rate']:.2f}% | {r['avg_kills']:.4f} | {r['passive_win_rate']:.2f}% |")
lines += ['', '| GAE so với | Chênh ActiveWR (điểm %) | CI95% theo cặp seed bàn |', '|---|---:|---|']
for name,c in a['comparisons'].items():
    lines.append(f"| {name} | {c['difference_pp']:+.2f} | [{c['ci95_pp'][0]:.2f}; {c['ci95_pp'][1]:.2f}] |")
lines += ['', '| Model/góc | ActiveWR | Hòa | Tự nổ | Kills/trận | Thắng thụ động |', '|---|---:|---:|---:|---:|---:|']
for name,r in a['models'].items():
    for corner,c in r['by_corner'].items():
        lines.append(f"| {name}/A{corner} | {c['active_win_rate']:.2f}% | {c['draw_rate']:.2f}% | {c['suicide_rate']:.2f}% | {c['avg_kills']:.4f} | {c['passive_win_rate']:.2f}% |")
lines += ['', 'Mỗi góc 200 trận/model. Gate yêu cầu tăng≥5điểm so với cả B và control, CI95% lower>0, tự nổ không cao hơn cả hai. Kết quả từng điều kiện: '+str(a['gates'])+'.', '',
    '## Validation và chẩn đoán', '', '| Nhánh | Best episode | Val ActiveWR ep0/100/200 | Learner steps | Clip gradient | Tự nổ train |', '|---|---:|---|---:|---:|---:|']
for arm,r in a['arms'].items():
    vals='/'.join(f"{v['active_win_rate']:.0f}%" for v in r['validations'])
    lines.append(f"| {arm} | {r['best_episode']} | {vals} | {r['learner_steps']} | {r['clipped_updates']}/200 | {r['training_suicides']} |")
lines += ['', 'Best được chọn trên validation 100 trận seed21000 tại 0/100/200, score và guard tự nổ cũ. Cả hai best đều ở episode 0 và có cùng weights B; vì vậy test của best không đo checkpoint GAE đã được cập nhật. CI [0;0] do cùng weights, không loại bỏ bất định về phương pháp GAE trên các lần train khác. Không dùng kết quả test để đổi sang final200.', '',
    '| Nhánh | Approach ep0/100/200 | Escape ep0/100/200 | GAE/raw std trung bình | MC residual std trung bình | Critic EV trung bình |', '|---|---|---|---:|---:|---:|']
for arm,r in a['arms'].items():
    groups=a['skills'][arm]
    approach='/'.join(str(groups[str(e)]['approach']['owned_kill_success']['count'])+'/32' for e in (0,100,200))
    escape='/'.join(str(groups[str(e)]['escape']['owned_kill_success']['count'])+'/32' for e in (0,100,200))
    ev='NA' if r['explained_variance_mean'] is None else f"{r['explained_variance_mean']:.4f}"
    lines.append(f"| {arm} | {approach} | {escape} | {r['advantage_raw_std_mean']:.4f} | {r['mc_raw_std_mean']:.4f} | {ev} |")
lines += ['', 'Thay đổi scale/clip hoặc kết quả32fixture không tự chứng minh tăng tỷ lệ thắng. MSE critic và explained variance đã đối chiếu bằng MC residual, không lẫn với GAE raw. Timeout terminal bootstrap0, learner death kết thúc trajectory; không cắt rollout giữa trận.', '',
    '## Thời gian, nguồn và kiểm chứng', '',
    f"Lượt đầy đủ này chạy {a['elapsed_seconds']/60:.2f} phút. Lượt đầu `gae-full-20261015-112853` dừng85,11phút ởcontrol200/GAE184 do trần train85 phút; test khi đó chưa mở. Sau review mới sửa trần train 100, tổng 150, rồi train lại cả hai từ B trong output mới. Không đổi hyperparameter/phương pháp hoặc xem test để chọn bản sửa. Hai lần dùng cùng trainseed nên không phải replication độc lập.", '',
    '174 regression tests và smoke CPU/CUDA đã qua trước pilot;49 tests liên quan sau sửa deadline cũng qua, gồm kiểm tra evaluation bị chặn bởi tổng thời gian còn lại. Source B0 estimator không thay trong B1; shared runner chỉ truyền flags. B hash `93bdf27812596f09edc0c2b40063600f2dc23a6964366d52581acd9f5c3b4f60` giữ nguyên.', '',
    f"Review raw ZIP{a['archive_files']}files/CRC/byte equality/source19/B/best/final hashes;400 train rows,6 validations,6 skill probes;{a['skills_replayed_games']}stored skill games/{a['skills_replayed_frames']}frames replay đúng,320rollouts thực do baseline cache 64 game ghi hai lần;1.200 test rows/rates/corners/pairedCI khớp. Model best/final khớp model digest của đúng skillprobeepisode.", '',
    'Raw ZIP nằm cạnh thư mục, không được sửa lại sau phân tích. `config.json`, `summary.json`, `selected_models.json`, `test_opened.json`, evaluation từnggame và source snapshot ghi protocol đã chạy. `analysis_results.json`/`review_results.py`/report được thêm sau khi tạo rawZIP.', '',
    '## Quyết định và bước kế tiếp', '',
    ('Giữ B cho đến khi lặp nhiều trainseed trên test mới và xác nhận tín hiệu.' if passed else 'Giữ B và mặc định MC/raw. B1 hoàn tất về thực thi, gate cải thiện FAIL; không coi đây là chứng minh GAE luôn vô ích. Chưa mở B2 critic lambda-return hoặc ghép normalization/PPO từ lượt này.'), '',
    'Ưu tiên tiếp theo là R0: chỉ log và audit contribution time/item/wall/kill/active-win/death/self-blast/passive-win-cancellation, bắt buộc tổng khớp shaped reward. So các nhóm timeout/thắng/chết để xác định có cần thử reward nào. Không sửa reward, mask, architecture hay train thêm trong B1.', '',
    'Giới hạn: một trainseed, RuleBot và duel 9×9;chưa có kết luận ổn định nhiều seed/FFA hoặc an toàn trước mọi phản bom tương lai.35000–35199 đã mở, không dùng để chọn thêm checkpoint/hyperparameter.31000/wood_branch không dùng trong B1.', '',
    'Protocol/tracker: `../../GAE_PILOT_GUIDE.md`, `../../PLAN_BOMBERMAN.md`.']
(out/'decision_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(out/'decision_report.md')
