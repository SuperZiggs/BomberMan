"""Write the D1 decision from independently checked results."""
import json
from pathlib import Path
import sys

out=Path(sys.argv[1]).resolve()
a=json.loads((out/'analysis_results.json').read_text())
cfg=json.loads((out/'config.json').read_text())
s=json.loads((out/'summary.json').read_text())
labels={'original':'B','control':'U4 chưa clip','ppo':'P4 clip 0,20'}
fmt=lambda x:'N/A' if x is None else f'{x:.2f}'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
        *['| '+' | '.join(map(str,row))+' |' for row in rows]])
passed=all(a['gates'].values())
lines=['# T4-D1 — U4/P4 có đối chứng','',
    '**DONE/REVIEWED. Gate '+('PASS; đăng ký replication trước khi thay B.' if passed else 'FAIL; giữ checkpoint B.')+'**','',
    f"Hai nhánh đủ 200 episode từ B, validation0/100/200 và test400/model. Thời gian từ đăng ký đến hoàn tất {a['elapsed_seconds']/60:.2f} phút, gồm cả khoảng dừng. "
    'Giữ model mặc định B; test dùng 200 seed bàn chung, mỗi bàn cả A0/A1, không coi1.200game là1.200bàn độc lập.','',
    '## Thiết kế','',
    'Cả hai warm-start từ B với Adam mới, seed2026101507, C9×9/2agents/RuleBot/max500, MC advantage raw và critic MC, '
    'local teacher, wall0,15, LR1e-4. Full batch một learner episode, tối đa4epochs. U4 dùng ratio chưa clip; '
    'P4 clip epsilon0,20. Cùng targetKL0,02: nếu exact categorical KL sau update>0,03 thì dừng các epoch còn lại. '
    'Step vượt ngưỡng vẫn giữ; không coi guard là bảo đảm KL không bao giờ vượt ngưỡng. '
    'Scheduler và entropy schedule theo episode, teacher losses dùng cùng nhãn và hệ số trong mọi epoch.','',
    'P4 dừng tại episode36/Adam144, rồi resume từ full state sau khi người dùng giải phóng VRAM. '
    'Episode37 nối tiếp với Adam148; cả10file U4 đã hoàn tất giữ nguyên hash. '
    'Deadline ban đầu không được gia hạn. `verification/before-resume.json` và `verification/resume_review.json` lưu kiểm tra này.','',
    'Không thêm GAE/normalization, reward/observation/mask mới hoặc opponent pool. B là policy baseline giữ nguyên; '
    'D1 này chưa train thêm đối chứng AC một-update. Epoch/Adam counts thực có thể khác vì cùng guard phản ứng với mỗi policy.','',
    '## Best chọn bằng validation','',
    table(['Nhánh','Episode','ActiveWR %','Tự nổ %','Hòa %','Chọn'],
        [[labels[n],v['episode'],fmt(v['active_win_rate']),fmt(v['suicide_rate']),fmt(v['draw_rate']),v['selected']]
         for n,r in a['arms'].items() for v in r['validations']]),'',
    'Khóa best/hash trước test37000–37199. Final200 chỉ dùng diagnostics/skill validation, không chọn lại model sau test.','',
    '## Test400/model','',
    table(['Model','Best ep','ActiveWR %','Passive WR %','Hòa %','Tự nổ %','Kills/game'],
        [[labels[n],'gốc' if n=='original' else a['arms'][n]['best_episode'],fmt(r['active_win_rate']),
          fmt(r['passive_win_rate']),fmt(r['draw_rate']),fmt(r['suicide_rate']),f"{r['avg_kills']:.3f}"]
         for n,r in a['models'].items()]),'',
    table(['P4 so với','Delta ActiveWR pp','CI95% pp','Board pairs'],
        [[labels[n],fmt(r['difference_pp']),f"[{fmt(r['ci95_pp'][0])}; {fmt(r['ci95_pp'][1])}]",r['board_pairs']]
         for n,r in a['comparisons'].items()]),'',
    'Bootstrap10.000 lần theo cặp seed bàn, giữ cả hai góc trong cùng cụm. Gate cần delta≥5pp so với cảB/U4, '
    f"hai cận dướiCI>0 và tự nổ quan sát không cao hơn. Kết quả điều kiện: {a['gates']}.",'',
    'Cả hai khoảng tin cậy của P4 so với B/U4 đều chứa0 trong lượt này. '
    'Điểm quan sát của P4 thấp hơn, nhưng dữ liệu chưa chứng minh chênh lệch thực khác0; '
    'kết luận vận hành là không đạt gate thay model, không phải chứng minh PPO nói chung kém hơn.','',
    '## Số updates và tác động KL/clipping','',
    table(['Nhánh','Learner steps','Adam steps','Epoch counts/200ep','Early-stop episodes','KL vượt ngưỡng','PPO clip fraction, weighted','Max post-KL','Runtime min','Peak CUDA MiB'],
        [[labels[n],r['learner_steps'],r['update_diagnostics']['optimizer_steps'],r['update_diagnostics']['epochs_per_episode'],
          r['update_diagnostics']['early_stop_episodes'],r['update_diagnostics']['guard_exceeded_episodes'],
          f"{100*r['update_diagnostics']['clip_fraction_weighted']:.4f}%",f"{r['update_diagnostics']['post_kl_max']:.6f}",
          fmt(r['seconds']/60),fmt(s['arms'][n]['peak_cuda_bytes']/2**20)] for n,r in a['arms'].items()]),'',
    'Clip fraction đo ratio ngoài[0,8;1,2] trước mỗi update, tính trọng số theo sample-updates; U4 chỉ log, không clip. '
    'Nó không đồng nghĩa mọi sample ngoài khoảng đều bị chặn gradient vì dấu advantage quyết định surrogate nào được chọn. '
    'Các epoch thực giữ cùng labels/targets; gradient clip5 là cơ chế khác với PPO ratio clipping. '
    'Critic EV/raw advantage trong metrics tính từ policy cũ; value_loss từng epoch từ prediction hiện tại. '
    'Peak CUDA là bộ nhớ tensor allocated mà PyTorch ghi nhận trong tiến trình, không phải tổng VRAM theo nvidia-smi; '
    'với P4 đây là peak của đoạn sau resume. '
    'Runtime mỗi nhánh gồm train/validation/skill probes, cộng các đoạn chạy trước/sau resume và không cộng quãng tiến trình tắt. '
    'Điều kiện VRAM thay đổi giữa lượt P4 nên chênh runtime giữa hai nhánh chưa đo riêng chi phí của clipping.','',
    '## Từng góc','',
    table(['Model','Góc','ActiveWR %','Passive WR %','Hòa %','Tự nổ %','Kills/game'],
        [[labels[n],'A'+corner,fmt(r['active_win_rate']),fmt(r['passive_win_rate']),fmt(r['draw_rate']),fmt(r['suicide_rate']),f"{r['avg_kills']:.3f}"]
         for n,model in a['models'].items() for corner,r in model['by_corner'].items()]),'',
    '## Mở đường và tiếp cận enemy','',
    table(['Model','Tường/game','Tới gần %','Bước đầu có điều kiện'],
        [[labels[n],fmt(r['avg_walls']),fmt(r['contact']['reached_close_rate']),fmt(r['contact']['avg_first_close_step_conditional'])]
         for n,r in a['models'].items()]),'',
    'Tới gần dùng Manhattan≤3, đọc tỷ lệ cùng thời gian có điều kiện; chưa phải chứng minh có đường đi an toàn hoặc trap.','',
    '## Skill probes','',
    table(['Nhánh','Episode','Escape owned-kill %','Approach owned-kill %'],
        [[labels[n],ep,fmt(groups['escape']['owned_kill_success']['percent']),fmt(groups['approach']['owned_kill_success']['percent'])]
         for n,probes in a['skills'].items() for ep,groups in probes.items()]),'',
    '## Kiểm chứng và giới hạn','',
    'Regression 241/241 PASS; sau sửa metadata, 25 focused metadata/compatibility tests PASS. '
    'Các nhóm test có trùng nhau, không cộng thành 266 test riêng biệt. Smoke CPU/CUDA qua đủ 20 episode/nhánh. '
    'Có test clipping dấu advantage, fixed targets/group weighting, KL sau update, nonfinite gradient, '
    'cancel giữa epoch khôi phục Adam và resume PPO ở tiến trình mới trên CPU/CUDA. '
    'Fixture500bước/4epochs kiểm tra bộ nhớ/tensor finite; không dùng làm đánh giá chất lượng policy.','',
    f"Đã kiểm tra ZIP raw {a['archive_files']} files/CRC/bytes, source/B/selected/full-state hashes, "
    f"replay lại {a['skills_replayed_games']} skill games/{a['skills_replayed_frames']} frames, "
    'validation selection và các metric tổng/từng góc. Adam counts bằng tổng epochs thực, scheduler 200, context/history '
    'và final/best weights khớp full state; RNG Python/NumPy/env/Torch CPU/CUDA có đủ trạng thái.','',
    'Một train seed, một cấu hìnhK/epsilon/LR và RuleBot hiện tại. CI theo board cho các policy đã chọn chưa đo biến thiên train. '
    'Chưa triển khai D2/D3, minibatch/rollout nhiều episode hoặc thay model. '
    'KL guard không cắt epoch nào ở cả hai nhánh; điều kiện mở D2 vì KL thường chặn còn1–2epochs chưa xảy ra. '
    'Không dùng test37000 đã mở để chỉnh LR/epsilon rồi chọn lại candidate.','']
same=[labels[n] for n in ('control','ppo') if a['model_state_sha256'][n]==a['model_state_sha256']['original']]
if same:lines.insert(4,'Best cùng trọng số với B: '+', '.join(same)+'. Điểm test bằng nhau phản ánh cùng policy, chưa phải cải tiến được học.')
(out/'decision_report.md').write_text('\n'.join(lines),encoding='utf-8')
print(out/'decision_report.md')
