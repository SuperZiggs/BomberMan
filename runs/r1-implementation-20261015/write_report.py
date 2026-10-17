"""Write the R1 decision report from independently recomputed results."""
import json
from pathlib import Path
import sys

OUT = Path(sys.argv[1]).resolve()
read = lambda path: json.loads(path.read_text(encoding='utf-8-sig'))
a, cfg, summary = (read(OUT / name) for name in ('analysis_results.json', 'config.json', 'summary.json'))
labels = {'original': 'B', 'control': 'Control 0,15', 'half_wall': 'Half wall 0,075'}
link = lambda name: f'[{name}](/D:/fpt/subject/BomberMan/runs/{OUT.name}/{name})'
fmt = lambda x: 'không có mẫu' if x is None else f'{x:.2f}'

def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                      *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]])

lines = ['# T4-R1 — thử giảm riêng thưởng phá tường', '',
    '**R1 DONE/REVIEWED.** ' + ('Gate PASS; cần replication trước khi thay B.' if all(a['gates'].values()) else
    '**Gate FAIL, giữ checkpoint B và reward hiện tại.**'), '',
    f"Đủ 200 episode và 200 Adam updates mỗi nhánh; validation ở0/100/200; 400 test games/model, "
    f"seeds36000–36199/cảA0-A1. Runner hoàn tất trong {a['elapsed_seconds']/60:.2f} phút. "
    'Các model dùng chung200 seed bàn; không coi1.200 game là1.200 bàn độc lập.', '',
    '## Thiết kế và kiểm tra', '',
    'Hai nhánh warm-start từ B với Adam mới, train seed2026101505. Control wall bonus0,15; treatment0,075. '
    'Cùng teacher local, reward item/kill/win/death/time, mask, MC returns, advantage không chuẩn hóa, '
    'observation12 kênh, LR1e-4 và mọi loss khác. Đây là thay đổi reward dùng train; physics và checkpoint B không đổi.', '',
    'Preflight định giá lại39.338 transition B của R0: delta chỉ bằng−0,075×số tường đủ điều kiện; '
    'các thành phần khác giữ nguyên. Bộ regression200/200 PASS, smoke CPU/CUDA đầy đủ20episode/nhánh. '
    'Eval/contact logging đối chiếu với evaluator trước R1: actions, kết quả và RNG giữ nguyên. '
    'Default caller cũ không bật logging mới.', '',
    '## Best chọn bằng validation', '',
    table(['Nhánh', 'Episode', 'ActiveWR %', 'Tự nổ %', 'Hòa %', 'Được chọn'],
        [[labels[arm], v['episode'], fmt(v['active_win_rate']), fmt(v['suicide_rate']), fmt(v['draw_rate']), v['selected']]
         for arm, data in a['arms'].items() for v in data['validations']]), '',
    'Best được chọn và khóa hash trước khi mở test. Final200 chỉ phục vụ validation/skill diagnostics; '
    'không chấm thêm final trên test để đổi candidate sau khi xem kết quả.', '',
    '## Test đã đăng ký trước', '',
    table(['Model', 'Best ep', 'ActiveWR %', 'Thắng thụ động %', 'Hòa %', 'Tự nổ %', 'Kills/game'],
        [[labels[name], 'gốc' if name == 'original' else a['arms'][name]['best_episode'],
          fmt(r['active_win_rate']), fmt(r['passive_win_rate']), fmt(r['draw_rate']),
          fmt(r['suicide_rate']), fmt(r['avg_kills'])] for name, r in a['models'].items()]), '',
    table(['Half wall so với', 'Delta ActiveWR (pp)', 'CI95% (pp)', 'Số seed bàn'],
        [[labels[name], fmt(r['difference_pp']), f"[{fmt(r['ci95_pp'][0])}; {fmt(r['ci95_pp'][1])}]", r['board_pairs']]
         for name, r in a['comparisons'].items()]), '',
    'CI percentile bootstrap10.000 lần theo cặp seed bàn, giữ cả hai góc trong mỗi cụm. '
    'Gate cần delta≥5pp so với cả B và control, hai cận dướiCI>0, tự nổ quan sát không cao hơn cả hai. '
    f"Các điều kiện recompute: {a['gates']}.", '',
    '## Từng góc', '',
    table(['Model', 'Góc', 'ActiveWR %', 'Thắng thụ động %', 'Hòa %', 'Tự nổ %', 'Kills/game'],
        [[labels[name], 'A'+corner, fmt(r['active_win_rate']), fmt(r['passive_win_rate']),
          fmt(r['draw_rate']), fmt(r['suicide_rate']), fmt(r['avg_kills'])]
         for name, model in a['models'].items() for corner, r in model['by_corner'].items()]), '',
    '## Mở đường và tiếp cận đối thủ trên test', '',
    table(['Model', 'Tường/game', 'Tường đủ bonus/game', 'Đã rời ô spawn %', 'Từng tới gần %', 'Bước đầu tới gần, có điều kiện'],
        [[labels[name], fmt(r['avg_walls']), fmt(r['avg_walls_rewarded']), fmt(r['contact']['left_spawn_rate']),
          fmt(r['contact']['reached_close_rate']), fmt(r['contact']['avg_first_close_step_conditional'])]
         for name, r in a['models'].items()]), '',
    'Tới gần nghĩa là Manhattan≤3 tới một đối thủ còn sống. Phải đọc tỷ lệ tới gần cùng thời gian có điều kiện; '
    'không dùng0 cho các trận chưa tới gần. Rời ô spawn và Manhattan là proxy, chưa chứng minh đường đi an toàn '
    'hoặc chiến thuật trap. Tường/game là attribution; tường đủ bonus loại tick learner đã chết.', '',
    '## Hành vi trong train', '',
    table(['Nhánh', 'Active wins/200', 'Tự nổ/200', 'Tường/ep', 'Tường đủ bonus/ep', 'Đã tới gần %', 'Bước đầu, có điều kiện'],
        [[labels[name], r['training_active_wins'], r['training_suicides'], fmt(r['walls_mean']),
          fmt(r['eligible_walls_mean']), fmt(r['contact']['reached_close_rate']),
          fmt(r['contact']['avg_first_close_step_conditional'])] for name, r in a['arms'].items()]), '',
    table(['Nhánh', 'Time reward mean', 'Wall mean', 'Item mean', 'Owned kill mean', 'Active win mean'],
        [[labels[name], *[fmt(r['reward_component_means'].get(key, 0))
                         for key in ('time', 'wall', 'item', 'owned_kill', 'active_win')]]
         for name, r in a['arms'].items()]), '',
    'Mean trên toàn200episode, gồm0 khi sự kiện không xảy ra. Tổng component khớp reward từng episode; '
    'wall/coefficient là số nguyên và không vượt số tường attribution. Reward giữa hai nhánh khác hệ số '
    'nên không dùng reward train cao/thấp để chọn winner. Trajectory thay đổi sau cập nhật; '
    'không diễn giải chênh lệch reward quan sát trong train như phép đối chiếu cùng action tape.', '',
    '## Kỹ năng trên tình huống validation cố định', '',
    table(['Nhánh', 'Episode', 'Escape owned-kill success %', 'Approach owned-kill success %'],
        [[labels[name], episode, fmt(groups['escape']['owned_kill_success']['percent']),
          fmt(groups['approach']['owned_kill_success']['percent'])]
         for name, probes in a['skills'].items() for episode, groups in probes.items()]), '',
    'Đây là bộ tình huống validation với đối thủ static đã dùng, giúp chẩn đoán kỹ năng; '
    'không thay thế full-game RuleBot test và không mở wood_branch/test31000.', '',
    '## Review và quyết định', '',
    f"ZIP raw {a['archive_files']} files/CRC/bytes, source snapshot, checkpoint và input R0 hashes qua. "
    f"Đã replay {a['skills_replayed_games']} skill games/{a['skills_replayed_frames']} frames, "
    'tính lại validation selection, toàn bộ aggregate/từng góc, contact/walls và bootstrap/gate. '
    'Các best có cùng model-state hash với B phải cho kết quả từng game giống B.', '',
    'Giữ hệ số mặc định0,15 và checkpoint B. ' +
    ('Có tín hiệu đủ gate để đăng ký replication nhiều train seed; chưa thay model.' if all(a['gates'].values()) else
     'Pilot chưa đạt tiêu chí thay model. Không tự chuyển sang wall bonus0 hoặc ghép sửa item/GAE/PPO. '
     'Bước tiếp theo cần dựa vào validation, kỹ năng và regression ở các bảng trên; không tiếp tục giảm reward chỉ vì reward train thấp hơn.'), '',
    'Giới hạn: một train seed, một mức treatment, test200 seed bàn và RuleBot hiện tại. '
    'CI mô tả chênh lệch policy được chọn trên seed bàn, không đo biến thiên giữa các lượt train. '
    'Không kết luận mọi mức giảm reward đều vô ích hoặc reward chắc chắn là nguyên nhân hòa.', '',
    'Nguồn dữ liệu: ' + ', '.join(link(name) for name in ('config.json', 'summary.json', 'analysis_results.json',
        'preflight.json', 'selected_models.json')) + '.']

same = [labels[name] for name in ('control', 'half_wall')
        if a['model_state_sha256'][name] == a['model_state_sha256']['original']]
if same:
    lines.insert(4, 'Best có cùng trọng số với B: ' + ', '.join(same) +
        '. Điểm test giống nhau phản ánh cùng policy; không phải một cải tiến đã học được.')
(OUT / 'decision_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(OUT / 'decision_report.md')
