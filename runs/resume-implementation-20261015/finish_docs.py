"""Record the completed resumable R1 pilot; refuse partial attempts."""
import json
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1]).resolve()
read = lambda name: json.loads((OUT / name).read_text(encoding='utf-8-sig'))
summary, analysis = read('summary.json'), read('analysis_results.json')
assert summary['status'] == 'complete' and summary['full_protocol'] and summary['hashes_unchanged']
assert set(analysis['training_state_checks']) == {'control', 'half_wall'}
gate = 'PASS' if all(analysis['gates'].values()) else 'FAIL'
rates = '/'.join(f"{analysis['models'][name]['active_win_rate']:.2f}%" for name in ('original', 'control', 'half_wall'))
best = '/'.join(str(analysis['arms'][name]['best_episode']) for name in ('control', 'half_wall'))
report = f'/D:/fpt/subject/BomberMan/runs/{OUT.name}/decision_report.md'
decision = 'cần replication trước khi thay B' if gate == 'PASS' else 'giữ B và wall bonus0,15'
status = (f'R1 DONE/REVIEWED: `{OUT.name}`. Đủ200episode/200Adam mỗi nhánh từB, '
    f'400test/model trên36000–36199/cảA0-A1. Best control/half_wall={best}; '
    f'ActiveWR B/control/half_wall={rates}. Gate{gate}; {decision}. '
    f'Runtime{analysis["elapsed_seconds"]/60:.2f}phút. [Báo cáo]({report}).')
feature = ('Phase C có `training_state.pt` gồm model/Adam/scheduler/global episode/best guard/history, '
    'RNG Python/NumPy/env/Torch CPU/CUDA và context log. Atomic temp+fsync+replace, OS writer lock, '
    '`run_reward_pilot.py --resume` khôi phục episode đã xác nhận; không reset schedule hoặc chọn lại model test. '
    '212test riêng biệt PASS; có CPU/CUDA exact resume trong tiến trình mới, interrupted episode/validation '
    'và torn-log recovery. `best/final_phaseC.pt` giữ dạng weights để chơi; loader từ chối nhầm full-state file. '
    'Resume yêu cầu cùng source/config/runtime/thiết bị; hiện chỉ hỗ trợ Phase C, deadline R1 gốc vẫn giữ.')

for name, heading in [('README.md', 'Phase C full-state resume và R1 hoàn tất'),
                      ('IMPROVEMENTS.md', '20. Phase C resume đầy đủ và R1 rerun'),
                      ('RESULTS.md', 'T4-R1 resumable — đánh giá đủ protocol')]:
    path = ROOT / name
    text = path.read_text(encoding='utf-8-sig')
    assert '\n## ' + heading + '\n' not in text
    path.write_text(text.rstrip() + '\n\n## ' + heading + '\n\n' + status + '\n\n' + feature + '\n', encoding='utf-8')

path = ROOT / 'PLAN_BOMBERMAN.md'
text = path.read_text(encoding='utf-8-sig')
old = next(line for line in text.splitlines() if line.startswith('| T4 |'))
text = text.replace(old, f'| T4 | Advantage, GAE, PPO và reward | A1/B1 gate FAIL; B0 PASS; R0 DONE; R1 gate {gate}; giữ B | R1 đủ200/nhánh,400test/model;212tests;full-state resume | `{OUT.name}/decision_report.md`, `REWARD_PILOT_GUIDE.md` |')
start = text.index('**Rerun RUNNING 15/10:** `runs/' + OUT.name + '`')
end = text.index('\n\n', start)
text = text[:start] + status + '\n\n' + feature + text[end:]
text = text.replace('- [ ] Pilot 200 episode; so kills, draw, ActiveWR, thời gian tiếp cận enemy và số tường phá.',
    '- [x] Pilot đủ200episode/nhánh;400test/model; báo kills/draw/ActiveWR/contact/walls và từng góc; bootstrap theo seed bàn.')
anchor = next(line for line in text.splitlines() if line.startswith('| reward-full-20261015-163227 |'))
entry = (f'| {OUT.name} | T4-R1 rerun/resume | B/source/R0 inputs/full states verified | '
    f'Wall0,15 vs0,075;MC raw/local;deterministic cảhai |2026101505;200ep/200Adam mỗi nhánh;'
    f'{analysis["elapsed_seconds"]/60:.2f}phút | val21000/0-100-200;test36000–36199,400/model | '
    f'Best{best};ActiveWR B/control/half={rates};384skill replays | DONE/REVIEWED;gate{gate};{decision};212testsPASS | '
    f'`runs/{OUT.name}/decision_report.md` |')
assert '| ' + OUT.name + ' |' not in text
text = text.replace(anchor, anchor + '\n' + entry)
path.write_text(text, encoding='utf-8')

path = ROOT / 'REWARD_PILOT_GUIDE.md'
text = path.read_text(encoding='utf-8-sig')
start = text.index('**Rerun có full-state checkpoint đang chạy:**')
end = text.index('\n\n', start)
text = text[:start] + status + text[end:]
text = text.replace('**Test36000–36199 chưa mở. Attempt mới phải train lại cả hai nhánh từB, dùng output mới.**',
    '**36000–36199 đã mở ở rerun này. Không chạy lại full trên dải này để chọn model khác.**')
text = text.replace('**36000–36199 đang được đánh giá trong rerun này. Không chạy lại full trên dải này để chọn model khác.**',
    '**36000–36199 đã mở ở rerun này. Không chạy lại full trên dải này để chọn model khác.**')
text = text.replace('Sau khi mở lại PowerShell:',
    'Chỉ dùng resume khi lượt còn dang dở, đã dừng tiến trình cũ và chưa hết deadline. Lượt DONE ở đầu trang không cần resume. Sau khi mở lại PowerShell:')
path.write_text(text, encoding='utf-8')
print(status)
