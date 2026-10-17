"""Record the completed R1 pilot without changing archived run inputs."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1]).resolve()
analysis = json.loads((OUT / 'analysis_results.json').read_text(encoding='utf-8-sig'))
summary = json.loads((OUT / 'summary.json').read_text(encoding='utf-8-sig'))
assert summary['status'] == 'complete' and summary['full_protocol'] and summary['hashes_unchanged']
assert (OUT / 'decision_report.md').exists()
gate = 'PASS' if all(analysis['gates'].values()) else 'FAIL'
report = f'/D:/fpt/subject/BomberMan/runs/{OUT.name}/decision_report.md'
models, arms = analysis['models'], analysis['arms']
rates = '/'.join(f"{models[n]['active_win_rate']:.2f}%" for n in ('original', 'control', 'half_wall'))
best = '/'.join(str(arms[n]['best_episode']) for n in ('control', 'half_wall'))
decision = 'Cần replication trước khi thay B' if gate == 'PASS' else 'Giữ B và reward mặc định'
paragraph = (f'R1 DONE/REVIEWED: `{OUT.name}`, wall bonus0,15 so0,075; đủ200episode/nhánh và400test/model '
    f'(36000–36199/cảA0-A1). Best control/half_wall tại{best}; ActiveWR B/control/half_wall={rates}. '
    f'Gate{gate}; {decision}. 200regression testsPASS, smokeCPU/CUDA vàreview384skill replays qua. '
    f'[Báo cáo]({report}), [protocol](/D:/fpt/subject/BomberMan/REWARD_PILOT_GUIDE.md). '
    'Một train seed chưa đo biến thiên giữa các lượt train; không đổi reward chỉ vì tổng reward thấp hơn.')
for name, heading in [('README.md', 'R1: giảm riêng thưởng phá tường'),
                      ('IMPROVEMENTS.md', '19. T4-R1 — pilot wall bonus có đối chứng'),
                      ('RESULTS.md', 'T4-R1 — wall bonus 0,15 so 0,075')]:
    path = ROOT / name
    text = path.read_text(encoding='utf-8-sig')
    marker = '\n## ' + heading + '\n'
    assert marker not in text
    path.write_text(text.rstrip() + marker + '\n' + paragraph + '\n', encoding='utf-8')

path = ROOT / 'REWARD_PILOT_GUIDE.md'
text = path.read_text(encoding='utf-8-sig')
lines = text.splitlines()
lines[2:2] = [paragraph, '', '**36000–36199 đã mở. Không chạy lại full trên dải này để chọn model.**', '']
path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

path = ROOT / 'PLAN_BOMBERMAN.md'
text = path.read_text(encoding='utf-8-sig')
text = text.replace('R1 RUNNING; giữ B', f'R1 DONE/REVIEWED gate{gate}; giữ B')
text = text.replace('R1 regression200PASS, smokeCPU/CUDA qua, pilot đang chạy',
    f'R1 đủ200/nhánh,400test/model; ActiveWR B/control/half={rates}')
start = text.index('**RUNNING ngày15/10:** `runs/' + OUT.name + '`.')
end = text.index('\n\n', start)
text = text[:start] + paragraph + text[end:]
text = text.replace('- [ ] Pilot 200 episode; so kills, draw, ActiveWR, thời gian tiếp cận enemy và số tường phá.',
    '- [x] Pilot đủ200episode/nhánh; 400test/model; đã báo kills/draw/ActiveWR/contact/walls và từng góc, bootstrap theo seed bàn.')
anchor = next(line for line in text.splitlines() if line.startswith('| r0-full-20261015-153908 |'))
ledger = (f'| {OUT.name} | T4-R1 | B/source/R0 inputs hashes verified | Wall0,15 vs0,075; MC raw/local; logging chung | '
    f'2026101505;200ep/200Adam mỗi nhánh;{analysis["elapsed_seconds"]/60:.2f}phút | '
    f'val21000/0-100-200;test36000–36199,400/model | Best{best};ActiveWR B/control/half={rates};384skill replays | '
    f'DONE/REVIEWED;gate{gate};{decision};200testsPASS | `runs/{OUT.name}/decision_report.md` |')
assert '| ' + OUT.name + ' |' not in text
text = text.replace(anchor, anchor + '\n' + ledger)
path.write_text(text, encoding='utf-8')
print(paragraph)
