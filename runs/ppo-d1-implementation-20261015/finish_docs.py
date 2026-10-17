"""Record the reviewed D1 result once; refuse partial protocol or duplicate ledger entries."""
import json
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parents[2]
out=Path(sys.argv[1]).resolve()
read=lambda name:json.loads((out/name).read_text(encoding='utf-8-sig'))
summary,a=read('summary.json'),read('analysis_results.json')
assert summary['status']=='complete' and summary['full_protocol'] and summary['hashes_unchanged']
assert set(a['training_state_checks'])=={'control','ppo'}
passed=all(a['gates'].values())
gate='PASS' if passed else 'FAIL'
rates='/'.join(f"{a['models'][name]['active_win_rate']:.2f}%" for name in ('original','control','ppo'))
best='/'.join(str(a['arms'][name]['best_episode']) for name in ('control','ppo'))
steps='/'.join(str(a['arms'][name]['update_diagnostics']['optimizer_steps']) for name in ('control','ppo'))
decision='cần replication trước khi thay B' if passed else 'giữ checkpoint B'
link=f'/D:/fpt/subject/BomberMan/runs/{out.name}/decision_report.md'
status=(f'D1 DONE/REVIEWED: `{out.name}`. Đủ 200 episode/nhánh từ B, Adam updates U4/P4={steps}; '
    f'best episode U4/P4={best}. Test400/model, seed37000–37199/cảA0-A1: '
    f'ActiveWR B/U4/P4={rates}. Gate {gate}; {decision}. '
    f'Runtime {a["elapsed_seconds"]/60:.2f} phút. [Báo cáo]({link}).')
feature=('D1 giữ MC raw/critic MC/local teacher/reward cũ. U4/P4 cùng tối đa4 epochs và post-update '
    'KL guard target0,02/threshold0,03; chỉ khác clipping epsilon0,20. '
    '241 regression và25 focused metadata/compat tests PASS; CPU/CUDA smoke20episode/80updates mỗi nhánh đã review. '
    'Có 500-step/4-epoch memory fixture, hủy giữa epoch khôi phục Adam, exact PPO resume qua tiến trình mới CPU/CUDA, '
    'full-state/source/ZIP/locked-model guards. CNN inference và Actor-Critic mặc định giữ nguyên.')
p=ROOT/'PPO_PILOT_GUIDE.md'
s=p.read_text(encoding='utf-8-sig')
start=s.index('**D1 pilot RUNNING:**')
end=s.index('\n\n',start)
s=s[:start]+status+'\n\n'+feature+s[end:]
s=s.replace('# Chỉ dùng dải test37000 khi chưa mở; lượt đã hoàn tất không cần chạy lại.',
    '# Test37000–37199 đã mở trong lượt DONE ở đầu trang; không chạy lại để chọn model khác.')
s=s.replace('Nếu tiến trình bị ngắt, sau khi mở lại PowerShell:',
    'Chỉ resume lượt đang dở và còn deadline; lượt DONE không cần resume. Nếu tiến trình bị ngắt, sau khi mở lại PowerShell:')
p.write_text(s,encoding='utf-8')
p=ROOT/'PLAN_BOMBERMAN.md'
s=p.read_text(encoding='utf-8-sig')
lines=s.splitlines()
lines[2]=(f'Cập nhật: MOVE/A1/B1/R1 đã review/gate FAIL; B0/D0 PASS tính đúng. D1 DONE/REVIEWED, gate {gate}; '
          f'{decision}. D2/D3 và các nhánh reward khác giữ điều kiện mở riêng; chưa triển khai tự động từ test D1.')
s='\n'.join(lines)+'\n'
start=s.index('**D1 pilot RUNNING:**')
end=s.index('\n\n',start)
s=s[:start]+status+'\n\n'+feature+s[end:]
start,end=s.index('#### D1 —'),s.index('#### D2 —')
section=s[start:end].replace('- [ ] So U4/P4','- [x] So U4/P4')
anchor='D1 implementation/tests đã qua:'
if anchor in section:
    x=section.index(anchor)
    y=section.index('\n\n',x)
    section=section[:x]+('D1 đủ train/validation/skill/test; actual updates, group targets và KL diagnostics đã audit. '
        'Một train seed chưa đủ kết luận ổn định. Xem báo cáo ở đầu mục này; không chọn lại checkpoint sau test.')+section[y:]
s=s[:start]+section+s[end:]
for prefix,new in (
    ('| T4 |',f'| T4 | Advantage, GAE, PPO và reward | D1 gate {gate}; {decision}; B0/D0 PASS | D1 đủ200ep/arm,400test/model; full-state verified | `runs/{out.name}/decision_report.md`, `PPO_PILOT_GUIDE.md` |'),
    ('| T4-D1 |',f'| T4-D1 | PPO clipping | U4 cùng4epochs/KL guard | Đủ200episode/arm/400test-model; best{best}; Adam{steps} | DONE/REVIEWED; gate {gate}; {decision} |')):
    old=next(line for line in s.splitlines() if line.startswith(prefix))
    s=s.replace(old,new)
old=next(line for line in s.splitlines() if line.startswith('**Đợt hiện tại:**'))
s=s.replace(old,f'**Đợt hiện tại:** D1 đã đóng, gate {gate}; {decision}. Mọi thử D2/D3 phải đăng ký riêng từ diagnostics, không sửa cấu hình hoặc chọn model lại trên test37000 đã mở.')
entry=(f'| {out.name} | T4-D1 U4/P4 | B/source/full-state verified | Clip0,20 vsunclip, cùng4epoch/KL0,02/MC raw/local | '
       f'2026101507;200ep/arm;Adam{steps};{a["elapsed_seconds"]/60:.2f}min | val21000;test37000–37199,400/model | '
       f'Best{best};ActiveWR B/U4/P4={rates};384skill replays | DONE/REVIEWED;gate{gate};{decision} | '
       f'`runs/{out.name}/decision_report.md` |')
assert '| '+out.name+' |' not in s
anchor=next(line for line in s.splitlines() if line.startswith('| ppo-d0-implementation-20261015 |'))
s=s.replace(anchor,anchor+'\n'+entry)
p.write_text(s,encoding='utf-8')
for name,heading in (('README.md','T4-D1 — U4/P4 hoàn tất'),('RESULTS.md','T4-D1 — paired clipping pilot'),
                     ('IMPROVEMENTS.md','22. D1 PPO clipping và resume theo episode')):
    p=ROOT/name
    s=p.read_text(encoding='utf-8-sig')
    assert '\n## '+heading+'\n' not in s
    p.write_text(s.rstrip()+'\n\n## '+heading+'\n\n'+status+'\n\n'+feature+'\n',encoding='utf-8')
print(status)
