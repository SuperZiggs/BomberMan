"""Render the reviewed R0 report and complete per-component tables."""
import csv,json
from pathlib import Path
import sys
out=Path(sys.argv[1]).resolve()
summary=json.loads((out/'summary.json').read_text())
stats=json.loads((out/'statistics.json').read_text())
review=json.loads((out/'review_results.json').read_text())
rankings=json.loads((out/'paired_outcome_rankings.json').read_text())
assert summary['status']=='complete' and review['status']=='reviewed'
names={'original':'B','mc_final':'MC final200','gae_final':'GAE final200'}
base='/'+out.as_posix()
link=lambda file:f'[{file}]({base}/{file})'
f=lambda x,places=3:'—' if x is None else f'{x:.{places}f}'
columns=('model','group','games','component','sum','mean','median','p95','min','max','gross_abs_share','net_abs_share')
with (out/'component_statistics.csv').open('w',encoding='utf-8',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader()
    for model,report in stats.items():
        for group,data in {'all':report['all'],**report['by_outcome']}.items():
            for component in ('time','item','wall','owned_kill','active_win','passive_win_env','passive_win_cancellation','death','self_blast'):
                value=data.get('components',{}).get(component)
                row={'model':model,'group':group,'games':data['games'],'component':component}
                if value is not None:
                    row.update(value,sum=data['component_sums'][component],gross_abs_share=data['gross_abs_shares'].get(component),net_abs_share=data['net_abs_shares'].get(component))
                writer.writerow(row)
lines=['# T4-R0 — audit reward hoàn tất', '',
'**R0 DONE/REVIEWED. Giữ checkpoint B và các hệ số reward hiện tại.** Không phát hiện sai lệch accounting trong các transition đã kiểm tra. Có cơ sở thử R1 như một ablation riêng; chưa chứng minh reward là nguyên nhân chính của hòa.', '',
f"Đã audit300 trận trên validation đã dùng:100/model,50board seeds21000–21049 và cảA0/A1. CPU một thread,9×9/2agents/max500,RuleBot,CNN lấy mẫu qua mask cũ. Thời gian runner {summary['elapsed_seconds']/60:.2f} phút. Không train, chọn model, mở test mới hoặc thay checkpoint.", '',
'## 1. Cohort và parity', '', '| Checkpoint | Active win | Passive win | Timeout | Chết do đối thủ | Tự nổ |', '|---|---:|---:|---:|---:|---:|']
for model,r in stats.items():
    g=r['by_outcome']
    lines.append(f"| {names[model]} | {g['active_win']['games']} | {g['passive_win']['games']} | {g['timeout']['games']} | {g['enemy_death']['games']} | {g['self_blast']['games']} |")
lines += ['', 'Các tỷ lệ win/active-win/draw/suicide và trung bình kills/bombs/items/steps khớp validation B1 ở đúng checkpoint: B dùngepisode0; hai final dùngepisode200. Đây là dữ liệu validation để chẩn đoán, không phải test mới hoặc bằng chứng thay model. Các model dùng chungboard/corner;300game không phải300board độc lập.', '',
'## 2. Vì sao timeout vẫn có reward dương?', '', '| Checkpoint | Timeout n | Reward mean / median / p95 | Time mean | Wall mean | Item mean | Reward dương |', '|---|---:|---|---:|---:|---:|---:|']
for model,r in stats.items():
    d=r['by_outcome']['timeout'];v=d['stats']['shaped_reward'];c=d['components']
    lines.append(f"| {names[model]} | {d['games']} | {f(v['mean'])} / {f(v['median'])} / {f(v['p95'])} | {f(c['time']['mean'])} | {f(c['wall']['mean'])} | {f(c['item']['mean'])} | {d['positive_reward_games']}/{d['games']} |")
lines += ['', 'Mọi timeout trong cohort có reward dương. Với B, time−0,5 bị vượt bởi wall≈+1,335 và item≈+0,081. Đây là vấn đề cần kiểm tra về mức thưởng phụ; kết quả không có nghĩa policy thích hòa hơn mọi lần thắng. Thắng chủ động vẫn nhận tổng reward lớn hơn rõ rệt trong cohort.', '',
'## 3. Hành vi: đã tới đối thủ nhưng chưa hạ được', '', '| Checkpoint | Timeout từng tới khoảng cách≤3 | Bom pressure>0, toàn timeout | Bom/100bước cuối | Pressure>0 trong bom cuối | Wall reward 100bước cuối | Tỷ lệ STOP cuối |', '|---|---:|---:|---:|---:|---:|---:|']
for model,r in stats.items():
    d=r['by_outcome']['timeout'];late=review['timeout_last100_summaries'][model];ls=late['stats']
    lines.append(f"| {names[model]} | {d['reached_close_games']}/{d['games']} | {f(100*d['pressure_bomb_fraction'],2)}% | {f(ls['bombs']['mean'])} | {f(100*late['pressure_bomb_fraction'],1)}% | {f(ls['wall_reward']['mean'])} | {f(100*ls['stop_fraction']['mean'],2)}% |")
lines += ['', 'Trong100bước cuối timeout, wall reward bằng0 ở cả ba model; mean reward khoảng−0,1. MC final có duy nhất một lần nhặt item trong toàn bộ các cửa sổ cuối, khiến mean reward−0,0993. CNN vẫn đặt khoảng10–11bom/100bước cuối, tất cả có pressure dương theo phép đo hiện tại. Dữ liệu không ủng hộ mô tả agent chỉ đứng farm gỗ cuối trận; các chỉ số này gợi ý kiểm tra chất lượng thế bom và khả năng chuyển sức ép thành kill; R0 chưa chỉ ra nguyên nhân chiến thuật cụ thể.', '',
'Pressure chỉ đo giảm refuges với bom đã biết và vị trí đối thủ hiện tại. Pressure>0 không chứng minh trap, kill hoặc combo thực thi được trước RuleBot. Khoảng cách Manhattan≤3 và hướng BFS là các proxy; không chứng minh an toàn toàn bộ đường đi. Bảng chi tiết và nguyên liệu: '+link('timeout_last100.json')+', '+link('games_original.json')+'.', '',
'## 4. Return đã chiết khấu và thứ tự kết quả', '', 'R0 tính thêm return tại bước đầu G0 bằng đúng gamma0,99 từ B1. Đây là MC return của critic/control; không giả định GAE actor dùng cùng target advantage. Tổng component returns được kiểm tra khớp G0.', '',
'| Checkpoint | G0 mean thắng chủ động | G0 mean timeout | Wall G0 mean timeout |', '|---|---:|---:|---:|']
for model in stats:
    d=review['discounted_summaries'][model]
    lines.append(f"| {names[model]} | {f(d['active_win']['return_t0']['mean'])} | {f(d['timeout']['return_t0']['mean'])} | {f(d['timeout']['components']['wall']['mean'])} |")
lines += ['', 'Trung bình G0 của thắng vẫn cao hơn timeout. Tuy nhiên có hai ví dụ trên cùng seed/góc mà return timeout cao hơn trajectory thắng, khi tính cả reward phụ và thời điểm nhận reward. Đây là ví dụ accounting, chưa chỉ ra nguyên nhân thay đổi policy. Toàn bộ88cặp thắng/timeout đủ điều kiện giữa các checkpoint được giữ, không chỉ hai ví dụ:', '', '| Seed/góc | Model thắng / G0 | Model timeout / G0 | Gap timeout−win sau giảm nửa wall |', '|---|---|---|---:|']
for r in rankings:
    if r['observed_gap']>review['positive_tolerance']:
        lines.append(f"| {r['seed']}/A{r['aid']} | {names[r['winning_model']]} / {f(r['win_G0'],6)} | {names[r['timeout_model']]} / {f(r['timeout_G0'],6)} | {f(r['wall_half_gap'],6)} |")
lines += ['', 'Hai cặp không phải hai thí nghiệm độc lập; cùngboard được dùng qua nhiều model. Giảm nửa wall xóa một đảo thứ tự và làm nhỏ đảo còn lại; giảm nửa item không xóa hai ví dụ này. Không suy ra thay đổi ActiveWR từ phép tính đó. Dữ liệu đầy đủ: '+link('discounted_returns.json')+', '+link('paired_outcome_rankings.json')+'.', '',
'## 5. What-if trên cùng action tape', '', '**Không train hoặc đổi hành động.** Chỉ tính lại reward nếu wall0,15→0,075 hoặc item0,05→0,025. Xem đây là độ lớn thay đổi mục tiêu, không phải dự báo hành vi hay win-rate. Dùng tolerance1e−10 khi phân loại dương/0/âm để không tính nhiễu số thực là thưởng.', '',
'| Checkpoint | Mean timeout hiện tại | Nếu wall0,075 | Dương /≈0/âm | Nếu item0,025 |', '|---|---:|---:|---|---:|']
for model,r in review['fixed_action_timeout_accounting'].items():
    wall=r['wall_0.075']
    lines.append(f"| {names[model]} | {f(r['observed']['distribution']['mean'])} | {f(wall['distribution']['mean'])} | {wall['positive_count']} / {wall['near_zero_count']} / {wall['negative_count']} | {f(r['item_0.025']['distribution']['mean'])} |")
lines += ['', 'Wall bonus chiếm phần lớn reward phụ ở timeout; giảm item một nửa tác động nhỏ hơn nhiều. Giảm nửa wall vẫn để phần lớn timeout có reward dương, nên R1 là phép thử mức thưởng, không phải cách ép agent thắng hoặc giải quyết toàn bộ chiến thuật.', '',
'## 6. Mean/median/p95 của mọi thành phần', '', 'Bảng dưới dùng toàn100game/model, gồm cả0 khi sự kiện không xảy ra. Tất cả nhóm kết quả, sum/min/max và tỷ trọng tuyệt đối nằm trong '+link('component_statistics.csv')+' và '+link('statistics.json')+'. Nhóm tự nổ có n=0 nên CSV để trống thống kê có điều kiện; không coi thiếu mẫu là penalty0. Mỗi model chỉ có1trận chết do đối thủ và đó là cùngseed/góc, không đủ để đề xuất chỉnh death penalty.', '']
for model,r in stats.items():
    lines += ['### '+names[model], '', '| Component | Mean | Median | P95 |', '|---|---:|---:|---:|']
    for key,d in r['all']['components'].items(): lines.append(f"| {key} | {f(d['mean'],4)} | {f(d['median'],4)} | {f(d['p95'],4)} |")
    lines.append('')
lines += ['Không chia reward phụ cho tổng reward có thể gần0/âm. Gross shares dùng tổng trị tuyệt đối các component theoepisode. Net shares loại cặp passive+10/−10 triệt tiêu; các khoản raw vẫn được lưu để kiểm chứng.', '',
'## 7. Biểu đồ và kiểm chứng', '',
f'![Reward theo kills và active-win indicator]({base}/reward_scatter.png)', '',
'300điểm được giữ nguyên ở y; jitter x xác định≤0,035 chỉ tránh chồng điểm. Active-win ở scatter là biến0/1; ActiveWR là tỷ lệ tổng hợp. Mỗi model có100game/50board, không coi hai góc là độc lập. Không smoothing, loại outlier hoặc CI. Reward có bonus kill/win theo định nghĩa nên tương quan giữa reward và win không chứng minh nguyên nhân hay chất lượng thuật toán.', '',
link('figure_data.csv')+'; '+link('reward_scatter.svg')+'; '+link('figure_manifest.json')+'. PNG3600×2100, nền trắng, cùngthang y, dùng cả marker và màu. Đã xem PNG để kiểm tra clipping và chữ. Palette audit có0cảnh báo contrast màu đặc trên trắng và2cặp cần phân biệt khi grayscale; marker khác nhau cung cấp mã hóa bổ sung. Không tuyên bố chứng nhận accessibility hoặc chuẩn journal.', '',
f"Review kiểm tra {review['learner_steps_checked']:,}transition: shaped reward mỗi bước khớp bit-exact công thức cũ; tổng component sai số tối đa{review['max_component_error']:.3g}, dưới ngưỡng1e−10. Replay lại{review['timeout_cases_replayed']}timeout/{review['timeout_frames_replayed']:,}frames;300hàng dữ liệu biểu đồ khớp game rows. ZIP raw{review['archive_files']}files/CRC/bytes,source11,checkpoint vàparent input hashes qua. Nguồn thô không sửa sau review.", '',
'Bộ regression188PASS/1SKIP(CUDA không khả dụng), tổng189tests. Có test default/logger CPU bit-parity với source-before cho actions,rewards,metrics vàweights; useful items/cap,late kill/chain,multiple kills,passive cancellation,mixed lingering self-flames,death+wall vàpost-deathFFA. Smoke6game/max20 qua; auditfull300game khớp validation B1 vàkhông cập nhật optimizer. CUDA logger vẫn chưa được chạy trên runtime hiện tại.', '',
'## 8. Quyết định và bước tiếp theo', '',
'Giữ checkpoint B và toàn bộ coefficient hiện tại. Đánh dấu R0DONE/REVIEWED. **Mở R1 như ablation một biến:** wall bonus0,15 so0,075, giữ item/kill/win/death/time/mask/teacher/MC raw. Cần runner đối chứng, cùngwarm-startB,validation chọnbest vàtest mới đã đăng ký; chưa trainR1 trongR0. Không ghép item,terminal penalty,GAE/PPO hoặc curriculum vào cùng lượt.', '',
'R1 có cơ sở để kiểm tra mức thưởng phụ và vài trường hợp thứ tự return lệch mục tiêu ActiveWR. Bằng chứng hiện tại không chứng minh farm gỗ là nguyên nhân chính của hòa, hoặc giảm wall bonus sẽ nâng ActiveWR. Agent vẫn giao tranh; chất lượng trap/combo là điểm nghẽn riêng còn tồn tại.', '',
'Giới hạn: audit mô tả trên50validation board đã xem và3checkpoint, không phải causal ablation. Có1death case/model,0self-blast; không suy ra an toàn hay penalty tối ưu. Runtime phụ thuộc máy; không dùng thời gian này để so tốc độ MC/GAE. Không mở thêm heldout hoặc dùng35000 chọn model. Các model tham chiếu vẫn nằm trongworkspace; ZIP raw không đóng gói checkpoint weights.', '',
'Files: '+link('summary.json')+', '+link('config.json')+', '+link('review_results.json')+', '+link('timeout_cases.json')+'. Script review và report writer lưu cùngthư mục; output phân tích được thêm sau ZIP raw.']
text='\n'.join(lines)+'\n'
for old,new in {'audit300':'audit 300','100trận':'100 trận','50board':'50 board','cảA0/A1':'cả A0/A1','CPU một thread,9':'CPU một thread, 9','episode0':'episode 0','episode200':'episode 200','cùngboard':'cùng board','300game':'300 game','100bước':'100 bước','1trận':'1 trận','gamma0,99':'gamma 0,99','88cặp':'88 cặp','12timeout':'12 timeout','9component':'9 component','37files':'37 files','11,checkpoint':'11, checkpoint','tolerance1e':'tolerance 1e','sum component':'tổng component','1death':'1 death','0self':'0 self','cùngseed':'cùng seed','mean reward−':'mean reward −','vàkhông':'và không','nhưng57':'nhưng 57'}.items():text=text.replace(old,new)
# Keep the report readable without changing paths or raw data.
for old,new in {'300trận':'300 trận','100/model':'100/model','50board':'50 board','cảA0/A1':'cả A0/A1','mỗi model chỉ có1':'mỗi model chỉ có 1','100game':'100 game','300game':'300 game','100bước':'100 bước','100trận':'100 trận','100%':'100%','88cặp':'88 cặp','2cặp':'2 cặp','188PASS':'188 PASS','1SKIP':'1 SKIP','189tests':'189 tests','6game':'6 game','300điểm':'300 điểm','300hàng':'300 hàng','0cảnh':'0 cảnh','2cặp':'2 cặp','cùngthang':'cùng thang','qua. Nguồn':'qua. Nguồn','cùngwarm':'cùng warm','1death':'1 death','0self':'0 self','3checkpoint':'3 checkpoint','không cập nhậtoptimizer':'không cập nhật optimizer','R0DONE':'R0 DONE','R1 trongR0':'R1 trong R0','thư mục':'thư mục','cùngthư':'cùng thư'}.items():text=text.replace(old,new)
(out/'R0_report.md').write_text(text,encoding='utf-8')
print(out/'R0_report.md')
