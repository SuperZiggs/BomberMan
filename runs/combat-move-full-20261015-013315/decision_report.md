# Certified combat MOVE — pilot đầy đủ, 15/10/2026

**Đã hoàn tất implementation,136 tests, smoke CPU/CUDA, hai lượt200 episode và1.200 trận test. Giữ checkpoint B: candidate có điểm tốt hơn trong lượt này nhưng chưa qua gate đã đăng ký.** Không đổi mặc định teacher, không thay model gốc hoặc chạy thêm test để tìm kết quả đẹp.

## Thay đổi và protocol

Mode`local_combat_move` giữ target action local. MOVE qua mask, tăng positive pressure, có next BOMB hợp lệ và escape trên clone tới hết bom/lửa đã biết thì chuyển từ navigation mean0,2 sang combat mean0,5. Các nhãn khác giữ nhóm cũ; mỗi bước chỉ một nhãn; CNN tiếp tục tự sample qua mask. Chuyển nhóm đồng thời đổi denominator của hai mean, không chỉ nhân một CE với2,5.

Certificate dùng enemy STOP/greedy escape hiện có, giới hạn32tick nguy hiểm; thất bại giữ navigation, không chứng minh không có lời giải. Có thể kiểm tra physics sau terminal để sống hết lửa; không bảo đảm CNN thực hiện prefix hoặc an toàn trước mọi move/counter-bomb tương lai. Không sửa env/mask/reward/12channels/CNN/MC returns/optimizer; runtime counter-bomb guard vẫn là nhánh riêng.

Config đăng ký trước test: cùngB `runs/pressure-C-next/final_phaseC.pt`,SHA256 `93bdf278...`;control local vàcandidate local_combat_move; PhaseC/RuleBot,9×9/2,max500,200ep/arm,seed2026101501,LR1e-4,CUDA/RTX3050. MộtAdam/200updates mỗiarm; validation100seed21000 ở0/100/200 chọnbest vớisuicideguardB0%. Skill64escape/approach/mốc chỉdiagnostic. TestB/controlbest/candidatebest400/model,200boards33000–33199×A0/A1,CPU1thread/categorical+mask,không teacher. Bootstrap10.000seed20261014 theo seed bàn giữ cảhai góc.

## Kiểm tra tính toàn vẹn

-136testsPASS: proof/purity/pressure/slots/collision/greedy failure, loss routing, actionCNN không override, actual promoted update CPU/CUDA vàdefaultlocal khớp before snapshot ởmọi tensor/action/reward.
-CPU/CUDA smoke4ep/arm vàdeadline đều đúng. Smokeonline0accepted; branch thực được test riêng trênpositive env. Smoke không được dùng làm bằng chứng tăng thắng.
-Preflight trên120unique navigation states C1 đã xem:28accepted,mean1,49ms/p955,22ms/max9,45ms; state/RNG không đổi. Đây là corpus static, không phải latency trênmọi state.
-ZIP43files CRC vàbyte-for-byte khớp từng file;source snapshots/root vàB/selected/final checkpoints khớp hashes. RawZIP không ghi đè sau review.
-400train metrics liên tục, learnerA0/A1 luân phiên, optimizer1..200; đủ6 validations/6skill probe records. Best vàfinal tensor digests khớp weights ởđúng probe episode.
-384stored skill games/8.884frames tái tạo masks/actions/positions/placements/kills/alive/labels/state-RNG; baseline64 đượccache vàghi cảhaiarm, nên chỉ320rollouts mới thực thi. Không xem chúng như384mẫu độc lập.
-Recompute1.200game rows/rates/averages/corners và200paired-board CIs khớp summary. Không mở31000 hoặcCNN test familywood_branch.

Review tái lập: [review_results.py](/D:/fpt/subject/BomberMan/runs/combat-move-full-20261015-013315/review_results.py), [analysis_results.json](/D:/fpt/subject/BomberMan/runs/combat-move-full-20261015-013315/analysis_results.json), [analysis_console.log](/D:/fpt/subject/BomberMan/runs/combat-move-full-20261015-013315/analysis_console.log).

## Kết quả test400/model

| Metric | B | Control best100 | Combat MOVE best100 |
|---|---:|---:|---:|
| ActiveWR |35% (140)|35,5% (142)|40,25% (161)|
| Win rate |40,25% (161)|42% (168)|45,5% (182)|
| Hòa |57,5% (230)|56% (224)|53% (212)|
| Tự nổ quan sát |1% (4)|0,75% (3)|0,25% (1)|
| Kills/game |0,35|0,355|0,4025|
| Thắng thụ động |5,25% (21)|6,5% (26)|5,25% (21)|
| Bombs/game |39,6875|39,375|35,6375|
| Items/game |1,7325|1,7575|1,7875|
| Steps/game |382,645|378,05|370,1625|

| Góc/metric | B | Control | Combat MOVE |
|---|---:|---:|---:|
| A0ActiveWR |33,5%|36,5%|42,5%|
| A1ActiveWR |36,5%|34,5%|38%|
| A0suicide |2% (4)|1,5% (3)|0,5% (1)|
| A1suicide |0|0|0|
| A0draw |56,5%|52,5%|49,5%|
| A1draw |58,5%|59,5%|56,5%|

Candidate có observed ActiveWR cao hơn, ít bom/hòa/tự nổ hơn trong sample. Không suy từ4/3/1deaths rằng phương pháp đã chứng minh giảm rủi ro; không có proof trước mọi counter-bomb. GócA0 cải thiện nhiều hơnA1, chưa có replication để biết đó là đặc điểm ổn định.

| So sánh | DeltaActiveWR |95%paired bootstrapCI | Gategain≥5pp | GateCI lower>0 |
|---|---:|---:|---|---|
| Candidate −B |+5,25pp|[-1,5;12]pp|PASS|FAIL|
| Candidate −control |+4,75pp|[-1,5;11]pp|FAIL|FAIL|

Observed suicide≤cảhai:PASS. **Gate tổngFAIL; `eligible_for_replication=false`, `decision=retain_original`.** CI chứa0 nên chưa xác nhận superiority; cũng không chứng minh hai phương pháp bằng nhau. Không nới threshold vì chỉ thiếu0,25pp so vớicontrol, không thêm games hoặc chọnfinal sau khi nhìn test.

## Validation, kỹ năng và tín hiệu học

| Checkpoint/mốc | ActiveWRvalidation | Suicide | Escape/32 | Approach/32 |
|---|---:|---:|---:|---:|
| B/ep0 cảhai |29%|0|32|28|
| Control100(best) |30%|0|32|28|
| Control200 |30%|0|32|30|
| Candidate100(best) |40%|0|32|30|
| Candidate200 |31%|0|32|30|

Best được chọn trước test, cảhaiep100. Candidatefinal giảm40→31% trênvalidation nhưnggiữskill approach30/32; không gọi đó là hồi quy mọi kỹ năng hoặc kết luận loss/optimizer cụ thể gây ra. Chỉmột seedtrain, checkpointselection cónoise; testđộc lập làđiểm quyết định.

Traincontrol72.409learner steps/200updates, candidate73.739/200; cùngepisodes/updates,không cùngtransition trajectories. Cảhai0self-blast trongtrain nhưngtestcódeaths, không diễn đạt thành“agent không còn tự nổ”. Trainactive wins74/200control và73/200candidate không thay thế held-outtest.

Candidate33.371queries/7.667accepted (22,98%),3.952selected-label agreements (51,55%accepted); meanpressuregain0,2365/accepted; mọiacceptedpath có10actions tronglượt này. Rejects:24.963next_bomb_masked,739no_pressure_increase,2escape_not_certified. Meanquery2,34ms, tổng78,09s trongtrain. Controlkhông chạycertifier, nênaccepted0control không phảicoverage thất bại.

All-MOVEagreement control56,32%/candidate55,82%; bomb-selected-instead-of-MOVE6,02%/6,27%; meantargetprobability0,5632/0,5569 vàentropy0,5626/0,6129. Những averages trêntrajectory khácnhau không chứng minh learning tốt/kém mộtcáchcausal; không cóaccepted-onlyprobabilitylog ởcontrol. BOMP thayMOVE làdiagnostic, cóthể cólýdo khác ngoài đặt sớm. Console `combat bombs x/y` códenominator gồmMOVEcombatlabels;không dùngtỷ lệđó như tỷ lệ chọnBOMB trênriêngBOMPtargets.

## Ngân sách và quyết định

Lượt đầyđủ3.502,54s≈58,38phút;control911,11s≈15,19phút,candidate1.147,82s≈19,13phút gồmvalidation/skills. Candidatechi phíarm cao hơnkhoảng26%; runtimephụ thuộctrajectory nênkhông gán toànbộchênhlệch chohelper. PeakCUDA≈88MB/arm. Đủhaiarm/testtrong2giờ; không còn bước chạy bắtbuộc cho pilot này.

1. GiữB và`local`mặc định; giữcandidatebest100 như artifact nghiêncứu,không phát hànhmodel thắng hơn.
2. Đóng pilot hiện tại vớiFAILgate.33000đãđọc;không dùnglại làmtestmới sau khituneCode/model.
3. Nhãn_MOVE cótínhiệu đáng nghiêncứu nhưngchưađượcgate xếpeligible. Nếutiếp tục, tạo protocol mới trướcchạy; không tự động kéo dài, nới ngưỡng hoặc chọncheckpoint qua33000.
4. Bước optimizer T4-A1(advantage normalization) vàruntime counter-bomb guard vẫn là các ablation riêng. Không ghépGAE/PPO/reward/curriculum vào kếtluận củapilot nhãn này. Cụthểphảiđóngbăng teacher cho cảhaiarm, cùngB/ngân sách vàtestmới, rồiđo lại; chưa triển khai/train bướcđó ở đây.

[Protocol/lệnh vàgiới hạn](/D:/fpt/subject/BomberMan/COMBAT_MOVE_GUIDE.md). Không cầnngười dùngchạy hoặcgửi thêmfile để hoàn tất request hiện tại. Rawdata đã cótrong workspace và [ZIP](/D:/fpt/subject/BomberMan/runs/combat-move-full-20261015-013315-send.zip); review/report thêm sauRawZIP.
