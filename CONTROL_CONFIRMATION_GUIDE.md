# Chấm riêng control best400 sau T3.2 dừng C1

**Bước offline sau confirmation đã xong15/10:** [T3_approach_decision.md](/D:/fpt/subject/BomberMan/runs/t3-approach-audit-20261015-final/T3_approach_decision.md). Đã audit8.224 frames và dựng96 delayed counter-bomb fixture checks;119 tests PASS. Giữ B và chưa sửa production mask/train mới. Không cần chạy lại confirmation hoặc audit để gửi thêm kết quả.

**Đã hoàn tất và review:** [confirmation_report.md](/D:/fpt/subject/BomberMan/runs/control-confirmation-20261014-201557/confirmation_report.md). B32,75%/control35% ActiveWR,delta+2,25pp/CI95%[-3,26;8],suicide0,5% cả hai; gate FAIL,giữ B. Không cần chạy lại bài này. Các lệnh bên dưới giữ để tái lập; test32000 đã đọc, không dùng để tune/chọn model khác rồi gọi là xác nhận mới.

T3.2 đã dừng theo gate ởep200, không cần train nối. Control chạy600ep và chọnbest400 trên validation21000 (ActiveWR42%, B29%); đó chưa phải test độc lập. Bước này **chỉ đánh giá B và control best đã cố định**, không train, không thay model hoặc mở test31000/wood_branch của curriculum.

Protocol đã đăng ký trước khi xem test: [control_confirmation_protocol.json](/D:/fpt/subject/BomberMan/runs/t3-curriculum-full-20261014-154703/control_confirmation_protocol.json).400 games/model,200 seed bàn32000–32199/cảA0-A1,9×9/2agents/max500,RuleBot,CPU1thread và mask hiện tại. Gate: ActiveWR hơn B≥5pp, bootstrap95% lower>0, suicide≤B. Một seed train vẫn chỉ là ứng viên cần lặp lại.

## 1. Chuẩn bị và kiểm tra hashes

Chạy PowerShell tại repo. Không sửa model/code/ngưỡng trong khi chấm.

```powershell
Set-Location 'D:\fpt\subject\BomberMan'
$py = 'D:\anaconda\envs\torch\python.exe'
$confirmProtocolPath = 'D:\fpt\subject\BomberMan\runs\t3-curriculum-full-20261014-154703\control_confirmation_protocol.json'
$confirmProtocol = Get-Content -LiteralPath $confirmProtocolPath -Raw | ConvertFrom-Json

function Assert-ControlConfirmationInputs {
    foreach ($entry in $confirmProtocol.source_sha256.PSObject.Properties) {
        $sourcePath = Join-Path 'D:\fpt\subject\BomberMan' $entry.Name
        if ((Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.Value) {
            throw "Source đã đổi: $($entry.Name). Dừng và gửi thông tin; chưa chấm test."
        }
    }
    foreach ($entry in $confirmProtocol.checkpoints.PSObject.Properties) {
        if ((Get-FileHash -LiteralPath $entry.Value.path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.Value.sha256) {
            throw "Checkpoint đã đổi: $($entry.Name). Dừng tại đây."
        }
    }
}
Assert-ControlConfirmationInputs

$confirmStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$confirmDir = Join-Path (Get-Location) "runs\control-confirmation-$confirmStamp"
if ((Test-Path -LiteralPath $confirmDir) -or (Test-Path -LiteralPath "$confirmDir-send.zip")) {
    throw 'Output đã tồn tại; chọn tên mới.'
}
New-Item -ItemType Directory -Path $confirmDir | Out-Null
Copy-Item -LiteralPath $confirmProtocolPath -Destination (Join-Path $confirmDir 'protocol.json')
Set-Content -LiteralPath 'runs\control-confirmation-last-path.txt' -Value $confirmDir
$confirmOriginal = Join-Path $confirmDir 'evaluation_original_32000.json'
$confirmControl = Join-Path $confirmDir 'evaluation_control_best_32000.json'
```

## 2. Chấm B rồi control best400

Hai lệnh dùng evaluator hiện tại; wrapper chỉ đặt Torch CPU1thread. Không có teacher chọn hành động.

```powershell
& $py -u -c "import runpy, torch; torch.set_num_threads(1); runpy.run_module('evaluate', run_name='__main__')" --mode seeded --board_size 9 --num_agents 2 --num_episodes 400 --max_steps 500 --seed 32000 --device cpu --ac_model ($confirmProtocol.checkpoints.original.path) --output $confirmOriginal 2>&1 |
    ForEach-Object { $_.ToString() } | Tee-Object -FilePath (Join-Path $confirmDir 'original-console.log')
if ($LASTEXITCODE -ne 0) { throw 'Chấm B lỗi; gửi console.log.' }

& $py -u -c "import runpy, torch; torch.set_num_threads(1); runpy.run_module('evaluate', run_name='__main__')" --mode seeded --board_size 9 --num_agents 2 --num_episodes 400 --max_steps 500 --seed 32000 --device cpu --ac_model ($confirmProtocol.checkpoints.control_best.path) --output $confirmControl 2>&1 |
    ForEach-Object { $_.ToString() } | Tee-Object -FilePath (Join-Path $confirmDir 'control-console.log')
if ($LASTEXITCODE -ne 0) { throw 'Chấm control lỗi; gửi console.log.' }

Assert-ControlConfirmationInputs
```

Mỗi model400 trận có thể mất nhiều phút; CLI này in tổng hợp khi hoàn tất, không có progress mỗi trận. Giữ máy hoạt động. Nếu mở PowerShell lại, đọc `runs\control-confirmation-last-path.txt` để lấy đúng thư mục cũ; nạp lại protocol/biến từ bước1 và giữ nguyên hai checkpoint/source khi hoàn tất phần chấm bị gián đoạn. Không chọn thêm checkpoint theo test32000 rồi gọi đó là cùng bài xác nhận.

## 3. Kiểm tra đủ800 trận, bootstrap và tạo ZIP

```powershell
@'
import hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
from pilot_two_bomb import paired_active_bootstrap

folder = Path(sys.argv[1])
p = json.loads((folder / 'protocol.json').read_text(encoding='utf-8-sig'))
for name, expected in p['source_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
for name, cp in p['checkpoints'].items():
    assert hashlib.sha256(Path(cp['path']).read_bytes()).hexdigest() == cp['sha256'], name
results = {}
for name in ('original', 'control_best'):
    r = json.loads((folder / f'evaluation_{name}_32000.json').read_text())
    assert (r['num_episodes'], r['seed'], r['board_size'], r['num_agents'], r['max_steps']) == (400, 32000, 9, 2, 500)
    assert len(r['games']) == 400
    assert {(g['seed'], g['aid']) for g in r['games']} == {(s,a) for s in range(32000,32200) for a in (0,1)}
    for c in r['by_corner'].values():
        assert c['num_episodes'] == 200
        c['passive_win_rate'] = c['win_rate'] - c['active_win_rate']
    r['passive_win_rate'] = r['win_rate'] - r['active_win_rate']
    results[name] = r
comparison = paired_active_bootstrap(results['control_best'], results['original'], samples=p['bootstrap']['samples'], seed=p['bootstrap']['seed'])
passed = (comparison['difference_pp'] >= p['gate']['minimum_activeWR_gain_pp'] and comparison['ci95_pp'][0] > 0
          and results['control_best']['suicide_rate'] <= results['original']['suicide_rate'])
summary = {'status': 'complete', 'confirmation_only': True, 'curriculum_protocol_completed': False,
    'hashes_unchanged': True, 'games_per_model': 400, 'comparison': comparison, 'gate_passed': bool(passed),
    'decision': 'candidate_for_replication' if passed else 'retain_original', 'automatic_model_replacement': False,
    'runtime': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__, 'evaluation_device': 'cpu', 'torch_threads': 1},
    'models': {n: {k:v for k,v in r.items() if k != 'games'} for n,r in results.items()}}
(folder / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
print(json.dumps(summary, indent=2))
'@ | & $py - $confirmDir 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath (Join-Path $confirmDir 'comparison-console.log')
if ($LASTEXITCODE -ne 0) { throw 'Chưa đủ dữ liệu/hashes/comparison; gửi các files hiện có.' }

if (Test-Path -LiteralPath "$confirmDir-send.zip") { throw 'ZIP đã có; đổi tên để giữ bản cũ.' }
Compress-Archive -LiteralPath $confirmDir -DestinationPath "$confirmDir-send.zip"
Get-Item -LiteralPath "$confirmDir-send.zip"
```

Gửi **`control-confirmation-...-send.zip`**. ZIP có protocol, hai JSON game-level, ba console logs và summary/CI/từng góc. Không cần train hoặc gửi lại ZIP T3.2 gốc. Nếu lỗi/gián đoạn, gửi logs/JSON hiện có; thiếu400/model thì không kết luận.

Qua gate mới có ứng viên để kiểm chứng/lặp train seed; vẫn giữ model gốc cho đến khi review. Không dùng kết quả này để tuyên bố curriculum C2/C3 hiệu quả.
