"""Build every figure of the paper from the logged runs in runs/.

Run from the repo root:  D:\\anaconda\\envs\\torch\\python.exe paper\\scripts\\make_figures.py [fig_name ...]
Outputs PDF + PNG figures to paper/figures/ and the numbers they plot to paper/figures/figure_data.json.
No new games are played except one short RuleBot-vs-RuleBot episode for the observation figure.
The game-board figure reuses the screenshots made by paper/scripts/capture_board.py.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RUNS = ROOT / 'runs'
OUT = ROOT / 'paper' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

from pilot_two_bomb import paired_active_bootstrap  # the repo's own paired bootstrap (seed 20261014)

INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
# One colour per agent across the whole paper: a hue per family, darker = later/stronger.
ENTITY = {
    'B': '#86b6ef', 'F': '#3987e5', 'G': '#184f95',                       # RL-trained CNNs (blue)
    'G+s1': '#f5ad8a', 'G+s2': '#eb6834', 'G+s3': '#a33d14',              # G + kill-search (orange)
    'S': '#b3a9f2', 'S2': '#7565e0', 'S3': '#3d2f99',                     # distilled CNNs (violet)
    'RuleBot': '#52514e', 'control': '#9a9993',
}
FAMILY = {'RL': 'G', 'Search': 'G+s2', 'Distilled': 'S2'}
# Outcome categories use hues that no agent uses.
OUTCOME = {'Active win': '#1baf7a', 'Passive/ring win': '#eda100', 'Draw': '#c9c8c2',
           'Self-kill': '#e34948', 'Killed/ring loss': '#e87ba4'}
SEQ = matplotlib.colors.LinearSegmentedColormap.from_list('seq', ['#fbfbfa', '#d9d8d3', '#8f8e88', '#3b3a36', '#0b0b0b'])

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
    'font.size': 8.5, 'axes.titlesize': 8.5, 'axes.labelsize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 7.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
    'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True, 'pdf.fonttype': 42,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02, 'lines.linewidth': 1.5})
SMALL = 7.5    # smallest annotation size used anywhere
COL1, COL2 = 3.45, 7.16   # IEEE column / page width in inches
DATA = {}


def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf')
    fig.savefig(OUT / f'{name}.png', dpi=200)
    plt.close(fig)
    print('wrote', name)


def games(path):
    return json.load(open(RUNS / path, encoding='utf-8'))


def rate(result, key):
    return 100.0 * np.mean([g[key] for g in result['games']])


def board_ci(result, key='active_win', samples=10000, seed=20261014):
    """Percentile CI of a single model's rate, resampling boards (both corners kept together)."""
    by_board = {}
    for g in result['games']:
        by_board.setdefault(g['seed'], []).append(g[key])
    per = np.array([np.mean(v) for _, v in sorted(by_board.items())])
    rng = np.random.default_rng(seed)
    draws = per[rng.integers(0, len(per), size=(samples, len(per)))].mean(1) * 100
    return float(per.mean() * 100), [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def outcome_shares(result):
    g = result['games']
    n = len(g)
    win = sum(x['win'] for x in g)
    active = sum(x['active_win'] for x in g)
    draw = sum(x['draw'] for x in g)
    suicide = sum(1 for x in g if not x['win'] and not x['draw'] and x['suicide'] > 0)
    loss = n - win - draw
    return {k: 100.0 * v / n for k, v in {
        'Active win': active, 'Passive/ring win': win - active, 'Draw': draw,
        'Self-kill': suicide, 'Killed/ring loss': loss - suicide}.items()}


def fmt(v):
    return f'{v:.2f}'.rstrip('0').rstrip('.')


def family_legend(ax, **kw):
    for fam, ent in FAMILY.items():
        ax.bar(0, 0, color=ENTITY[ent], label={'RL': 'CNN, RL-trained', 'Search': 'G + kill-search',
                                               'Distilled': 'CNN, distilled'}[fam])
    ax.legend(frameon=False, handlelength=1.0, columnspacing=0.8, **kw)


# ----------------------------------------------------------------------------------------------
MILESTONES = [  # label, family, file, test block; every row is schedule E (B on the cross-check block 60000)
    ('B', 'RL', 'crosscheck-friend-a6650a0/evaluation_E3_B_E_60000.json', 60000),
    ('F', 'RL', 'sd53-20261018/evaluation_F_E_43000.json', 43000),
    ('G', 'RL', 'sd78-20261019/evaluation_G_E_46000.json', 46000),
    ('S', 'Distilled', 'sd53-20261018/evaluation_sd53_best_E_43000.json', 43000),
    ('S2', 'Distilled', 'sd8a-20261020/evaluation_S2_E_48000.json', 48000),
    ('S3', 'Distilled', 'sd9-20261023/evaluation_S3_E_57000.json', 57000),
    ('G+s1', 'Search', 'sd78-20261019/evaluation_search_E_46000.json', 46000),
    ('G+s2', 'Search', 'sd79-20261020/evaluation_G_search2_E_47000.json', 47000),
    ('G+s3', 'Search', 'sd8b-20261020/evaluation_G_search3_E_49000.json', 49000),
]


def fig_milestones():
    rows = []
    for label, fam, path, seed in MILESTONES:
        r = games(path)
        assert len(r['games']) == 400, (label, len(r['games']))
        m, ci = board_ci(r)
        rows.append({'label': label, 'family': fam, 'seed': seed, 'active': m, 'ci': ci,
                     'wr': rate(r, 'win'), 'draw': rate(r, 'draw'),
                     'suicide': 100.0 * np.mean([g['suicide'] > 0 for g in r['games']]),
                     'outcomes': outcome_shares(r), 'steps': float(np.mean([g['steps'] for g in r['games']]))})
    DATA['milestones'] = rows
    fig, ax = plt.subplots(figsize=(COL1, 2.35))
    x = np.arange(len(rows))
    for i, r in enumerate(rows):
        ax.bar(i, r['active'], width=0.66, color=ENTITY[r['label']], zorder=2)
        ax.errorbar(i, r['active'], yerr=[[r['active'] - r['ci'][0]], [r['ci'][1] - r['active']]],
                    color=INK, lw=0.8, capsize=2, zorder=3)
        ax.text(i, r['ci'][1] + 1.5, fmt(r['active']), ha='center', va='bottom', fontsize=SMALL, color=INK)
    ax.set_xticks(x, [r['label'] for r in rows])
    ax.set_ylabel('ActiveWR vs RuleBot (%)')
    ax.set_ylim(0, 106)
    ax.grid(axis='x', visible=False)
    family_legend(ax, loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3)
    save(fig, 'fig_milestones')

    fig, ax = plt.subplots(figsize=(COL1, 2.35))
    left = np.zeros(len(rows))
    for cat, col in OUTCOME.items():
        vals = np.array([r['outcomes'][cat] for r in rows])
        ax.barh(x, vals, left=left, color=col, edgecolor='white', linewidth=0.8, height=0.72, label=cat)
        left += vals
    ax.set_yticks(x, [r['label'] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel('Share of 400 test games (%)')
    ax.grid(axis='y', visible=False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.42, 1.27), ncol=3, frameon=False, handlelength=1.0,
              columnspacing=0.8)
    save(fig, 'fig_outcomes')


# ----------------------------------------------------------------------------------------------
PAIRS = [  # group, label, candidate, reference, expected logged delta (for a self-check)
    ('Search as RL loss', 'RL + sparse search labels − F', 'sd5-20261017/evaluation_search_E_42000.json', 'sd5-20261017/evaluation_F_E_42000.json', -2.75),
    ('Search as RL loss', 'plain RL control − F', 'sd5-20261017/evaluation_control_E_42000.json', 'sd5-20261017/evaluation_F_E_42000.json', -6.5),
    ('RL refinements', 'group RL − control', 'sd75-20261018/evaluation_group_E_45000.json', 'sd75-20261018/evaluation_control_E_45000.json', None),
    ('RL refinements', 'group RL − F', 'sd75-20261018/evaluation_group_E_45000.json', 'sd75-20261018/evaluation_F_E_45000.json', 3.75),
    ('RL refinements', 'cut-off rule − F (val.)', 'sd6-cutoff-probe-21000-b/evaluation_cutoff_21000.json', 'sd6-cutoff-probe-21000-b/evaluation_f_21000.json', -1.75),
    ('RL refinements', 'mask margin k=15 − k=0', 'sd4-margin-test-41000/evaluation_k15_41000.json', 'sd4-margin-test-41000/evaluation_k0_41000.json', None),
    ('Search at play time', 'G+s1 − G', 'sd78-20261019/evaluation_search_E_46000.json', 'sd78-20261019/evaluation_G_E_46000.json', 16.5),
    ('Search at play time', 'G+s2 − G+s1', 'sd79-20261020/evaluation_G_search2_E_47000.json', 'sd79-20261020/evaluation_G_search_E_47000.json', 21.0),
    ('Search at play time', 'G+s3 − G+s2', 'sd8b-20261020/evaluation_G_search3_E_49000.json', 'sd8b-20261020/evaluation_G_search2_E_49000.json', 8.25),
    ('Dense distillation', 'round 1: S − F', 'sd53-20261018/evaluation_sd53_best_E_43000.json', 'sd53-20261018/evaluation_F_E_43000.json', 16.25),
    ('Dense distillation', 'round 2: S2 − G', 'sd8a-20261020/evaluation_S2_E_48000.json', 'sd8a-20261020/evaluation_G_E_48000.json', 35.0),
    ('Dense distillation', 'round 3: S3 − S2', 'sd9-20261023/evaluation_S3_E_57000.json', 'sd9-20261023/evaluation_S2_E_57000.json', 10.5),
    ('Student vs teacher', 'round 1: S − (F+s1)', 'sd53-20261018/evaluation_sd53_best_E_43000.json', 'sd53-20261018/evaluation_F_search_E_43000.json', None),
    ('Student vs teacher', 'round 2: S2 − (G+s2)', 'sd8a-20261020/evaluation_S2_E_48000.json', 'sd8a-20261020/evaluation_G_search2_E_48000.json', -3.5),
    ('Student vs teacher', 'round 3: S3 − (S2+s3)', 'sd9-20261023/evaluation_S3_E_57000.json', 'sd9-20261023/evaluation_S2_search3_E_57000.json', -0.75),
]
# Each comparison group borrows the colour of the agent family it is about.
GROUP_COLOR = {'Search as RL loss': ENTITY['control'], 'RL refinements': ENTITY['F'],
               'Search at play time': ENTITY['G+s2'], 'Dense distillation': ENTITY['S2'],
               'Student vs teacher': ENTITY['S3']}
GROUP_MARK = {'Search as RL loss': 's', 'RL refinements': 'o', 'Search at play time': 'D',
              'Dense distillation': 'o', 'Student vs teacher': '^'}


def fig_forest():
    rows = []
    for group, label, cand, ref, expected in PAIRS:
        a, b = games(cand), games(ref)
        res = paired_active_bootstrap(a, b)
        wr = paired_active_bootstrap({'games': [{**g, 'active_win': g['win']} for g in a['games']]},
                                     {'games': [{**g, 'active_win': g['win']} for g in b['games']]})
        if expected is not None:
            assert abs(res['difference_pp'] - expected) < 1e-6, (label, res['difference_pp'], expected)
        rows.append({'group': group, 'label': label, 'seed': a['games'][0]['seed'], 'active': res, 'win': wr})
    DATA['paired'] = rows
    fig, ax = plt.subplots(figsize=(COL2, 3.3))
    y, yt, yl = 0, [], []
    for g in GROUP_COLOR:
        ax.text(-14.6, y + 0.8, g, fontsize=8, color=INK, fontweight='bold', va='center')
        for r in [r for r in rows if r['group'] == g]:
            d, (lo, hi) = r['active']['difference_pp'], r['active']['ci95_pp']
            ax.plot([lo, hi], [y, y], color=GROUP_COLOR[g], lw=2.0, solid_capstyle='round')
            ax.plot(d, y, GROUP_MARK[g], ms=5.5, color=GROUP_COLOR[g], mec='white', mew=0.8)
            ax.text(43.5, y, f'{d:+.2f}  [{lo:+.2f}, {hi:+.2f}]', va='center', ha='left', fontsize=SMALL, color=INK)
            yt.append(y); yl.append(r['label'])
            y -= 1
        y -= 1.3
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.axvline(5, color=MUTED, lw=0.8, ls=(0, (2, 2)))
    ax.text(5.5, 0.55, 'gate +5', fontsize=SMALL, color=MUTED)
    ax.set_yticks(yt, yl)
    ax.set_xlim(-15, 42)
    ax.set_xlabel('Paired ΔActiveWR (pp) with 95% bootstrap CI over boards')
    ax.grid(axis='y', visible=False)
    save(fig, 'fig_forest')


def fig_wr_vs_active():
    cases = [
        ('Ring mask margin k=15\n(B, test 41k)', 'sd4-margin-test-41000/evaluation_k0_41000.json', 'sd4-margin-test-41000/evaluation_k15_41000.json'),
        ('Cut-off rule\n(F, val. 21k)', 'sd6-cutoff-probe-21000-b/evaluation_f_21000.json', 'sd6-cutoff-probe-21000-b/evaluation_cutoff_21000.json'),
        ('Sudden death on\n(G, test 46k)', 'sd78-20261019/evaluation_G_off_46000.json', 'sd78-20261019/evaluation_G_E_46000.json'),
        ('Sudden death on\n(S3, test 57k)', 'sd9-20261023/evaluation_S3_off_57000.json', 'sd9-20261023/evaluation_S3_E_57000.json'),
    ]
    rows = []
    for label, before, after in cases:
        b, a = games(before), games(after)
        rows.append({'label': label, 'wr': [rate(b, 'win'), rate(a, 'win')],
                     'active': [rate(b, 'active_win'), rate(a, 'active_win')]})
    DATA['wr_vs_active'] = rows
    fig, axes = plt.subplots(1, 2, figsize=(COL1, 2.35), sharey=True)
    for ax, key, title in zip(axes, ['wr', 'active'], ['Win rate', 'ActiveWR']):
        for i, r in enumerate(rows):
            v0, v1 = r[key]
            ax.plot([v0, v1], [i, i], color='#b9b8b2', lw=2.4, zorder=1)
            ax.plot(v0, i, 'o', mfc='white', mec=INK, mew=1.1, ms=5.5, zorder=2)
            ax.plot(v1, i, 'o', color=INK, ms=4.0, zorder=3)
            ax.text(max(v0, v1) + 5, i, f'{v1 - v0:+.2f}', va='center', fontsize=SMALL, color=INK)
        ax.set_title(title)
        ax.set_xlim(20, 128)
        ax.set_xticks([25, 50, 75, 100])
        ax.set_xlabel('% of 400 games')
        ax.grid(axis='y', visible=False)
    axes[0].set_yticks(range(len(rows)), [r['label'] for r in rows], fontsize=SMALL)
    axes[0].invert_yaxis()
    axes[0].plot([], [], 'o', mfc='white', mec=INK, label='before')
    axes[0].plot([], [], 'o', color=INK, label='after')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(0.6, -0.02), ncol=2, frameon=False)
    save(fig, 'fig_wr_vs_active')


# ----------------------------------------------------------------------------------------------
def _jsonl(path):
    return [json.loads(line) for line in open(RUNS / path, encoding='utf-8') if line.strip()]


def fig_training_curves():
    # The sparse-label arm was trained and validated on a second machine (torch 2.14.1); its validation
    # numbers are not comparable with the other two (same F, same seed: 43.75 vs 47.0 at episode 0).
    arms = [('Plain RL, control (torch 2.5.1)', 'sd5-20261017/control', ENTITY['control'], '-'),
            ('RL + sparse depth-1 search labels (torch 2.14.1, other machine)', 'sd5-20261017/search', ENTITY['G+s1'], '--'),
            ('RL + group advantage = G (torch 2.5.1)', 'sd75-20261018/full', ENTITY['G'], '-')]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.5))
    window = 200
    out = {}
    for label, path, col, ls in arms:
        m = _jsonl(f'{path}/metrics.jsonl')
        act = np.array([r['active_win'] for r in m], dtype=float) * 100
        roll = np.convolve(act, np.ones(window) / window, mode='valid')
        axes[0].plot(np.arange(window, len(act) + 1), roll, color=col, ls=ls, lw=1.3, label=label)
        v = _jsonl(f'{path}/validation_metrics.jsonl')
        ep = [r['episode'] for r in v]
        va = [r['active_win_rate'] for r in v]
        axes[1].plot(ep, va, marker='o', ls=ls, color=col, ms=4, lw=1.3, label=label)
        out[label] = {'train_episodes': len(act), 'train_active_first200': float(act[:200].mean()),
                      'train_active_last200': float(act[-200:].mean()), 'validation': list(zip(ep, va))}
    axes[0].set_title(f'Training games vs RuleBot (rolling mean of {window})')
    axes[1].set_title('Validation (seed 21000, 400 games)')
    for ax in axes:
        ax.set_xlabel('Episode')
        ax.set_ylabel('ActiveWR (%)')
    axes[1].annotate('G selected (ep. 1000)', xy=(1000, out[arms[2][0]]['validation'][2][1]), xytext=(1250, 51.3),
                     fontsize=SMALL, arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.7))
    fig.tight_layout()
    fig.legend(*axes[1].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=2,
               frameon=False)
    DATA['training_curves'] = out
    save(fig, 'fig_training_curves')


def fig_distill_curves():
    runs = [('Round 1: init F, teacher F+s1 → S', 'sd53-20261018', 'S', 'F'),
            ('Round 2: init G, teacher G+s2 → S2', 'sd8a-20261020', 'S2', 'G'),
            ('Round 3: init S2, teacher S2+s3 → S3', 'sd9-20261023', 'S3', 'S2')]
    selected = {'S': 8, 'S2': 16, 'S3': 12}
    fig, axes = plt.subplots(2, 2, figsize=(COL2, 3.9))
    (a_loss, a_acc), (a_kl, a_val) = axes
    out = {}
    for label, run, student, start in runs:
        col = ENTITY[student]
        m = json.load(open(RUNS / run / 'train' / 'metrics.json', encoding='utf-8'))['metrics']
        ep = [r['epoch'] for r in m]
        loss = [(r['epoch'], r['train_loss']) for r in m if 'train_loss' in r]
        acc = [r['decision_argmax_best_rate'] for r in m]
        kl = [r['anchor_kl'] for r in m]
        a_loss.plot(*zip(*loss), marker='o', ms=3, color=col, label=label)
        a_acc.plot(ep, acc, marker='o', ms=3, color=col, label=label)
        a_kl.plot(ep, kl, marker='o', ms=3, color=col, label=label)
        val = json.load(open(RUNS / run / 'validation.json', encoding='utf-8'))
        res = val.get('results', val)
        pts = [(0, res[start]['active_win_rate'])] + sorted(
            (int(k.split('_')[1]), v['active_win_rate']) for k, v in res.items() if k.startswith('epoch_'))
        a_val.plot(*zip(*pts), marker='o', ms=4, color=col, label=label)
        sel = dict(pts)[selected[student]]
        a_val.plot(selected[student], sel, marker='*', ms=11, color=col, mec='white', mew=0.6, zorder=5)
        out[label] = {'train_loss': loss, 'heldout_acc': list(zip(ep, acc)), 'anchor_kl': list(zip(ep, kl)),
                      'val_active': pts, 'selected_epoch': selected[student]}
    a_loss.set_title('Training loss (soft CE + KL anchor)')
    a_acc.set_title('Held-out decision states: argmax in teacher best set (%)')
    a_kl.set_title('Held-out anchor states: KL(reference ‖ student)')
    a_val.set_title('Validation ActiveWR (%); $\\star$ = selected epoch')
    for ax in axes.flat:
        ax.set_xlabel('Epoch')
    a_val.text(0.4, 47, 'epoch 0 = starting model', fontsize=SMALL, color=MUTED)
    fig.legend(*a_val.get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False)
    fig.tight_layout(h_pad=1.2)
    DATA['distill_curves'] = out
    save(fig, 'fig_distill_curves')


# ----------------------------------------------------------------------------------------------
def _confusion(run, models):
    import torch
    import run_sd53_pilot as sd53
    from train_actor_critic import ActorCriticNet, smart_load_weights
    chunks = {k: [] for k in ('decision_obs', 'decision_mask', 'decision_scores', 'decision_seed', 'anchor_seed')}
    for path in sorted((RUNS / run / 'collection').glob('shard-*.npz')):
        with np.load(path, allow_pickle=False) as data:
            for k in chunks:
                chunks[k].append(data[k])
    rec = {k: np.concatenate(v) for k, v in chunks.items()}
    held = sd53._heldout_board_ids(np.concatenate((rec['decision_seed'], rec['anchor_seed'])), 0.10, 530)
    idx = np.array([i for i, s in enumerate(rec['decision_seed']) if int(s) in held])
    target = np.stack([sd53.masked_soft_target(rec['decision_scores'][i], rec['decision_mask'][i]) for i in idx])
    useful = target.sum(1) > 0
    unique = (target > 0).sum(1) == 1
    result = {'heldout_decisions': int(len(idx)), 'useful': int(useful.sum()), 'unique_best': int((useful & unique).sum())}
    mats = {}
    for name, ckpt in models:
        net = ActorCriticNet(board_size=9)
        smart_load_weights(net, str(RUNS / ckpt), 'cpu')
        net.eval()
        with torch.no_grad():
            logits, _ = net(torch.from_numpy(rec['decision_obs'][idx]).float())
        logits = logits.masked_fill(~torch.from_numpy(rec['decision_mask'][idx]), -1e9)
        choice = logits.argmax(1).numpy()
        correct = target[np.arange(len(idx)), choice] > 0
        sel = useful & unique
        mat = np.zeros((6, 6))
        for t, p in zip(target[sel].argmax(1), choice[sel]):
            mat[t, p] += 1
        mats[name] = mat
        result[name] = {'set_accuracy': float(correct[useful].mean() * 100),
                        'unique_accuracy': float((np.diag(mat).sum() / mat.sum()) * 100),
                        'row_counts': mat.sum(1).astype(int).tolist()}
    return mats, result


def fig_confusion():
    short = ['ST', 'UP', 'DN', 'LT', 'RT', 'BO']
    panels = [('sd8a-20261020', [('G', 'G (before round 2)', 'sd75-20261018/full/best_phaseC.pt'),
                                  ('S2', 'S2 (after round 2)', 'sd8a-20261020/train/epoch_16.pt')]),
              ('sd9-20261023', [('S2', 'S2 (before round 3)', 'sd8a-20261020/train/epoch_16.pt'),
                                 ('S3', 'S3 (after round 3)', 'sd9-20261023/train/epoch_12.pt')])]
    fig, axes = plt.subplots(1, 4, figsize=(COL2, 2.55))
    out = {}
    k = 0
    for run, models in panels:
        mats, res = _confusion(run, [(name, ckpt) for _, name, ckpt in models])
        out[run] = res
        for ent, name, _ in models:
            mat = mats[name]
            norm = mat / np.maximum(mat.sum(1, keepdims=True), 1)
            ax = axes[k]
            ax.imshow(norm * 100, cmap=SEQ, vmin=0, vmax=100)
            for i in range(6):
                for j in range(6):
                    if mat[i].sum() and norm[i, j] >= 0.005:
                        ax.text(j, i, f'{norm[i, j] * 100:.0f}', ha='center', va='center', fontsize=7,
                                color='white' if norm[i, j] > 0.5 else INK)
            ax.set_xticks(range(6), short, fontsize=7.5)
            rows_n = [f'{s} ({int(n)})' for s, n in zip(short, mat.sum(1))]
            ax.set_yticks(range(6), rows_n, fontsize=7)
            ax.grid(False)
            ax.set_title(f"{name}\nacc. {res[name]['unique_accuracy']:.1f}% (n={int(mat.sum())})",
                         fontsize=8, color=INK)
            ax.set_xlabel('student argmax', fontsize=7.5)
            if k == 0:
                ax.set_ylabel('teacher best action (row n)', fontsize=7.5)
            for s in ax.spines.values():
                s.set_visible(False)
            k += 1
    fig.tight_layout(w_pad=0.6)
    DATA['confusion'] = out
    save(fig, 'fig_confusion')


# ----------------------------------------------------------------------------------------------
def _shard_games(run, prefix):
    out = []
    for path in sorted(glob.glob(str(RUNS / run / 'shards' / f'{prefix}_*.json'))):
        if not Path(path).stem[len(prefix) + 1:].isdigit():   # S3_vs_G_* must not match S3_vs_G_search3_*
            continue
        d = json.load(open(path, encoding='utf-8'))
        out.extend(d[2] if isinstance(d, list) else d['games'])
    return out


def fig_episode_length():
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.3), sharey=True)
    out = {}
    for ent, path in [('G', 'sd9-20261023/evaluation_G_E_57000.json'), ('S2', 'sd9-20261023/evaluation_S2_E_57000.json'),
                      ('S3', 'sd9-20261023/evaluation_S3_E_57000.json')]:
        steps = np.sort([g['steps'] for g in games(path)['games']])
        axes[0].step(steps, np.arange(1, len(steps) + 1) / len(steps) * 100, where='post', color=ENTITY[ent], label=ent)
        out[f'{ent}_vs_RuleBot'] = {'median': float(np.median(steps)), 'at_500': float(np.mean(steps >= 500) * 100)}
    for opp, ls in [('S2', '--'), ('G', ':'), ('S3', '-')]:
        g = _shard_games('sd91-21000', f'S3_vs_{opp}')
        steps = np.sort([x['steps'] for x in g])
        axes[1].step(steps, np.arange(1, len(steps) + 1) / len(steps) * 100, where='post', color=ENTITY[opp], ls=ls,
                     label=f'S3 vs {opp}')
        out[f'S3 vs {opp}'] = {'games': len(g), 'median': float(np.median(steps)), 'at_500': float(np.mean(steps >= 500) * 100)}
    axes[0].set_title('Learner vs RuleBot (test 57000)')
    axes[1].set_title('S3 vs frozen CNNs (val. 21000), colour = opponent')
    for ax in axes:
        ax.set_xlabel('Game length (steps)')
        ax.axvline(475, color=MUTED, lw=0.8, ls=(0, (2, 2)))
        ax.legend(loc='upper left', frameon=False)
        ax.set_xlim(0, 505)
    axes[0].text(468, 10, 'ring starts', rotation=90, ha='right', fontsize=SMALL, color=MUTED)
    axes[0].set_ylabel('Cumulative % of games')
    DATA['episode_length'] = out
    save(fig, 'fig_episode_length')


def fig_cnn_matchups():
    s = json.load(open(RUNS / 'sd91-21000' / 'summary.json', encoding='utf-8'))['results']
    keys = [k for k in ['S2_vs_S2', 'S2_vs_G', 'S2_vs_S3', 'S3_vs_S2', 'S3_vs_G', 'S3_vs_S3',
                        'S2_vs_G_search3', 'S3_vs_G_search3'] if k in s]
    cats = [('Active win', lambda r: r['active_win_rate']),
            ('Passive/ring win', lambda r: r['win_rate'] - r['active_win_rate']),
            ('Draw', lambda r: r['draw_rate']),
            ('Killed/ring loss', lambda r: 100 - r['win_rate'] - r['draw_rate'])]
    fig, ax = plt.subplots(figsize=(COL1, 2.5))
    left = np.zeros(len(keys))
    for name, fn in cats:
        vals = np.array([fn(s[k]) for k in keys])
        ax.barh(range(len(keys)), vals, left=left, color=OUTCOME[name], edgecolor='white', linewidth=0.8,
                height=0.72, label=name.replace('Killed/ring loss', 'Loss'))
        left += vals
    labels = [k.replace('_vs_', ' vs ').replace('G_search3', 'G+s3') + f" ({s[k]['num_episodes']})" for k in keys]
    ax.set_yticks(range(len(keys)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel('Share of games (%); (n) = games per pair')
    ax.grid(axis='y', visible=False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.35, 1.22), ncol=2, frameon=False, handlelength=1.0, columnspacing=0.8)
    DATA['cnn_matchups'] = {k: s[k] for k in keys}
    save(fig, 'fig_cnn_matchups')


def fig_draw_probe():
    d = json.load(open(RUNS / 'sd92-21000' / 'summary.json', encoding='utf-8'))
    opps = list(d['per_opponent'])
    fig, ax = plt.subplots(figsize=(COL1, 2.3))
    w = 0.36
    out = {}
    styles = [('draws_with_kill_rulebot', 'search assumes RuleBot (wrong model)', dict(color='#c9c8c2', hatch='')),
              ('draws_with_kill_oracle', "search uses the opponent's own CNN (correct)", dict(color=ENTITY['S3']))]
    for j, (key, label, st) in enumerate(styles):
        for i, o in enumerate(opps):
            r = d['per_opponent'][o][key]
            x = i + (j - 0.5) * w
            ax.bar(x, r['percent'], width=w * 0.92, label=label if i == 0 else None, zorder=2,
                   edgecolor=INK, linewidth=0.5, **st)
            ax.errorbar(x, r['percent'], yerr=[[r['percent'] - r['ci95'][0]], [r['ci95'][1] - r['percent']]],
                        color=INK, lw=0.8, capsize=2, zorder=3)
            ax.text(x, r['ci95'][1] + 2, f"{r['percent']:.0f}", ha='center', fontsize=SMALL)
            out.setdefault(o, {})[key] = r
    out['pooled'] = d.get('pooled', {})
    ax.set_xticks(range(len(opps)), [f"vs {o}\n({d['per_opponent'][o]['draws']} draws)" for o in opps])
    ax.set_ylabel("S3's draws with a kill plan (%)")
    ax.set_ylim(0, 118)
    ax.grid(axis='x', visible=False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=1, frameon=False)
    DATA['draw_probe'] = out
    save(fig, 'fig_draw_probe')


def fig_search_depth():
    rows = [('G', 'sd78-20261019/evaluation_G_E_46000.json', 46000),
            ('G+s1', 'sd78-20261019/evaluation_search_E_46000.json', 46000),
            ('G+s1', 'sd79-20261020/evaluation_G_search_E_47000.json', 47000),
            ('G+s2', 'sd79-20261020/evaluation_G_search2_E_47000.json', 47000),
            ('G+s2', 'sd8b-20261020/evaluation_G_search2_E_49000.json', 49000),
            ('G+s3', 'sd8b-20261020/evaluation_G_search3_E_49000.json', 49000)]
    pts = []
    for label, path, seed in rows:
        m, ci = board_ci(games(path))
        pts.append({'label': label, 'seed': seed, 'active': m, 'ci': ci})
    fig, ax = plt.subplots(figsize=(COL1, 2.3))
    depth = {'G': 0, 'G+s1': 1, 'G+s2': 2, 'G+s3': 3}
    for seed, ls in [(46000, '-'), (47000, '--'), (49000, ':')]:
        ps = [p for p in pts if p['seed'] == seed]
        ax.plot([depth[p['label']] for p in ps], [p['active'] for p in ps], color=MUTED, ls=ls, lw=1.1, zorder=1,
                label=f'paired test block {seed}')
    for p in pts:
        x = depth[p['label']] + {46000: -0.06, 47000: 0.0, 49000: 0.06}[p['seed']]
        ax.errorbar(x, p['active'], yerr=[[p['active'] - p['ci'][0]], [p['ci'][1] - p['active']]], fmt='o',
                    color=ENTITY[p['label']], ms=5.5, capsize=2, lw=1.1, zorder=3)
        side = 1 if (p['label'], p['seed']) not in {('G+s2', 49000), ('G+s1', 46000)} else -1
        ax.text(x + 0.1 * side, p['active'] + (-1.5 if side > 0 else 1.5), f"{p['active']:.2f}", fontsize=SMALL,
                ha='left' if side > 0 else 'right', va='top' if side > 0 else 'bottom')
    ax.set_xticks(range(4), ['0 (G alone)', '1', '2', '3'])
    ax.set_xlabel('Kill-search depth (learner moves)')
    ax.set_ylabel('ActiveWR vs RuleBot (%)')
    ax.set_ylim(38, 102)
    ax.set_xlim(-0.4, 3.5)
    ax.legend(loc='lower right', frameon=False)
    DATA['search_depth'] = {'points': pts}
    save(fig, 'fig_search_depth')


# ----------------------------------------------------------------------------------------------
def fig_observation():
    from bomberman_env import BombermanFFAEnv
    from rule_based_bot import RuleBasedAgent
    env = BombermanFFAEnv(board_size=9, num_agents=2, max_steps=500, seed=21007, sudden_death=True,
                          sudden_death_start=475, sudden_death_interval=25, sudden_death_warning_steps=40)
    obs = env.reset()
    np.random.seed(21007)  # RuleBot breaks ties with np.random; fixes the snapshot
    bots = [RuleBasedAgent(0), RuleBasedAgent(1)]
    snap = None
    for step in range(1, 500):
        obs, _, _, info = env.step([b.select_action(env) for b in bots])
        o = obs[0]
        if snap is None and o[2].any() and o[3].any() and o[11].any():
            snap = (step, o.copy())
        if info.get('episode_done'):
            break
    s_step, o = snap
    names = ['0 rigid walls', '1 wooden walls', '2 bomb timer', '3 flames', '4 ammo item', '5 range item',
             '6 self', '7 opponent', '8 free bomb slots', '9 bomb capacity', '10 blast range',
             '11 1/(time to\ndamage)', '12 ring warning']
    fig, axes = plt.subplots(2, 7, figsize=(COL2, 2.6))
    for i, ax in enumerate(axes.flat):
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color('#b9b8b2')
        if i < 13:
            ax.imshow(o[i], cmap=SEQ, vmin=0, vmax=1)
            ax.set_title(names[i], fontsize=SMALL)
            if not o[i].any():
                ax.text(4, 4, 'all zero\nat this step', ha='center', va='center', fontsize=6.5, color=MUTED)
        else:
            ax.axis('off')
    DATA['observation'] = {'board_seed': 21007, 'step': s_step, 'agent': 0}
    save(fig, 'fig_observation')


def fig_gameplay():
    import matplotlib.image as mpimg
    meta = json.load(open(OUT / 'board' / 'frames.json', encoding='utf-8'))
    text = {'a_place': 'S3 (A0) places a bomb', 'b_mid': 'RuleBot (A1) boxed in\nbetween two bombs',
            'c_before': 'forecast blast covers\nevery escape (shaded)', 'd_kill': 'blast: A1 eliminated,\ncredited to S3'}
    fig, axes = plt.subplots(1, 4, figsize=(COL2, 2.25))
    for ax, f in zip(axes, meta['frames']):
        img = mpimg.imread(ROOT / f['png'])[..., :3]
        bg = np.array([0x10, 0x18, 0x27]) / 255          # visualize.BG canvas colour
        keep = np.abs(img - bg).sum(-1) > 0.08
        rows, cols = np.where(keep.any(1))[0], np.where(keep.any(0))[0]
        ax.imshow(img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1])
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"step {f['step']}", fontsize=8.5)
        ax.set_xlabel(text[f['name']], fontsize=SMALL)
    fig.tight_layout(w_pad=0.4)
    DATA['gameplay'] = meta
    save(fig, 'fig_gameplay')


# ----------------------------------------------------------------------------------------------
# Block diagrams are drawn in points (1 axis unit = 1 pt) so every box can be checked against the
# measured size of its text; _check_fit fails the build if any label leaves its box.
def _tint(hex_color, amount=0.78):
    rgb = np.array(matplotlib.colors.to_rgb(hex_color))
    return tuple(rgb + (1 - rgb) * amount)


def _on(hex_color):
    r, g, b = matplotlib.colors.to_rgb(hex_color)
    return 'white' if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.55 else INK


class _Canvas:
    def __init__(self, w, h):
        self.fig = plt.figure(figsize=(w / 72, h / 72))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, w); self.ax.set_ylim(0, h); self.ax.axis('off')
        self.fits = []          # (text artist, (x0, y0, x1, y1) it must stay inside)

    def text(self, x, y, s, box, **kw):
        kw = {'fontsize': SMALL, 'color': INK, 'va': 'center', **kw}
        t = self.ax.text(x, y, s, **kw)
        self.fits.append((t, box))
        return t

    def panel(self, x, y, w, h, title, col):
        """Dashed group border with a solid header tab, as in the reference figure."""
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=4',
                                         fc=_tint(col, 0.94), ec=col, lw=0.9, ls=(0, (4, 2.5))))
        t = self.ax.text(x + 7, y + h - 7.5, title, fontsize=SMALL, color='white', weight='bold', va='center')
        self.fig.canvas.draw()
        tw = self._width(t)
        self.ax.add_patch(FancyBboxPatch((x, y + h - 15), tw + 14, 15, boxstyle='round,pad=0,rounding_size=3',
                                         fc=col, ec=col, lw=0.9, zorder=1.5))
        t.set_zorder(3)
        self.fits.append((t, (x, y, x + w, y + h)))

    def card(self, x, y, w, h, head, lines, col, bold_last=False):
        """White node with a coloured header strip naming it."""
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=3',
                                         fc='white', ec=col, lw=1.0, zorder=2))
        self.ax.add_patch(FancyBboxPatch((x, y + h - 13), w, 13, boxstyle='round,pad=0,rounding_size=3',
                                         fc=col, ec=col, lw=1.0, zorder=2.1))
        box = (x, y, x + w, y + h)
        self.text(x + w / 2, y + h - 6.5, head, box, ha='center', color=_on(col), weight='bold', zorder=3)
        step = (h - 13) / (len(lines) + 1)
        for i, line in enumerate(lines):
            txt, c = line if isinstance(line, tuple) else (line, INK)
            weight = 'bold' if bold_last and i == len(lines) - 1 else 'normal'
            self.text(x + w / 2, y + h - 13 - step * (i + 1), txt, box, ha='center', color=c, weight=weight, zorder=3)
        return box

    def arrow(self, pts, col=INK, ls='-', lw=0.9):
        """Orthogonal polyline; the arrow head sits on the last segment."""
        xs, ys = zip(*pts)
        if len(pts) > 2:
            self.ax.plot(xs[:-1], ys[:-1], color=col, lw=lw, ls=ls, solid_capstyle='butt', zorder=1.8)
        self.ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle='-|>', mutation_scale=8, color=col, lw=lw,
                                          linestyle=ls, shrinkA=0, shrinkB=0.5, zorder=1.8))

    def _width(self, t):
        r = self.fig.canvas.get_renderer()
        return t.get_window_extent(r).width * 72 / self.fig.dpi

    def legend(self, x, y, w, items):
        """Bottom strip: tab + items laid out left to right by their measured width."""
        self.ax.add_patch(FancyBboxPatch((x, y), w, 22, boxstyle='round,pad=0,rounding_size=4',
                                         fc='white', ec=INK, lw=0.9, ls=(0, (4, 2.5))))
        t = self.ax.text(x + 7, y + 11, 'LEGEND', fontsize=SMALL, color='white', weight='bold', va='center', zorder=3)
        self.fig.canvas.draw()
        self.ax.add_patch(FancyBboxPatch((x, y), self._width(t) + 14, 22, boxstyle='round,pad=0,rounding_size=3',
                                         fc=INK, ec=INK, lw=0.9, zorder=2))
        cx = x + self._width(t) + 24
        for kind, col, label, *ls in items:
            if kind == 'arrow':
                self.arrow([(cx, y + 11), (cx + 16, y + 11)], col=col, ls=ls[0] if ls else '-')
                cx += 20
            elif kind == 'swatch':
                self.ax.add_patch(FancyBboxPatch((cx, y + 6), 10, 10, boxstyle='round,pad=0,rounding_size=1.5',
                                                 fc=col, ec=col, zorder=2))
                cx += 14
            t = self.text(cx, y + 11, label, (x, y, x + w, y + 22), color=col if kind == 'text' else INK)
            self.fig.canvas.draw()
            cx += self._width(t) + 14

    def done(self, name):
        self.fig.canvas.draw()
        r = self.fig.canvas.get_renderer()
        to_pt = self.ax.transData.inverted()
        for t, (x0, y0, x1, y1) in self.fits:
            (a, b), (c, d) = to_pt.transform(t.get_window_extent(r).get_points())
            assert x0 + 1 <= a and c <= x1 - 1 and y0 <= b and d <= y1, f'{name}: "{t.get_text()}" leaves its box'
        save(self.fig, name)


def fig_pipeline():
    cv = _Canvas(515, 214)
    rl, se, di = ENTITY['G'], ENTITY['G+s3'], ENTITY['S3']
    wr = lambda v: f'ActiveWR {v}'
    # 1. RL training
    cv.panel(0, 112, 262, 100, '1  RL training against RuleBot', rl)
    b = cv.card(10, 122, 70, 66, 'B', ['masked PG', '+ teachers', wr('40.0')], ENTITY['B'], bold_last=True)
    f = cv.card(96, 122, 70, 66, 'F', ['B + ~3,000 ep', 'stronger teachers', wr('44.25')], ENTITY['F'], bold_last=True)
    g = cv.card(182, 122, 70, 66, 'G', ['F + group-', 'relative adv.', wr('48.25')], ENTITY['G'], bold_last=True)
    cv.arrow([(80, 155), (96, 155)]); cv.arrow([(166, 155), (182, 155)])
    # 2. expert = G + kill-search at play time (no weights change)
    cv.panel(274, 112, 241, 100, '2  Expert: G + kill-search at play time', se)
    for i, (k, d, v, extra) in enumerate([('G+s1', 1, '64.75', ''), ('G+s2', 2, '82.0', ''),
                                          ('G+s3', 3, '94.0', '213 ms / move')]):
        cv.card(284 + 76 * i, 122, 68, 66, k, [f'depth {d}', extra or ' ', wr(v)], ENTITY[k], bold_last=True)
    cv.arrow([(252, 155), (284, 155)], col=se, ls=(0, (3, 2)))
    # 3. dense distillation: init weights come from above, labels from the named teacher
    cv.panel(0, 28, 515, 76, '3  Dense distillation', di)
    teach = lambda s: (f'teacher {s}', ENTITY['G+s2'])
    s = cv.card(96, 34, 70, 54, 'S', [teach('F+s1'), wr('60.5')], ENTITY['S'], bold_last=True)
    s2 = cv.card(182, 34, 70, 54, 'S2', [teach('G+s2'), wr('83.0')], ENTITY['S2'], bold_last=True)
    s3 = cv.card(276, 34, 84, 54, 'S3  (main agent)', [teach('S2+s3'), '1.9 ms / move', wr('92.25')], ENTITY['S3'],
                 bold_last=True)
    cv.arrow([(131, 122), (131, 88)]); cv.arrow([(217, 122), (217, 88)]); cv.arrow([(252, 61), (276, 61)])
    note = (372, 34, 508, 88)
    for i, line in enumerate(['each round: label 6.2k / 8.4k / 10.4k', 'states with the teacher\'s best set B(s);',
                              'soft-target CE + KL anchor (λ = 1);', 'epoch chosen on validation']):
        cv.text(376, 80 - 12.5 * i, line, note, color=MUTED)
    cv.legend(0, 0, 515, [('arrow', INK, 'initial weights'), ('arrow', se, 'adds search at play time', (0, (3, 2))),
                          ('text', ENTITY['G+s2'], 'teacher whose labels are distilled'),
                          ('text', MUTED, 'ActiveWR: % vs RuleBot, own test block')])
    cv.done('fig_pipeline')


def fig_architecture():
    import torch
    from train_actor_critic import ActorCriticNet
    net = ActorCriticNet(board_size=9)
    n_params = sum(p.numel() for p in net.parameters())
    out = net(torch.zeros(1, 13, 9, 9))
    assert out[0].shape == (1, 6) and out[1].shape == (1,)
    DATA['architecture'] = {'parameters': int(n_params)}
    enc = head = '#3b3a36'  # neutral: the network is shared by every agent
    cv = _Canvas(515, 128)
    cv.panel(0, 28, 398, 98, '1  Shared encoder (padding 1 keeps 9×9)', enc)
    blocks = [('Input', ['13 planes', '9×9']), ('Conv 1', ['3×3, 32, ReLU', '→ 32×9×9']),
              ('Conv 2', ['3×3, 64, ReLU', '→ 64×9×9']),
              ('Conv 3', ['3×3, 64, ReLU', '→ 64×9×9']),
              ('Flatten', ['5,184']), ('FC', ['256, ReLU'])]
    x, w, gap = 8, 52, 12
    for i, (name, lines) in enumerate(blocks):
        cv.card(x, 40, 56 if i in (1, 2, 3) else w, 58, name, lines, '#8f8e88' if i in (0, 4) else enc)
        x += (56 if i in (1, 2, 3) else w) + gap
        if i < len(blocks) - 1:
            cv.arrow([(x - gap, 69), (x, 69)])
    fc_right = x - gap
    cv.panel(410, 28, 105, 98, '2  Heads', head)
    cv.card(418, 70, 89, 38, 'Actor', ['6 logits, safe mask', '→ softmax'], head)
    cv.card(418, 34, 89, 30, 'Critic', ['V(s)'], '#8f8e88')
    mid = (fc_right + 418) / 2
    cv.arrow([(fc_right, 69), (mid, 69), (mid, 89), (418, 89)])
    cv.arrow([(mid, 69), (mid, 49), (418, 49)])
    cv.legend(0, 0, 515, [('text', INK, f'{n_params:,} parameters'),
                          ('text', MUTED, 'same network for B, F, G, S, S2 and S3'),
                          ('text', MUTED, 'unsafe actions are masked out of the logits before sampling')])
    cv.done('fig_architecture')


ALL = [fig_pipeline, fig_architecture, fig_observation, fig_gameplay, fig_milestones, fig_forest, fig_wr_vs_active,
       fig_training_curves, fig_distill_curves, fig_confusion, fig_episode_length, fig_cnn_matchups,
       fig_draw_probe, fig_search_depth]

if __name__ == '__main__':
    only = set(sys.argv[1:])
    for fn in ALL:
        if not only or fn.__name__ in only:
            fn()
    path = OUT / 'figure_data.json'
    old = json.load(open(path, encoding='utf-8')) if path.exists() and only else {}
    old.update(DATA)
    path.write_text(json.dumps(old, indent=1, default=float), encoding='utf-8')
    print('wrote figure_data.json')
