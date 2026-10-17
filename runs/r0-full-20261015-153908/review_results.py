"""Review R0 raw accounting, legacy reward parity, timeout tapes and plot source data."""
from collections import Counter, defaultdict
import csv,gzip,hashlib,importlib.util,json,math
from itertools import combinations
from pathlib import Path
import sys,zipfile
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import audit_r0 as audit
from audit_t2 import digest,dump
from visualize import capture_state
OUT=Path(sys.argv[1]).resolve()
cfg=json.loads((OUT/'config.json').read_text(encoding='utf-8'))
s=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
assert s['status']=='complete' and s['full_protocol'] and s['games_completed']==300
assert s['training_updates']==0 and s['validation_parity'] and s['hashes_unchanged']
for name,sha in cfg['source_sha256'].items():
    assert digest(ROOT/name)==digest(OUT/'source'/name)==sha
for record in cfg['checkpoints'].values():
    assert digest(Path(record['path']))==record['sha256']
for path,sha in cfg['parent_input_sha256'].items():
    assert digest(Path(path))==sha
before=ROOT/'runs/r0-implementation-20261015/before/train_actor_critic.py'
parent=json.loads((audit.PARENT/'config.json').read_text())
assert digest(before)==parent['source_sha256']['train_actor_critic.py']
spec=importlib.util.spec_from_file_location('legacy_reward_before_r0',before)
legacy=importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
with zipfile.ZipFile(str(OUT)+'-send.zip') as z:
    assert z.testzip() is None
    for item in z.infolist():
        if not item.is_dir(): assert z.read(item)==(OUT/item.filename).read_bytes(),item.filename
    archive_files=sum(not x.is_dir() for x in z.infolist())
stats=json.loads((OUT/'statistics.json').read_text())
all_rows,step_count,maximum_error={ },0,0.
discounted,late_windows={},{}
gamma=parent["gamma"]
assert gamma==.99
for name in cfg['checkpoints']:
    rows=json.loads((OUT/f'games_{name}.json').read_text())
    assert [(r['seed'],r['aid']) for r in rows]==[(seed,aid) for seed in range(21000,21050) for aid in (0,1)]
    audit.validation_parity(rows,cfg['expected_validation'][name])
    discounted[name],late_windows[name]=[],[]
    per_game=defaultdict(list)
    for line in gzip.open(OUT/f'steps_{name}.jsonl.gz','rt',encoding='utf-8'):
        step=json.loads(line)
        per_game[(step['seed'],step['aid'])].append(step)
        step_count+=1
    assert len(per_game)==100
    for row in rows:
        parts,rewards,raws,kills=Counter(),[],[],0
        tape=per_game[(row['seed'],row['aid'])]
        assert [t['step'] for t in tape]==list(range(1,row['learner_steps']+1))
        for t in tape:
            aid=row['aid']
            info=t['info']
            info['kills']=[tuple(x) for x in info['kills']]
            for key in ('items_collected','walls_destroyed'):
                info[key]={int(k):v for k,v in info[key].items()}
            kills+=sum(k==aid and v!=aid for k,v in info['kills'])
            expected=legacy._combat_reward(t['env_reward'],aid,info,kills)
            if aid in info['walls_destroyed'] and info['alive'][aid]:
                expected+=.15*info['walls_destroyed'][aid]
            assert expected==t['shaped_reward']
            assert math.isclose(sum(t['reward_components'].values()),expected,rel_tol=0,abs_tol=1e-10)
            parts.update(t['reward_components'])
            rewards.append(expected);raws.append(t['env_reward'])
            maximum_error=max(maximum_error,abs(sum(t['reward_components'].values())-expected))
            assert t['mask'][t['actions'][aid]]
        assert kills==row['kills'] and parts==Counter(row['reward_components'])
        assert sum(rewards)==row['shaped_reward'] and sum(raws)==row['env_reward']
        for key in audit.COMPONENTS:
            assert math.isfinite(parts[key])
        assert row['pressure_bombs']+row['wood_only_bombs']+row['neither_bombs']==row['bombs']
        assert row['close_steps']==sum(t['distance']<=3 for t in tape)
        assert row['walls_all']==sum(t['info']['walls_destroyed'].get(row['aid'],0) for t in tape)
        assert row['walls_rewarded']==sum(t['info']['walls_destroyed'].get(row['aid'],0) if t['info']['alive'][row['aid']] else 0 for t in tape)
        assert row['initial_distance']==tape[0]['distance'] and row['min_distance']==min(t['distance'] for t in tape)
        def discounted_sum(values):
            carry=0.
            for value in reversed(values): carry=float(value)+gamma*carry
            return carry
        component_returns={key:discounted_sum([t['reward_components'][key] for t in tape]) for key in audit.COMPONENTS}
        g0=discounted_sum([t['shaped_reward'] for t in tape])
        assert math.isclose(sum(component_returns.values()),g0,rel_tol=0,abs_tol=1e-10)
        discounted[name].append({'seed':row['seed'],'aid':row['aid'],'outcome':row['outcome'],
            'return_t0':g0,'components':component_returns})
        if row['timeout']:
            last=tape[-100:]
            placements=[t for t in last if row['aid'] in t['info']['bombs_placed']]
            late_windows[name].append({'seed':row['seed'],'aid':row['aid'],'steps':len(last),
                'reward':sum(t['shaped_reward'] for t in last),
                'wall_reward':sum(t['reward_components']['wall'] for t in last),
                'item_reward':sum(t['reward_components']['item'] for t in last),
                'close_fraction':sum(t['distance']<=3 for t in last)/len(last),
                'bombs':len(placements),'pressure_bombs':sum(t['pressure']>0 for t in placements),
                'wood_only_bombs':sum(t['pressure']<=0 and t['projected_wood']>0 for t in placements),
                'stop_fraction':sum(t['actions'][row['aid']]==0 for t in last)/len(last),
                'approach_available_fraction':sum(t['approach_action'] is not None for t in last)/len(last),
                'approach_chosen_fraction':sum(t['approach_action'] is not None and t['actions'][row['aid']]==t['approach_action'] for t in last)/len(last)})
    recomputed={'all':audit.describe(rows),'by_outcome':{g:audit.describe([r for r in rows if r['outcome']==g]) for g in audit.GROUPS},'by_corner':{str(a):audit.describe([r for r in rows if r['aid']==a]) for a in (0,1)}}
    assert recomputed==stats[name] and s['models'][name]==recomputed['all']
    all_rows[name]=rows
    print(name,'accounting passed',sum(len(v) for v in per_game.values()),'steps',flush=True)
cases=json.loads((OUT/'timeout_cases.json').read_text())
assert len(cases)==s['timeout_cases']==12
frames_checked=0
canonical=lambda value:json.loads(json.dumps(value))
for case in cases:
    game=json.loads(gzip.open(OUT/case['path'],'rt',encoding='utf-8').read())
    assert game['game']['timeout']==1 and len(game['trace'])==500 and len(game['frames'])==501
    env=audit.BombermanFFAEnv(board_size=9,num_agents=2,max_steps=500,seed=case['seed'])
    env.reset()
    for tick,frame in enumerate(game['frames']):
        assert canonical(capture_state(env,frame['labels'],frame['actions'],frame['rewards']))==frame
        frames_checked+=1
        if tick<500:
            trace=game['trace'][tick]
            _,raw,_,info=env.step(trace['actions'])
            assert canonical(info)==trace['info']
            assert raw[case['aid']]==trace['env_reward']
    assert env._episode_done and all(a.alive for a in env.agents)
manifest=json.loads((OUT/'figure_manifest.json').read_text())
assert manifest['data_sha256']==digest(OUT/'figure_data.csv') and manifest['points']==300
with (OUT/'figure_data.csv').open(encoding='utf-8',newline='') as f:
    chart=list(csv.DictReader(f))
assert len(chart)==300
source=[dict(model=n,**r) for n,rows in all_rows.items() for r in rows]
for actual,raw in zip(chart,source):
    assert actual['model']==raw['model'] and actual['outcome']==raw['outcome']
    for key in ('seed','aid','kills','active_win','shaped_reward'): assert float(actual[key])==raw[key]
tolerance=1e-10
whatifs={}
for name,rows in all_rows.items():
    timeouts=[r for r in rows if r['timeout']]
    whatifs[name]={}
    for label,fn in [('observed',lambda r:r['shaped_reward']),('wall_0.075',lambda r:r['shaped_reward']-.075*r['walls_rewarded']),('item_0.025',lambda r:r['shaped_reward']-.025*r['items'])]:
        values=[fn(r) for r in timeouts]
        whatifs[name][label]={'n':len(values),'distribution':audit.distribution(values),
            'positive_count':sum(v>tolerance for v in values),
            'near_zero_count':sum(abs(v)<=tolerance for v in values),
            'negative_count':sum(v<-tolerance for v in values)}
analysis={'status':'reviewed','archive_files':archive_files,'source_and_checkpoints_pass':True,
    'games':300,'learner_steps_checked':step_count,'max_component_error':maximum_error,
    'timeout_cases_replayed':len(cases),'timeout_frames_replayed':frames_checked,
    'chart_rows_verified':len(chart),'training_updates':0,'heldout_tests_opened':False,
    'fixed_action_timeout_accounting':whatifs,
    'positive_tolerance':tolerance,
    'discounted_gamma':gamma,'discounted_returns_file':'discounted_returns.json',
    'timeout_last100_file':'timeout_last100.json',
    'counterfactual_limit':'Reward arithmetic on unchanged trajectories, not a performance prediction',
    'decision':'retain_original'}
dump(OUT/'discounted_returns.json',discounted)
dump(OUT/'timeout_last100.json',late_windows)
analysis['discounted_summaries']={}
for name,rows in discounted.items():
    analysis['discounted_summaries'][name]={}
    for group in ('all',*audit.GROUPS):
        subset=[r for r in rows if group=='all' or r['outcome']==group]
        analysis['discounted_summaries'][name][group]={'games':len(subset),
            'return_t0':audit.distribution([r['return_t0'] for r in subset]),
            'components':{key:audit.distribution([r['components'][key] for r in subset]) for key in audit.COMPONENTS}}
analysis['timeout_last100_summaries']={name:{'games':len(rows),'stats':{key:audit.distribution([r[key] for r in rows]) for key in ('reward','wall_reward','item_reward','close_fraction','bombs','pressure_bombs','wood_only_bombs','stop_fraction','approach_available_fraction','approach_chosen_fraction')},'pressure_bomb_fraction':sum(r['pressure_bombs'] for r in rows)/sum(r['bombs'] for r in rows) if sum(r['bombs'] for r in rows) else None} for name,rows in late_windows.items()}
rankings=[]
ranking_summary={}
for left,right in combinations(discounted,2):
    index={(r['seed'],r['aid']):r for r in discounted[right]}
    pair=[]
    for x in discounted[left]:
        y=index[x['seed'],x['aid']]
        if {x['outcome'],y['outcome']}!={'active_win','timeout'}:
            continue
        win,draw=(x,y) if x['outcome']=='active_win' else (y,x)
        winner=left if x['outcome']=='active_win' else right
        loser=right if x['outcome']=='active_win' else left
        entry={'seed':x['seed'],'aid':x['aid'],'winning_model':winner,'timeout_model':loser,
            'win_G0':win['return_t0'],'timeout_G0':draw['return_t0'],
            'observed_gap':draw['return_t0']-win['return_t0'],
            'wall_half_gap':draw['return_t0']-.5*draw['components']['wall']-(win['return_t0']-.5*win['components']['wall']),
            'item_half_gap':draw['return_t0']-.5*draw['components']['item']-(win['return_t0']-.5*win['components']['item'])}
        pair.append(entry)
    ranking_summary[left+'__'+right]={'eligible_pairs':len(pair),
        **{key:sum(r[key]>tolerance for r in pair) for key in ('observed_gap','wall_half_gap','item_half_gap')}}
    rankings.extend(pair)
dump(OUT/'paired_outcome_rankings.json',rankings)
analysis['paired_outcome_ranking_summaries']=ranking_summary
dump(OUT/'review_results.json',analysis)
print(json.dumps({k:v for k,v in analysis.items() if k not in ('discounted_summaries','timeout_last100_summaries')},indent=2))
