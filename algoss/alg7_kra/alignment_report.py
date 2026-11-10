"""Report PN/core alignment WITHOUT clean TX reference (real-time safe)."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser(description='Inspect ALG7 TimePack timing/recovered 320-sample payloads')
    p.add_argument('--rx',type=Path,required=True,help='RX output directory (frames.csv, summary.json)')
    a=p.parse_args()
    csv_file=a.rx/'frames.csv'
    if not csv_file.is_file():p.error('Missing '+str(csv_file))
    with csv_file.open(newline='') as f:rows=list(csv.DictReader(f))
    if not rows:p.error('frames.csv is empty')
    idx=[int(row['frame_index']) for row in rows]
    ok=[row for row in rows if row['status']=='ok']
    lost=len(rows)-len(ok)
    corrected=[int(row['core_correction_samples']) for row in ok if row.get('core_correction_samples','') not in ('',None)]
    changed=[v for v in corrected if v!=0]
    estimated=sum(row.get('guard_source')=='estimated' for row in ok)
    corrs=[]
    for row in ok:
        try:corrs.append(float(row['corr']))
        except (ValueError,TypeError,KeyError):pass
    jumps=[(idx[i-1],idx[i]) for i in range(1,len(idx)) if idx[i]!=idx[i-1]+1]
    result={'frames_total':len(rows),'frames_ok':len(ok),'frames_lost':lost,
            'range':[min(idx),max(idx)],'nonsequential_indices':jumps[:20],
            'adjusted_core_frames':len(changed), 'max_core_adjust_samples':max((abs(v) for v in changed),default=0),
            'estimated_guard_frames':estimated, 'median_guard_correlation':statistics.median(corrs) if corrs else None,
            'pn_gain_median':None}
    gains=[]
    for row in ok:
        try:gains.append(float(row['pn_gain']))
        except (KeyError,ValueError,TypeError):pass
    if gains:result['pn_gain_median']=statistics.median(gains)
    summary=a.rx/'summary.json'
    if summary.is_file():
        data=json.loads(summary.read_text())
        result['sync_stats']=data.get('sync_stats')
        result['last_frame']=data.get('last_frame')
    print(json.dumps(result,indent=2,ensure_ascii=False))
    print('NOTE: a 64-frame ambiguity cannot be proven/removed from PN alone after mid-stream RX join.')
    print('NOTE: PN alignment restores frame boundaries, not exact payload samples after Opus/AAC.')

if __name__=='__main__':main()
