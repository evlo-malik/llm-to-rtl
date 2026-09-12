#!/usr/bin/env python3
"""Plot mapped area from syn/compare.py; requires matplotlib."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('report',type=Path);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    data=json.loads(args.report.read_text())
    rows={(r['bits'],r['kind']):r['cell_area']/1e6 for r in data['results']}
    fig,ax=plt.subplots(figsize=(7,3.8),layout='constrained')
    x=[0,1,2];bits=[8,4,2]
    ax.bar([i-.18 for i in x],[rows[b,'programmable'] for b in bits],width=.36,color='#a6adb4',label='Programmable weights')
    ax.bar([i+.18 for i in x],[rows[b,'fixed'] for b in bits],width=.36,color='#2463a5',label='Fixed weights')
    ax.set_xticks(x,['INT8','INT4','Ternary'])
    ax.set_ylabel('Sky130 library area (millions of units)')
    ax.set_title(f"Full {data['shape'][0]} × {data['shape'][1]} query projection")
    ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False)
    fig.savefig(args.out,metadata={'Date':None} if args.out.suffix=='.svg' else {})
    plt.close(fig)


if __name__=='__main__':main()
