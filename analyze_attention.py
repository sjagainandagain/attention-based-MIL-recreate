"""
analyze_attention.py
bag 91 질문: attention이 epithelial에 덜 모이는 bag의 특징은?
- cv_attention.csv를 읽어 bag × 설정마다 시드 3개 평균 (같은 bag 반복 측정을 3번 세지 않도록)
- 패치 수 / epithelial 비율(기준선) vs 모델의 몫, 패치 수 vs attention 엔트로피
- 그림 저장: figures/attention_vs_bag.png

사용 (Colab): %run analyze_attention.py            # 기본 cv_results
             %run analyze_attention.py cv_results
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "cv_results"

att = pd.read_csv(os.path.join(OUT_DIR, "cv_attention.csv"))
bag = (att.groupby(["setting", "bag"])
          .agg(n_patch=("n_patch", "first"), baseline=("baseline", "first"),
               share=("share", "mean"), norm_entropy=("norm_entropy", "mean"))
          .reset_index())

# 상관 (Spearman: 순위 기반, 이상치에 덜 민감)
pairs = [("n_patch", "share"), ("baseline", "share"),
         ("n_patch", "baseline"), ("n_patch", "norm_entropy")]
for s in ["no_aug", "aug"]:
    b = bag[bag.setting == s]
    print(f"[{s}] positive bags = {len(b)}")
    for x, y in pairs:
        r, p = spearmanr(b[x], b[y])
        print(f"  {x:>8} vs {y:<12} rho = {r:+.2f}  (p = {p:.3f})")

# 그림 (Colab 기본 글꼴에 한글이 없어 영어 라벨 사용)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
style = {"no_aug": dict(marker="o", label="no aug"),
         "aug": dict(marker="^", label="aug")}
panels = [("n_patch", "share", "Patches per bag", "Attention share on epithelial"),
          ("baseline", "share", "Epithelial fraction (baseline)", "Attention share on epithelial"),
          ("n_patch", "norm_entropy", "Patches per bag", "Normalized attention entropy")]
for ax, (x, y, xl, yl) in zip(axes, panels):
    for s, kw in style.items():
        b = bag[bag.setting == s]
        ax.scatter(b[x], b[y], alpha=0.7, s=30, **kw)
    if (x, y) == ("baseline", "share"):
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="share = baseline")
    for _, r in bag[(bag.setting == "aug") & (bag.share < 0.5)].iterrows():
        ax.annotate(int(r.bag), (r[x], r[y]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(alpha=0.3)
axes[1].legend(loc="lower right")
fig.suptitle("Colon Cancer positive bags (mean of 3 seeds per bag)")
fig.tight_layout()

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/attention_vs_bag.png", dpi=150)
plt.show()
print("저장: figures/attention_vs_bag.png")

# 집중이 약한 bag 목록 (모델의 몫 < 0.5, 설정별)
print(bag[bag.share < 0.5].sort_values(["setting", "share"]).round(3).to_string(index=False))
