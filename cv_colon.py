"""
cv_colon.py
Colon Cancer 반복 층화 5-fold 교차검증

- 설정 2개: no_aug(증강 없음, 20 epochs) / aug(증강, 40 epochs) = README 실험 1, 2
- 시드 여러 개 × fold 5개 반복
- fold마다 기록: train acc, test acc, test AUC
- fold마다 test positive bag attention 검증 (verify_attention_colon.py와 같은 방식)
  + bag 91 질문용으로 패치 수, attention 엔트로피도 같이 저장
- 결과는 CSV에 run 하나 끝날 때마다 바로 저장
  -> Colab이 끊겨도 진행분 보존, 같은 명령으로 다시 실행하면 끝난 run은 건너뜀

사용 예
  python cv_colon.py --out_dir results_test --seeds 1 --settings aug --epochs 1   # 동작 확인용
  python cv_colon.py --out_dir cv_results --seeds 1 2 3                           # 본 실험
  python cv_colon.py --out_dir cv_results --summary_only                          # 요약만 다시 출력
"""

import os
import csv
import math
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from colon_cancer_bags import ColonCancerBags
from mil_model import AttentionMIL
from verify_attention_colon import cell_type_from_filename


# [1] 실험 설정: train_colon.py의 실험 1, 2와 동일
AUG_TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=180),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
])
PLAIN_TRANSFORM = transforms.Compose([transforms.ToTensor()])

SETTINGS = {
    "no_aug": {"transform": PLAIN_TRANSFORM, "epochs": 20},
    "aug":    {"transform": AUG_TRANSFORM,   "epochs": 40},
}
N_FOLDS = 5


def train_one(model, train_ds, optimizer, epochs, device, run_seed):
    """학습 후 마지막 epoch의 train acc 반환"""
    g = torch.Generator().manual_seed(run_seed)
    model.train()
    train_acc = 0.0
    for epoch in range(epochs):
        # [2] epoch마다 bag 순서를 섞음 (기존 train_colon.py는 고정 순서)
        order = torch.randperm(len(train_ds), generator=g).tolist()
        correct = 0
        for j in order:
            bag, label = train_ds[j]
            bag = bag.to(device)
            y = torch.tensor([float(label)], device=device)
            optimizer.zero_grad()
            prob, _, _ = model(bag)
            loss = F.binary_cross_entropy(prob.view(-1), y)
            loss.backward()
            optimizer.step()
            correct += int((prob.view(-1) > 0.5).float().item() == y.item())
        train_acc = correct / len(train_ds)
    return train_acc


def evaluate(model, test_ds, device):
    """test acc, AUC 반환"""
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for bag, label in test_ds:
            prob, _, _ = model(bag.to(device))
            probs.append(prob.view(-1).item())
            labels.append(label)
    probs, labels = np.array(probs), np.array(labels)
    acc = float(((probs > 0.5).astype(int) == labels).mean())
    # [3] AUC: 0.5 기준선 하나가 아니라 모든 기준선에서 양성/음성을 얼마나 잘 가르는지
    auc = float(roc_auc_score(labels, probs))
    return acc, auc


def attention_rows(model, ds_plain, test_idx, device):
    """test positive bag마다 attention 검증 결과 한 줄씩"""
    model.eval()
    rows = []
    with torch.no_grad():
        for i in test_idx:
            if ds_plain.labels[i] != 1:  # positive bag만
                continue
            bag, _ = ds_plain[i]
            _, _, attn = model(bag.to(device))
            attn = attn.view(-1).cpu()
            is_epi = torch.tensor(
                [cell_type_from_filename(p) == "epithelial" for p in ds_plain.bag_paths[i]]
            )
            n = len(attn)
            baseline = is_epi.float().mean().item()   # 기준선: epithelial 비율
            share = attn[is_epi].sum().item()         # 모델의 몫: epithelial attention 합
            # [4] 엔트로피: attention이 소수 패치에 몰리면 낮고, 고르게 퍼지면 높음
            #     norm_entropy = 엔트로피 / log(n) -> 0(한 패치에 집중) ~ 1(완전 균등)
            entropy = -(attn * torch.log(attn + 1e-12)).sum().item()
            norm_entropy = entropy / math.log(n) if n > 1 else 0.0
            rows.append({
                "bag": i, "n_patch": n,
                "baseline": round(baseline, 4), "share": round(share, 4),
                "diff": round(share - baseline, 4),
                "entropy": round(entropy, 4), "norm_entropy": round(norm_entropy, 4),
            })
    return rows


def append_rows(path, rows):
    """CSV에 이어 쓰기 (파일이 없으면 헤더부터)"""
    if not rows:
        return
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new_file:
            w.writeheader()
        w.writerows(rows)


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def summarize(fold_csv, attn_csv):
    folds = read_rows(fold_csv)
    attns = read_rows(attn_csv)
    if not folds:
        print("결과 없음")
        return
    print("\n===== 요약 (평균 ± 표준편차, fold 단위) =====")
    for setting in SETTINGS:
        fs = [r for r in folds if r["setting"] == setting]
        if not fs:
            continue
        tr = np.array([float(r["train_acc"]) for r in fs])
        acc = np.array([float(r["test_acc"]) for r in fs])
        auc = np.array([float(r["test_auc"]) for r in fs])
        gap = tr - acc
        seeds = sorted({r["seed"] for r in fs})
        print(f"\n[{setting}] runs={len(fs)} (seeds {', '.join(seeds)} × {N_FOLDS} folds)")
        print(f"  train acc : {tr.mean():.3f} ± {tr.std(ddof=1) if len(fs) > 1 else 0:.3f}")
        print(f"  test acc  : {acc.mean():.3f} ± {acc.std(ddof=1) if len(fs) > 1 else 0:.3f}")
        print(f"  test AUC  : {auc.mean():.3f} ± {auc.std(ddof=1) if len(fs) > 1 else 0:.3f}")
        print(f"  격차      : {gap.mean():.3f} ± {gap.std(ddof=1) if len(fs) > 1 else 0:.3f}")

        at = [r for r in attns if r["setting"] == setting]
        if at:
            d = np.array([float(r["diff"]) for r in at])
            print(f"  attention : 모델의 몫 > 기준선 {int((d > 0).sum())}/{len(d)} "
                  f"(positive bag × seed), 평균 차이 {d.mean():+.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="Patches")
    p.add_argument("--out_dir", default="cv_results")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--settings", nargs="+", default=list(SETTINGS), choices=list(SETTINGS))
    p.add_argument("--epochs", type=int, default=None, help="동작 확인용 epoch 덮어쓰기")
    p.add_argument("--summary_only", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fold_csv = os.path.join(args.out_dir, "cv_folds.csv")
    attn_csv = os.path.join(args.out_dir, "cv_attention.csv")

    if args.summary_only:
        summarize(fold_csv, attn_csv)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    ds_plain = ColonCancerBags(root=args.root, transform=PLAIN_TRANSFORM)  # 평가용 (증강 X)
    labels = np.array(ds_plain.labels)
    print(f"bags: {len(labels)}, positive 비율: {labels.mean():.2f}")

    done = {(r["setting"], r["seed"], r["fold"]) for r in read_rows(fold_csv)}

    for setting in args.settings:
        cfg = SETTINGS[setting]
        epochs = args.epochs or cfg["epochs"]
        ds_train_full = ColonCancerBags(root=args.root, transform=cfg["transform"])

        for seed in args.seeds:
            # [5] 층화(Stratified): fold마다 양성/음성 비율을 전체와 비슷하게 유지
            #     분할은 seed로만 정해짐 -> no_aug와 aug가 같은 분할에서 비교됨
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
            for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
                key = (setting, str(seed), str(fold))
                if key in done:
                    print(f"skip {key}")
                    continue

                run_seed = seed * 100 + fold
                torch.manual_seed(run_seed)  # 모델 초기값·증강 무작위성 고정
                model = AttentionMIL(in_channels=3).to(device)
                optimizer = optim.Adam(model.parameters(), lr=1e-4,
                                       betas=(0.9, 0.999), weight_decay=1e-4)

                train_acc = train_one(model, Subset(ds_train_full, tr_idx.tolist()),
                                      optimizer, epochs, device, run_seed)
                acc, auc = evaluate(model, Subset(ds_plain, te_idx.tolist()), device)
                arows = attention_rows(model, ds_plain, te_idx.tolist(), device)

                # [6] run 하나 끝날 때마다 바로 저장
                append_rows(attn_csv, [{"setting": setting, "seed": seed, "fold": fold, **r}
                                       for r in arows])
                append_rows(fold_csv, [{
                    "setting": setting, "seed": seed, "fold": fold, "epochs": epochs,
                    "n_train": len(tr_idx), "n_test": len(te_idx),
                    "train_acc": round(train_acc, 4), "test_acc": round(acc, 4),
                    "test_auc": round(auc, 4),
                }])
                print(f"[{setting} seed{seed} fold{fold}] train {train_acc:.3f} | "
                      f"test acc {acc:.3f} | AUC {auc:.3f} | positive bags {len(arows)}")

    summarize(fold_csv, attn_csv)


if __name__ == "__main__":
    main()
