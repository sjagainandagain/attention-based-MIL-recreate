# Attention-based Deep MIL 재현 프로젝트 — 진행 노트

이 문서는 Claude(claude.ai 채팅)와 함께 진행한 작업 전체를 정리한 것입니다.
README 작성 시 이 문서를 참고 자료로 사용해주세요.

## 프로젝트 목표

Ilse et al. (2018), "Attention-based Deep Multiple Instance Learning" 논문을
직접 재현하는 것이 목표입니다. 새로운 모델을 제안하는 게 아니라, 논문의 핵심
주장 — "attention 가중치가 bag 라벨을 결정짓는 instance를 정확히 짚어낸다" —
을 작은 벤치마크에서 직접 확인하는 데 초점을 맞췄습니다.

배경: 의과대학생(예과)으로, 전역 후 학부연구생 지원을 위한 포트폴리오로
진행했습니다. 전공 지식 없이 자기주도 학습으로 시작했고, 코드 작성은 대부분
Claude의 도움을 받았습니다 — 이 점은 README에 솔직하게 명시하는 걸 권합니다.

## 왜 MIL인가 (개념 요약)

Multiple Instance Learning(MIL)은 "가방(bag)" 하나에 여러 "인스턴스(instance)"가
들어있고, bag 전체에만 라벨이 붙는 약지도학습(weakly supervised) 세팅입니다.
병리 슬라이드 한 장(bag) 안에 패치 여러 개(instance)가 있고, 그중 암세포
패치가 하나라도 있으면 슬라이드가 양성으로 판정되는 구조가 대표적입니다.

Attention-MIL은 각 instance마다 "중요도 점수"를 학습으로 매기고, softmax로
비율로 바꾼 뒤 가중평균을 내어 bag을 대표하는 벡터 하나를 만듭니다. 이
가중치가 바로 "모델이 무엇을 보고 판단했는지"를 보여주는 해석 가능성의
근거가 됩니다.

## 진행 순서 및 결과

### 1단계 — MNIST-bags 사고 검증 (sanity check)

실제 의료 데이터로 바로 들어가지 않고, MNIST 숫자 이미지를 bag으로 묶은
합성 데이터로 파이프라인부터 검증했습니다. bag 안에 숫자 9가 하나라도
있으면 라벨 1을 부여하는 구조입니다.

- 결과: test error 0.06 (94% 정확도)
- attention 검증: positive bag에서 9가 포함된 instance 2개에 전체 attention의
  약 99.95%(0.7418 + 0.2577)가 집중됨을 확인. 다른 숫자(7, 1, 2, 4, 5, 6)는
  전부 attention 거의 0.

이 결과는 "모델이 우연이 아니라 실제로 target instance를 찾아낸다"는
논문의 핵심 주장을 가장 단순한 형태로 재현한 것입니다.

### 2단계 — 실제 데이터셋: Colon Cancer

원 논문이 실제로 사용한 Colon Cancer 데이터셋(H&E 염색 조직 이미지에서
추출한 27×27 패치, 세포 종류별 라벨 포함)으로 넘어갔습니다. 공식
저장소(AMLab-Amsterdam/AttentionDeepMIL)가 안내하는 원본 소스는 접근이
막혀 있어, 이미 patch 단위로 전처리된 버전을 공개 구글 드라이브 링크를
통해 사용했습니다.

- 데이터 규모: 99 bags (positive 52%, 논문이 보고한 51/48 분포와 거의 일치)
- bag 구조: `Patches/<label>/<img_id>/*.bmp`, label 0/1, 파일명에 세포
  종류(epithelial, fibroblast, inflammatory 등)와 좌표 포함
- train/test 분할: 79 / 20 (8:2, seed 고정)

**실험 1 — 증강 없음**
- train acc: 94.94% (20 epoch), test acc: 75.00% (error 0.25)
- train-test 격차 19.94%p → 과적합 의심

**실험 2 — 데이터 증강 추가**
학습 데이터에만 RandomHorizontalFlip, RandomVerticalFlip,
RandomRotation(180도), ColorJitter(brightness/contrast/saturation 0.2) 적용.
테스트 데이터는 원본 그대로 유지(공정한 평가를 위해).

- train acc: 86.08% (40 epoch), test acc: 75.00% (동일)
- train-test 격차 11.08%p로 감소

**해석**: 증강이 암기(과적합)는 완화했지만 test 정확도 자체는 그대로였음.
이는 test bag이 20개뿐이라(1개 차이 = 5%p) 애초에 숫자 자체가 불안정하고,
동시에 데이터 양(79 bags) 자체가 성능의 한계로 보인다는 해석. 이 부분을
"정직한 한계점"으로 README에 명시할 가치가 있음.

### 3단계 — Attention이 올바른 근거를 보는지 검증

정확도만으로는 "모델이 옳은 이유로 판단했는지" 알 수 없다는 문제의식에서,
attention 가중치가 실제로 epithelial(상피) 세포 — 이 데이터셋에서 양성
판정의 생물학적 근거 — 에 집중되는지 검증했습니다.

**1차 검증 (단일 bag)**: positive bag 1개에서 attention 상위 3개가 전부
epithelial. 다만 이 bag 자체에 epithelial이 매우 흔해서(300개 중 다수),
우연일 가능성을 배제하기 어렵다는 한계를 스스로 지적하고 2차 검증으로 발전.

**2차 검증 (체계적 검증, 최종 버전)**: test에 속한 positive bag **9개 전체**에
대해 두 지표를 비교:
- 기준선(baseline): 그 bag에서 무작위로 뽑았을 때 epithelial일 확률
  (= bag 내 epithelial 비율)
- 모델의 몫(model share): attention 가중치 중 epithelial instance들이
  차지하는 합

결과:
```
bag   n_patch  baseline  model_share  diff
80    326      0.73      0.96         +0.23
92    332      0.85      0.96         +0.11
50    207      0.68      1.00         +0.32
57    110      0.80      1.00         +0.20
61    84       0.30      0.99         +0.70
95    359      0.38      1.00         +0.62
69    187      0.37      1.00         +0.63
91    692      0.14      0.74         +0.60
93    505      0.50      0.98         +0.48

모델의 몫이 기준선보다 높은 bag: 9개 중 9개
평균 차이: +0.431
```

**해석**: 9개 bag 전부에서 모델의 몫이 기준선을 상회. 특히 기준선이 낮은
bag(61: 0.30, 91: 0.14)에서도 attention이 거의 전부 epithelial에
집중되었다는 점이 우연이 아니라는 강한 근거. 이는 test accuracy(75%)가
논문보다 낮음에도, 모델이 "왜 그렇게 판단했는지"의 근거 자체는 일관되게
타당하다는 걸 보여줌 — 정확도와 해석 가능성은 별개로 평가해야 한다는
포인트.

**후속 관찰**: bag 91만 유독 model share가 0.74로 다른 bag(0.96~1.00)보다
낮음. 이 bag은 patch 수가 692개로 가장 많고 기준선도 가장 낮음(0.14).
patch 수가 많아지면 attention이 분산되는 경향이 있는지는 확인 안 된 흥미로운
후속 질문으로 남겨둠.

## 코드 구조

| 파일 | 역할 |
|---|---|
| `mnist_bags.py` | MNIST-bags 합성 데이터셋 (instance별 실제 숫자 레이블도 저장) |
| `mil_model.py` | Attention MIL 모델 (in_channels 파라미터로 MNIST/Colon 모두 대응, AdaptiveAvgPool2d로 patch 크기 무관하게 동작) |
| `colon_cancer_bags.py` | Colon Cancer 데이터셋 로더 (`Patches/<label>/<img_id>/*.bmp` 구조) |
| `train.py` | MNIST-bags 학습 스크립트 |
| `train_colon.py` | Colon Cancer 학습 스크립트 (train/test 다른 transform 적용, 증강 포함) |
| `visualize_attention.py` | MNIST 단일 bag attention 확인용 (초기 버전) |
| `visualize_attention_colon.py` | Colon Cancer 단일 bag attention 확인용 (초기 버전) |
| `verify_attention_colon.py` | test positive bag 전체에 대한 체계적 attention 검증 (최종 분석, 위 "2차 검증" 결과를 생성한 스크립트) |

## 데이터 출처

- MNIST: torchvision 기본 제공
- Colon Cancer: 공식 저장소(AMLab-Amsterdam/AttentionDeepMIL)가 안내하는
  Warwick 원본은 로그인이 필요해 접근 실패. 대신 이미 patch 단위로 전처리된
  공개 구글 드라이브 버전 사용 (`Patches.tar.gz`, gdown으로 다운로드)

## README에 포함하면 좋을 내용 (제안)

1. 프로젝트 목적과 재현 대상 논문
2. MIL 개념 간단 설명 (bag/instance, attention pooling)
3. 결과 표 (MNIST sanity check → Colon Cancer → attention 검증)
4. **정직한 한계점**: test set이 작아 평가가 불안정함(20 bags), 교차검증
   미실시, 증강해도 test acc 정체, bag 91 같은 예외 사례
5. 코드 구조 설명
6. AI 도구 사용 범위에 대한 투명한 명시
7. 재현 방법 (Colab 링크 또는 실행 순서)

## 교차검증 (2026-10 ~ 11)

- 이유: 단일 split test 20 bags → 75%가 불안정. 증강 효과도 판단 불가
- 방법: 층화 5-fold × 시드 3개, 증강 없음/증강을 같은 분할에서 비교 (`cv_colon.py`)
- 기존 `train_colon.py`와 차이: epoch마다 bag 순서 셔플, AUC 추가 기록, fold마다 attention 검증
- 결과: test acc 0.801 → 0.835 (증강), 증강 효과는 격차 감소만 확인 (성능 향상은 우연과 구분 안 됨)
- 75%는 운이 나쁜 분할 하나였음 (fold별 범위 0.55–0.95)
- attention: 모든 positive bag에서 153/153, 152/153
- 정확도와 AUC가 fold마다 반대로 움직이는 경우 있음 (예: 증강 시드 2 fold2 acc 0.80·AUC 0.97, fold3 acc 0.85·AUC 0.76) → 0.5 기준선과 보정 문제. bag별 확률 미저장이라 확인 X
- 실행 환경 교훈: 데이터는 드라이브에서 직접 읽지 말고 `/content`에 압축 해제. 결과만 드라이브 저장

## bag 91 분석 (2026-11)

- 질문: 패치가 많아서 attention이 분산되는가?
- 결과: 패치 수와 모델의 몫·엔트로피 무관 (ρ ≈ 0). epithelial 비율과 양의 상관 (ρ ≈ +0.55)
- 해석 주의: 균등 attention이면 몫 = 기준선 → 상관 일부는 당연
- 결론: bag 91은 패치 수가 아니라 epithelial이 드문 bag의 한 예

## 아직 안 한 것

- bag별 예측 확률 저장 → 보정 확인
- 특징 추출기 CNN → ViT 교체 (ViT scratch to MIL 프로젝트)
- ACMIL로의 확장 (선택사항, 보류)
