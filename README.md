# Minority-Class Weighted Classifier

**데이터가 적은 클래스에 학습 가중치를 더 주고, 집단별 예측 성능 차이를 함께 확인하는 ALINJob 기반 분류 코드입니다.**

채용 이력에는 대졸 지원자가 많고 다른 학력의 지원자는 적을 수 있습니다. 이때 모델은 대졸만 자주 예측해도 높은 정확도를 얻지만, 고졸 같은 소수 학력의 지원자를 잘 구분하지 못할 수 있습니다. 이 코드는 **표본이 적은 학력의 사례를 학습에서 더 중요하게 다루는 방식**으로 이러한 불균형을 완화합니다.

예를 들어 기준을 250건으로 정하면, 학습 표본이 50건인 클래스에는 5배, 100건인 클래스에는 2.5배의 가중치를 줍니다. 250건 이상인 클래스는 기존 가중치를 유지합니다. 행을 복제하는 오버샘플링과 달리 학습 손실에 반영할 가중치를 조정합니다.

성별 등 보호속성은 가중치 산출이나 모델 입력에 사용하지 않습니다. 보호속성은 평가에만 사용하여 적용 전후 집단별 TPR과 DEO를 비교합니다. **클래스 불균형 완화가 집단 공정성 향상을 보장하지는 않으므로 두 가지를 함께 평가합니다.**

## 알고리즘 정보 (조사 회신용)

- **알고리즘명:** ALINJob — 소수 클래스 선택 가중 (Minority-Class Selective Weighting)
- **분류:** 전처리(Preprocessing) 기반 표본 가중치 조정. 원 ALINJob에서는 LightGBM 학습에 적용합니다.
- **설명:** 학습 데이터에서 표본이 적은 클래스에 더 높은 가중치를 부여해 다수 클래스에 치우친 학습을 완화합니다. 보호속성은 모델 입력과 가중치 계산에 사용하지 않고, 적용 전후 집단별 TPR과 DEO를 비교해 공정성 변화를 평가합니다.
- **적용 가능 데이터셋:** SCOUT 최종학력 분류 데이터(비공개), 연구실 생성 합성 데이터(`data/synthetic.csv`, 생성 방법은 [합성 데이터 생성 방법](#합성-데이터-생성-방법) 참고)
- **GitHub:** https://github.com/seungkukim/minority-class-weighted-classifier
- **코드:** `algorithms/preprocessing/MinorityClassWeighting.py`

관련 논문은 별도로 기재하지 않습니다.

## 방법

학습 데이터에서 클래스 `c`의 원시 행 수를 `n_c`라고 하면:

```text
class_weight(c) = max(1, floor / n_c)
new_instance_weight = original_instance_weight * class_weight(c)
```

- 가중치 산출은 전처리(Preprocessing) 구성요소로 제공합니다. ALINJob에서는 이를 LightGBM의 가중 학습에 적용합니다.
- 가중치 함수는 이진·다중 클래스에 사용할 수 있습니다. MAF 실행 예제는 AIF360 이진 분류와 Logistic Regression을 사용합니다.
- `fit()`은 학습 클래스 배율을 저장하고 `transform()`은 이를 재사용합니다. 평가 가중치는 변경하지 않습니다.
- 두 모델은 같은 분할과 특징을 사용합니다. StandardScaler는 학습 데이터에만 적합합니다.

```python
from algorithms.preprocessing.MinorityClassWeighting import (
    minority_weights, MinorityClassWeighting, ALINJobMinorityWeighting,
)

# 일반 분류기의 sample_weight로 사용 (학습 레이블만 사용)
weights = minority_weights(y_train, floor=250)
classifier.fit(X_train, y_train, sample_weight=weights)

# AIF360 학습 데이터 변환
weighted_train = MinorityClassWeighting(floor=250).fit_transform(aif_train)

# MAF 가이드 방식: 적용 전후 지표 사전 반환
before, after = ALINJobMinorityWeighting(dataset_name="synthetic").run()
```

위의 `classifier`, `X_train`, `y_train`, `aif_train`은 사용자가 준비한 객체입니다. 기존 표본 가중치가 있다면 AIF360 변환기는 이를 곱해 보존합니다. 빈도 기준은 가중치 합이 아닌 행 수입니다.

### 지표 정의

```text
DEO = abs(TPR_unprivileged - TPR_privileged)
accuracy_gap_pp = 100 * abs(accuracy_weighted - accuracy_unweighted)
```

DEO `0.0136`은 `1.36%p`이며, 정확도 차이는 상대 감소율이 아닌 퍼센트포인트입니다. 두 집단 모두 양성을 거의 인식하지 못해도 작은 DEO가 나올 수 있으므로 소수 클래스 recall을 함께 봅니다.

## 설치 및 실행

Python 3.10, CPU 환경 기준입니다. 아래 명령은 저장소에 포함된 합성 데이터로 실행하며, 별도 데이터 다운로드가 필요 없습니다.

```bash
git clone https://github.com/seungkukim/minority-class-weighted-classifier.git
cd minority-class-weighted-classifier
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py --dataset synthetic --protected sex --floor 2000 --seed 1
```

JSON 형식의 적용 전후 지표를 출력합니다. `--floor`를 생략하면 합성 데이터는 2000, 그 외에는 250을 사용합니다. 파일로 저장하거나 검증·제출 파일을 만들려면 다음과 같이 실행합니다.

```bash
python run.py --dataset synthetic --output outputs.json
python validate.py
python build_submission.py
```

`validate.py`는 회귀 테스트를 실행하며, 합성 데이터에서 seed 1~5 모두 정확도 85% 초과, 가중 DEO 5%p 미만, 가중 전후 정확도 차이 3%p 미만임을 검사합니다. `build_submission.py`는 제출용 ZIP과 파일별 SHA256 manifest를 만듭니다.

## 데이터

| 데이터 | 공개 여부 | 용도 | 위치 |
|---|---|---|---|
| 합성 데이터 (20,000건) | 포함 | 실행 예제·회귀 테스트 | `data/synthetic.csv` |
| SCOUT 최종학력 (약 9,991건) | 비공개 | 기존 ALINJob 실험 | 없음 (이용 협의 필요) |

### 합성 데이터 (포함)

`data/synthetic.csv`는 고정 RNG(seed=1)로 생성한 가상 레코드 20,000건입니다. 실제 지원자 정보나 SCOUT 레코드를 변형·증강한 데이터가 아니며, 처음부터 가상으로 생성했습니다.

| 열 | 의미 |
|---|---|
| `x1`, `x2` | 가상의 숫자형 특징 |
| `sex` | 0/1 집단 평가용 합성 변수; 모델 입력에서 제외 |
| `label` | 0/1 분류 레이블; 양성 10% (2,000건) |

#### 합성 데이터 생성 방법

`data/generate_synthetic.py`가 다음 순서로 생성합니다. `python data/generate_synthetic.py`로 같은 파일을 다시 만들 수 있습니다.

1. `numpy.random.default_rng(1)`로 난수를 고정하고 20,000건을 만듭니다.
2. 특징 `x1`, `x2`는 서로 독립인 표준정규분포 N(0, 1)에서 추출합니다.
3. 보호속성 `sex`는 0/1을 같은 확률로 독립 추출합니다(실제 양성률: sex=0 약 9.9%, sex=1 약 10.1%).
4. 잠재 점수를 `latent = 1.2·x1 + 0.6·x2 + ε`로 계산합니다. `ε`는 N(0, 1) 잡음입니다. **`sex`는 점수에 포함하지 않으므로 레이블과 독립입니다.**
5. 잠재 점수 상위 10%를 `label=1`, 나머지를 `label=0`으로 정합니다. 클래스 불균형(양성 10%)이 가중치의 대상이 됩니다.

이 방식은 SCOUT의 분포를 모사하지 않는 시연용 예제입니다. 보호속성 효과가 없는 데이터이므로 낮은 DEO가 알고리즘이 실제 편향을 줄였다는 근거는 아니며, 공정성 수치를 알고리즘의 일반적 효과로 해석하지 마십시오.

### SCOUT (비공개)

원 ALINJob은 SCOUT 이력사항 DB의 부표본 약 9,991행·59개 원본 열을 사용해 최종학력을 5개 클래스(고졸·전문대·대졸·석사·박사)로 예측합니다. SCOUT 원본/전처리 CSV, 이력 텍스트, 식별자 및 학습 모델은 공개하지 않습니다. 데이터는 소유자와 별도 이용 협의를 통해 확보해야 하며 공개 다운로드 링크는 없습니다.

기존 실험 코드를 `experiments/scout/`에 보존했습니다. `source_manifest.json`의 해시는 이 저장소에 복사한 원본 코드 사본을 식별합니다. 단일 행의 수치까지 재현하려면 기존 `data_stage1.csv`와 같은 전처리·행 순서의 파일이 필요합니다.

```bash
python -m pip install -r requirements-scout.txt
cd experiments/scout
OMP_NUM_THREADS=2 python -B test_edu_gbm.py --data /absolute/path/data_stage1.csv --skip-point
```

`--skip-point`는 기존 pickle을 읽지 않고 5-fold OOF를 다시 학습합니다. LightGBM 코드의 `n_jobs=-1`로 사용 가능한 CPU를 활용할 수 있으므로 클러스터 자원 정책에 맞춰 실행하십시오. 이 데이터가 없어도 기본 MAF 예제는 실행됩니다.

#### SCOUT 입력 스키마

보존된 `train_edu_gbm.py` 기준 필수 열은 다음과 같습니다. 일반적인 다른 CSV를 그대로 대입하는 인터페이스가 아니라 기존 SCOUT 전처리 출력에 대한 재현 코드입니다.

| 역할 | 열 |
|---|---|
| 목표 | `최종학력` (기존 세분류 레이블; 코드에서 5분류로 병합) |
| 평가 집단 | `학교명` ('여자' 포함 여부; 모델 특징에서 제외) |
| 숫자 파싱 | `희망연봉`, `최종연봉`, `전체경력`, `졸업년도`, `입학년도`, `평점`, `만점` |
| 범주형 | `희망(경력)업종1`, `희망(경력)직종1`, `희망(경력)직종2`, `근무형태`, `계열`, `전공`, `희망(경력)세부직종1`, `근무지역` |
| 파생 개수 | `직무태그` (문자열 자체가 아닌 쉼표로 구분한 태그 개수) |

보호집단 대리변수는 실제 성별 전체를 뜻하지 않습니다. 장애 여부는 이 실험의 보호속성으로 검증하지 않았습니다. 학력 병합은 기존 코드의 목표 정의이며 원래 세분류 정확도와 직접 비교하면 안 됩니다.

## 사용법 (MAF 연동)

첨부 MSIT AI FAIR 가이드라인(2024-09-06)의 전처리 예제에 따라 AIF360 데이터, Logistic Regression 비교 모델, `run()`의 두 지표 사전 반환 구조를 사용합니다.

### 코드 배치

1. `algorithms/preprocessing/MinorityClassWeighting.py`를 MAF 저장소의 동일 경로에 복사합니다.
2. MAF의 기존 `__init__.py` 파일은 유지합니다.
3. 이 저장소의 `requirements.txt`를 별도 환경에 설치하거나 MAF 환경과 호환성을 확인합니다.
4. 합성 예제를 쓸 경우 `data/synthetic.csv`도 함께 배치하거나 `data_path="/path/synthetic.csv"`를 지정합니다.

```python
from algorithms.preprocessing.MinorityClassWeighting import ALINJobMinorityWeighting

algorithm = ALINJobMinorityWeighting(
    dataset_name="synthetic", protected="sex", floor=2000, seed=1,
)
metrics_orig, metrics_transf = algorithm.run()
```

### 입력과 반환값

사용자 데이터는 `ALINJobMinorityWeighting(dataset=aif_dataset, protected="sex")`로 전달합니다.

- `BinaryLabelDataset`의 두 레이블과 favorable/unfavorable 메타데이터가 일치해야 합니다.
- 선택한 보호속성은 `0=unprivileged`, `1=privileged`로 인코딩합니다.
- 특징은 숫자형이며 결측·무한대가 없어야 합니다. 전처리 상태는 학습 데이터에서 적합합니다.
- 표본 가중치는 유한한 비음수 1차원 배열이며, 각 분할의 각 클래스에 양의 가중치 합이 필요합니다.
- 모든 보호속성 이름에 해당하는 열을 모델 입력에서 제외합니다. 다른 이름으로 복제된 대리변수까지 자동 탐지하는 것은 아닙니다.

공통 AIF360 분류·공정성 지표와 다음 정보를 반환합니다.

| 키 | 의미 |
|---|---|
| `equal_opportunity_difference` | AIF360의 signed TPR 차이: unprivileged − privileged |
| `absolute_deo` | 위 TPR 차이의 절댓값 (0–1) |
| `tpr_unprivileged`, `tpr_privileged` | 집단별 양성 recall |
| `group_support_*`, `positive_support_*` | 집단별 원시 행 수 및 양성 행 수 |
| `positive_weight_*` | 집단별 실제 양성의 평가 가중치 합 |
| `deo_measurable` | 두 집단의 양성 분모가 존재하는지 여부 |

정의되지 않는 지표는 `None`/JSON `null`입니다. `deo_measurable`은 통계적 신뢰도 보장이 아닙니다. `run.py`의 `accuracy_gap_pp`는 `100 * abs(acc_weighted − acc_baseline)`이며 단위는 퍼센트포인트입니다.

ZIP에는 실행 코드·합성 예제·집계 결과를 포함하고, 파일별 SHA256 manifest를 생성합니다. 원본 데이터·개인정보·모델 파일·가상환경은 포함하지 않습니다. 로컬 AIF360 실행과 가이드 인터페이스를 검증하며, 실제 MAF2025/MAF-DEMO 서버 등록·화면 연결은 코난테크놀로지의 통합 과정에서 수행해야 합니다.

## 결과

**기존 SCOUT 보고값**과 **합성 데이터 실행 결과**를 구분합니다. 서로 다른 데이터·목표·모델의 수치를 같은 벤치마크 성능으로 비교하지 않습니다.

### 1. 기존 SCOUT 보고값

비공개 SCOUT 학력 5분류 / LightGBM / 5-fold OOF 결과입니다.

| 단계 / 방법 | 정확도 | absolute DEO | 비가중 모델과 정확도 차이 |
|---|---:|---:|---:|
| 이전 발표 초안 | 85.30% | 0.0408 | 0.79%p |
| 최신 ALINJob 보고 운영점 (`floor=250`, `num_leaves=63`) | 85.98% | 0.0136 | 0.06%p |

최신 운영점은 LightGBM `num_leaves=63`, 소수 클래스 가중 `floor=250`이며, 학력 5분류와 pooled 5-fold OOF(seed=42)를 사용한 것으로 보고돼 있습니다. 전체 데이터는 약 9,991행, 다수 클래스 기저확률은 85.78%, 고졸 recall은 약 0.129로 보고돼 있습니다.

공정성 양성 클래스는 고졸이고, 보호집단은 학교명에 '여자'가 포함되는지의 대리변수입니다. 보호집단의 실제 고졸이 7명이므로 집단 TPR은 1명 변화에 약 0.143씩 바뀝니다. OOF 통합도 이 표본 크기 한계를 없애지는 않습니다. 장애인 집단에 대한 공정성이 직접 검증된 결과가 아닙니다.

지표 정의는 [지표 정의](#지표-정의)를 참고하십시오.

#### 기존 문서에 보고된 동일 태스크 비교

| 방법 | 정확도 | absolute DEO |
|---|---:|---:|
| 가중치 없는 튜닝 백본 | 86.04% | 별도 기재 없음 |
| TabPFN v2 | 85.96% | 0.0612 |
| 기본 LightGBM | 85.57% | 0.0884 |
| FairGBM (고졸 head에만 등기회 제약) | 약 85.1% | 약 0.02–0.04 |

FairGBM은 실행 간 변동이 있어 근사 범위로 기술합니다. 위 비교는 기존 자체 SCOUT 실험의 보고값이지 공개 리더보드나 원 논문의 수치가 아닙니다. 정확도의 작은 차이만으로 통계적 우월성을 주장하지 않습니다.

#### 보고값 출처

원 프로젝트 `recruitment/recruitment_2/`의 다음 로컬 문서를 대조해 옮겼습니다.

- `작성내용_진도점검발표자료.md`, `발표슬라이드_채용2_2장.md`: 이전 발표 초안.
- `SOTA_REFERENCES.md`, `작성내용_대표성과_서술형_채용2.md`: 최신 보고 운영점과 비교 방법.
- 프로젝트 출처: https://github.com/jiwoohong93/ai_dep

이 문서들이 원격에 모두 게시돼 있다는 뜻은 아닙니다. 보고된 수치는 `results/scout_reported.json`에 기계 판독 형식으로도 정리했습니다. 원 보고자의 기록을 전사한 것이며, 저장소 준비 과정에서 SCOUT 전체 실험을 재실행하지 않았습니다. 보존된 `experiments/scout/` 코드와 [SCOUT 데이터 조건](#scout-비공개)을 통해 데이터 이용 권한이 있는 환경에서 재현할 수 있습니다.

### 2. 합성 데이터 MAF 실행 결과

`results/maf_benchmarks.json`은 합성 데이터에서 실제 실행한 결과와 환경을 담습니다. 하나의 고정 분할(seed=1, test_size=0.3)을 사용하며, 가중 전후 모델은 Logistic Regression이고 `floor=2000`입니다. 합성 데이터는 시연용으로 생성 파라미터를 선택했고 그 과정에서 테스트 지표를 참고했습니다([합성 데이터 생성 방법](#합성-데이터-생성-방법)). 이 값은 SCOUT 보고값을 재현한 것이 아닙니다.

| 데이터 | 기본 정확도 | 가중 정확도 | 기본 DEO | 가중 DEO |
|---|---:|---:|---:|---:|
| synthetic | 92.25% | 91.78% | 0.03290 | 0.02073 |

위 표는 JSON 값을 반올림한 표시입니다. 적용 전후 평가는 동일한 평가 가중치를 사용합니다. `floor` 이하인 클래스가 없는 데이터에서는 가중치가 모두 1이므로 적용 전후가 동일할 수 있습니다. 클래스 인식 개선과 집단 간 TPR 격차 개선은 같은 목표가 아닙니다. 단일 seed 실행은 평균 성능·유의성·공정성 보장의 근거가 아닙니다.

## 프로젝트 구조

```text
algorithms/preprocessing/MinorityClassWeighting.py  # 가중치 함수와 MAF 연결 클래스
run.py                                            # 합성 데이터 실행
validate.py                                       # 회귀 테스트
build_submission.py                               # MAF 제출 ZIP 생성
requirements.txt                                  # MAF 실행 의존성
requirements-scout.txt                            # SCOUT 재현 추가 의존성
data/synthetic.csv                                # 실제 개인정보 없는 합성 데이터
data/generate_synthetic.py                        # 합성 데이터 생성 스크립트
results/                                          # 집계 결과와 설정
experiments/scout/                                # 기존 LightGBM 실험 코드 사본
```

## 라이선스 및 관련 프로젝트

- MAF: https://github.com/konanaif/MAF2025
- MAF-DEMO: https://github.com/konanaif/MAF2025-DEMO
- 원 ALINJob 프로젝트: https://github.com/jiwoohong93/ai_dep
- AIF360: https://github.com/Trusted-AI/AIF360
- LightGBM: https://github.com/microsoft/LightGBM

원 프로젝트의 MIT 라이선스 고지를 [LICENSE](LICENSE)에 보존했습니다. SCOUT 이력 데이터와 학습 모델은 포함하지 않습니다.
