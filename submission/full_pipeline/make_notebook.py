"""별도 패키지 없이 설명과 실행 코드를 담은 파일럿 노트북을 만든다.

파이프라인 단계: 보조 도구
입력: 파일 안에 작성된 설명과 코드 셀을 사용한다.
출력: 실행 출력이 없는 LH2_pilot.ipynb를 현재 폴더에 저장한다.
최종 결과에는 쓰이지 않는 보조 도구이다."""
import json
from pathlib import Path

cells = []
def md(text):
    cells.append(dict(cell_type='markdown', metadata={}, source=text.splitlines(True)))
def code(text):
    cells.append(dict(cell_type='code', metadata={}, source=text.splitlines(True), execution_count=None, outputs=[]))

md('# LH2 흡입배관 대리모델: 200개 파일럿\n동일 조건 모델 비교, 데이터 누수 방지, 물리 오차 분리와 최종 테스트 평가. 실험 설계와 한계는 README.md에 정리했습니다.')
code('''from pathlib import Path
import sys
ROOT = Path.cwd()
if (ROOT / '.local_deps').exists():
    sys.path.insert(0, str(ROOT / '.local_deps'))
import lh2_pipeline as study
import pandas as pd
DATA = ROOT / 'data/dataset_ai.csv'
OUT = ROOT / 'outputs/pilot'
''')
md('## 1. 데이터 감사\n결측·중복·로그 변환·단위·압력 구성식을 검사합니다. 입력 전용 1만 개 파일은 합치지 않습니다.')
code('df = study.load_data(DATA)\ndisplay(df.describe().T)')
md('## 2. 개발 데이터에서 모델 선택\n160개 개발 자료에서 동일 5-fold × 3회 CV를 사용합니다. 튜닝 모델당 6개 후보, seed 2026. 기존 실험은 재학습하지 않습니다.')
code('''if not (OUT / 'manifest.json').exists():
    study.development(DATA, OUT, learning=True)
display(pd.read_csv(OUT / 'cv_summary.csv'))''')
md('## 3. 학습 곡선과 수렴 확인\n테스트셋을 사용하지 않습니다. CV는 모델 선택용이며 독립적인 최종 성능 추정치가 아닙니다. 수렴 경고를 manifest에서 확인합니다.')
code('''import json
from IPython.display import Image, display
manifest = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
print('Selected:', manifest['selected'])
print('Recorded warnings:', len(manifest['warnings']))
for target in study.TARGETS:
    for suffix in ['__learning.png', '__mlp_history.png']:
        image = OUT / (target + suffix)
        if image.exists():
            display(Image(filename=str(image)))''')
md('## 4. 최종 테스트: 모델 확정 후 한 번만\n선정 모델과 사전 지정 MLP를 평가합니다. 결과를 보고 재튜닝하지 않습니다. 완료한 실험은 저장된 표만 확인합니다.')
code('''if not (OUT / 'final_test_metrics.csv').exists():
    study.finalize(DATA, OUT)
display(pd.read_csv(OUT / 'final_test_metrics.csv'))
print((OUT / 'final_pressure_metrics.json').read_text(encoding='utf-8'))''')
md('## 5. 해석과 한계\n최종 테스트는 40개뿐입니다. 수력식 근사오차와 AI 오차를 분리해서 보고합니다. 실제 설계 확정에는 솔버 재검증이 필요합니다. 열침입 물리 기준선, 참고논문 재현 ANN과 최적화는 추가 자료 확보 후 진행합니다.')
for index, cell in enumerate(cells):
    cell['id'] = f'lh2-{index:02d}'
notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'), language_info=dict(name='python', version='3.12')), nbformat=4, nbformat_minor=5)
Path('LH2_pilot.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding='utf-8')
