# rorry

VSCode에서 `.ipynb` 파일에 특정 **virtual environment(가상 환경)**를 설정하는 방법은 다음과 같습니다. 이 설정은 주로 **Jupyter Notebook** 환경에서 특정 가상 환경을 사용할 때 필요합니다. 여기서는 두 가지 방법을 소개하겠습니다.

### 1. 가상 환경에 Jupyter 설치 및 커널 등록

먼저, 특정 가상 환경에서 **Jupyter**와 **IPython kernel**을 설치하고, 해당 가상 환경을 Jupyter 커널로 등록해야 합니다.

#### 가상 환경에서 Jupyter 설치:
0. 가상 환경 설치
   - `python -m venv myenv`

2. 가상 환경을 활성화합니다:
   - Linux/Mac:
     ```bash
     source myenv/bin/activate
     ```
   - Windows:
     ```bash
     myenv\Scripts\activate
     ```

3. 가상 환경에서 Jupyter와 IPython kernel 설치:
   ```bash
   pip install jupyter ipykernel
   ```

4. 가상 환경을 Jupyter 커널로 등록:
   ```bash
   python -m ipykernel install --user --name myenv --display-name "Python (myenv)"
   ```
   - `myenv`는 커널의 이름으로 자유롭게 지정할 수 있으며, Jupyter Notebook에서 이 이름이 나타납니다.
   - `"Python (myenv)"`는 VSCode에서 해당 가상 환경을 선택할 때 표시될 이름입니다.

### 2. VSCode에서 Jupyter 커널 선택

1. VSCode에서 `.ipynb` 파일을 엽니다.

2. 파일 상단에 **Kernel**을 선택할 수 있는 드롭다운 메뉴가 있습니다. 이 드롭다운을 클릭하여 방금 등록한 가상 환경 커널을 선택하세요.
   - 가상 환경을 Jupyter 커널로 등록한 후, 해당 환경이 목록에 나타나게 됩니다.
   - 선택한 후부터는 해당 가상 환경에서 `.ipynb` 파일을 실행할 수 있습니다.

### 3. VSCode에서 가상 환경 자동 활성화 확인

VSCode에서 터미널을 열고 가상 환경이 자동으로 활성화되지 않으면 `"python.terminal.activateEnvironment": true` 설정을 확인하여 환경이 제대로 활성화되는지 보세요.

### 요약
1. **가상 환경 활성화** 후, `jupyter`와 `ipykernel`을 설치.
2. `python -m ipykernel install` 명령어로 가상 환경을 Jupyter 커널에 등록.
3. VSCode에서 `.ipynb` 파일을 열고, 상단의 **Kernel** 메뉴에서 해당 가상 환경을 선택.

이렇게 하면 `.ipynb` 파일이 해당 가상 환경을 사용하여 실행됩니다.
