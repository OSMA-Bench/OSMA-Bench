"""VLM 호출과 응답 파싱.

VLM 이 만들어 내는 것은 **라벨 번호뿐**이다. 좌표도, 코드도 아니다. 그래서 파싱이
정규식 한 줄로 끝나고, 범위를 벗어난 번호는 즉시 걸러진다. VoxPoser 쪽에서 4B 가
`avoidance_map` 키워드를 중복 생성해 SyntaxError 로 재시도를 소진하던 실패 유형이
이 방식에는 없다.

이미지는 질의마다 한 장씩 보낸다. 서버가 `--limit-mm-per-prompt image=4` 로 뜨는데
계층 모드는 3~5회 질의하므로, 한 요청에 여러 장을 싣지 않는 것이 상한을 피하는
가장 단순한 방법이다.
"""
import base64
import json
import re
import urllib.error
import urllib.request

from annotate import to_png_bytes


class VLMError(Exception):
    """호출 자체가 실패했다 (네트워크·서버). 형식 오류와 구분한다."""


class ParseError(Exception):
    """응답은 왔는데 라벨을 못 읽었다. 재시도 사다리를 타는 신호."""


_PATH_RE = re.compile(r"PATH\s*:\s*([0-9,\s]+)", re.IGNORECASE)
# NEXT: 3 / Circle: 3 / 그냥 "3". 재질의 문구가 프롬프트와 다른 낱말을 쓰면
# (프롬프트는 NEXT, 재질의는 Circle) 모델이 숫자만 답한다 - 실측: 응답 '2' 가
# "NEXT 줄 없음" 으로 3 회 연속 실패해 에피소드가 통째로 버려졌다.
_NEXT_RE = re.compile(r"(?:NEXT|Circle)\s*[:：]\s*(\d+)|^\s*(\d+)\s*$", re.I | re.M)
_KV_RE = {
    "speed": re.compile(r"SPEED\s*:\s*(normal|slow|very_slow)", re.IGNORECASE),
    "clearance": re.compile(r"CLEARANCE\s*:\s*(default|wide|very_wide)", re.IGNORECASE),
    "side": re.compile(r"SIDE\s*:\s*(left|right|either)", re.IGNORECASE),
}


class VLMClient:
    """OpenAI 호환 chat/completions 엔드포인트에 이미지 한 장 + 텍스트를 보낸다."""

    def __init__(self, model, base_url="http://localhost:8000/v1",
                 temperature=0.0, max_tokens=512, timeout=180, seed=None):
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        # temperature=0 만으로는 재현되지 않는다. vLLM 은 연속 배칭이라 같은 요청도
        # 함께 묶인 요청에 따라 수치가 갈리고, 실측으로 같은 에피소드가 위반
        # 98.2 / 80.0 / 10.9 % 로 나왔다. seed 를 실어야 그 갈래가 닫힌다.
        # None 이면 필드를 아예 넣지 않는다 - seed 를 모르는 엔드포인트도 있다.
        self.seed = seed

    def ask(self, prompt, image=None):
        """프롬프트(+이미지)를 보내고 응답 텍스트를 돌려준다."""
        content = [{"type": "text", "text": prompt}]
        if image is not None:
            b64 = base64.b64encode(to_png_bytes(image)).decode()
            content.insert(0, {"type": "image_url",
                               "image_url": {"url": f"data:image/png;base64,{b64}"}})
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **({} if self.seed is None else {"seed": int(self.seed)}),
        }).encode()
        req = urllib.request.Request(self.url, data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                doc = json.loads(r.read())
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise VLMError(str(e)) from e
        try:
            return doc["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise VLMError(f"unexpected response shape: {doc}") from e


_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)


def strip_thinking(text):
    """thinking 모델의 사고 블록을 걷어낸다.

    <think> 안에는 후보 번호가 여러 번 등장한다 - "3번은 너무 가깝고 5번이 낫다"
    같은 문장이다. 그 첫 숫자를 답으로 읽으면 모델이 기각한 선택을 고르게 된다.
    닫히지 않은 채 잘린 경우(토큰 상한)도 있어 그때는 마지막 블록 이후만 남긴다.
    """
    if not text:
        return text
    out = _THINK_RE.sub(" ", text)
    if "<think>" in out.lower():                      # 닫히지 않은 블록
        out = out[:out.lower().rindex("<think>")]
    return out


def parse_path(text, n_labels):
    """'PATH: 4, 11, 19' -> [4, 11, 19] (1-기반 라벨).

    범위 밖 번호는 버리고, 중복은 순서를 유지한 채 제거한다. 하나도 남지 않으면
    ParseError — 재시도 사다리가 후보를 줄여 다시 묻는다.
    """
    text = strip_thinking(text)
    m = _PATH_RE.search(text or "")
    if not m:
        raise ParseError("PATH 줄 없음")
    out, seen = [], set()
    for tok in m.group(1).split(","):
        tok = tok.strip()
        if not tok.isdigit():
            continue
        v = int(tok)
        if 1 <= v <= n_labels and v not in seen:
            seen.add(v)
            out.append(v)
    if not out:
        raise ParseError(f"유효 라벨 없음 (1~{n_labels} 범위 밖)")
    return out


def parse_next(text, n_labels):
    """'NEXT: 7' -> 7. iterative 모드용."""
    text = strip_thinking(text)
    m = _NEXT_RE.search(text or "")
    if not m:
        raise ParseError("NEXT 줄 없음")
    v = int(m.group(1) or m.group(2))
    if not (1 <= v <= n_labels):
        raise ParseError(f"라벨 {v} 가 1~{n_labels} 범위 밖")
    return v


def parse_constraints(text):
    """제약 3종을 읽는다. 빠진 항목은 기본값으로 채운다 (Phase 2).

    형식이 조금 틀려도 셋 중 읽힌 것은 살린다 — 제약은 보조 정보라 하나 놓쳤다고
    에피소드를 버릴 이유가 없다.
    """
    out = {"speed": "normal", "clearance": "default", "side": "either"}
    for k, rx in _KV_RE.items():
        m = rx.search(text or "")
        if m:
            out[k] = m.group(1).lower()
    return out
