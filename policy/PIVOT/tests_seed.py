"""시드가 실제로 요청에 실리는지 확인한다.

설정에 값을 적어 두는 것과 그 값이 서버까지 가는 것은 다르다 - 이 저장소에서
"켰는데 아무 데도 안 닿았다" 가 이미 세 번 있었다(--dynamics 두 번, pivot 한 번).
"""
import json
import sys
import urllib.request

sys.path.insert(0, "lib")

_sent = {}


class _Resp:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()


def _fake_open(req, timeout=None):
    _sent.clear()
    _sent.update(json.loads(req.data))
    return _Resp()


urllib.request.urlopen = _fake_open

from vlm import VLMClient                                        # noqa: E402

VLMClient("m", seed=7).ask("hi")
assert _sent.get("seed") == 7, _sent
assert _sent.get("temperature") == 0.0, _sent
print(f"seed=7 전달됨: temperature={_sent['temperature']} seed={_sent['seed']}")

VLMClient("m").ask("hi")
assert "seed" not in _sent, "시드를 모르는 엔드포인트를 위해 필드를 빼야 한다"
print("seed 미지정: 필드 없음 (정상)")

# 훅이 config 에서 시드를 읽어 세 곳에 나눠 주는지.
import yaml                                                      # noqa: E402

cfg = yaml.safe_load(open("config.yaml"))
assert cfg["main"]["seed"] == 0
assert cfg["subgoal_solver"]["seed"] == 0
assert cfg["path_solver"]["seed"] == 0
print("config: main / subgoal_solver / path_solver 시드 일치")
print("통과")
