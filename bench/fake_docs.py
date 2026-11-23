"""Minimal in-memory fake of the Google Docs API (documents.create/get/batchUpdate)
for testing DocWriter's index arithmetic and formatting without network access.

Follows the real API's rules: body indices start at 1, are counted in UTF-16 code
units, the body always ends with a newline that can't be written after, and
out-of-range requests raise errors.
"""


class _Exec:
    def __init__(self, fn):
        self.fn = fn

    def execute(self):
        return self.fn()


class FakeDocs:
    def __init__(self):
        self.units: list[str] = ["\n"]   # one entry per UTF-16 code unit (Japanese BMP chars = 1)
        self.style: list[set] = [set()]
        self.requests = 0

    # API surface
    def documents(self):
        return self

    def create(self, body):
        return _Exec(lambda: {"documentId": "fake-doc"})

    def get(self, documentId, fields=None):
        return _Exec(lambda: {"body": {"content": [{"endIndex": len(self.units) + 1}]}})

    def batchUpdate(self, documentId, body):
        def run():
            self.requests += 1
            for r in body["requests"]:
                self._apply(r)
            return {}
        return _Exec(run)

    # helpers
    @staticmethod
    def _split(text):
        out = []
        for ch in text:
            out += [ch] + ([""] if ord(ch) > 0xFFFF else [])  # astral chars take 2 units
        return out

    def _apply(self, r):
        n = len(self.units)
        if "insertText" in r:
            i = r["insertText"]["location"]["index"]
            if not 1 <= i <= n:  # can insert before any unit incl. the final newline, never after it
                raise ValueError(f"insert index {i} out of range 1..{n}")
            new = self._split(r["insertText"]["text"])
            inherit = set(self.style[i - 2]) if i >= 2 else set()
            self.units[i - 1:i - 1] = new
            self.style[i - 1:i - 1] = [set(inherit) for _ in new]
        elif "updateTextStyle" in r or "updateParagraphStyle" in r:
            key = "updateTextStyle" if "updateTextStyle" in r else "updateParagraphStyle"
            a, b = r[key]["range"]["startIndex"], r[key]["range"]["endIndex"]
            if not (1 <= a < b <= n + 1):
                raise ValueError(f"{key} range {a}..{b} out of 1..{n + 1}")
            if key == "updateParagraphStyle":
                return
            ts = r[key]["textStyle"]
            fields = r[key]["fields"].split(",")
            for k in range(a - 1, b - 1):
                for f in fields:
                    self.style[k].discard(f)
                    if f in ts:
                        self.style[k].add(f)
        elif "replaceAllText" in r:
            old = r["replaceAllText"]["containsText"]["text"]
            new = r["replaceAllText"]["replaceText"]
            text = "".join(self.units)
            while old in text:
                i = text.index(old)
                k = len("".join(self.units)[:i])
                # map string index -> unit index (units hold "" for astral second halves)
                pos, acc = 0, 0
                while acc < i:
                    acc += len(self.units[pos]); pos += 1
                span, acc2 = pos, 0
                while acc2 < len(old):
                    acc2 += len(self.units[span]); span += 1
                st = self.style[pos]
                self.units[pos:span] = self._split(new)
                self.style[pos:span] = [set(st) for _ in self._split(new)]
                text = "".join(self.units)
        else:
            raise ValueError(f"unsupported request {list(r)}")

    def render(self) -> str:
        """Text with **bold** and ((grey)) markers."""
        out, cur = "", set()
        for u, s in zip(self.units, self.style):
            tag = ("b" if "bold" in s else "") + ("g" if "foregroundColor" in s else "")
            if tag != cur:
                out += {"b": "**", "g": "((", "bg": "**((", "": ""}.get(cur, "") and \
                       {"b": "**", "g": "))", "bg": "))**"}[cur] if cur else ""
                out += {"b": "**", "g": "((", "bg": "**((", "": ""}[tag]
                cur = tag
            out += u
        return out
