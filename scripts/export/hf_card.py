"""Write the Hugging Face dataset card (README.md) for an export dir: YAML configs for every data/<config>/<split>-*
present, then docs/DATA.md."""
from __future__ import annotations

import sys
from pathlib import Path

root = Path(sys.argv[1]); repo = Path(__file__).resolve().parents[2]
cfgs = []
for c in sorted(p for p in (root / "data").iterdir() if p.is_dir()):
    splits = sorted({f.name.split("-")[0] for f in c.glob("*.parquet")})
    if splits:
        lines = [f"- config_name: {c.name}", "  data_files:"]
        for s in splits:
            lines += [f"  - split: {s}", f"    path: data/{c.name}/{s}-*.parquet"]
        cfgs.append("\n".join(lines))
header = "---\nlicense: other\nlicense_name: mixed-see-card\npretty_name: Llama-3.1-8B neuron attributions (Stat 236)\n" \
         "configs:\n" + "\n".join(cfgs) + "\n---\n\n"
(root / "README.md").write_text(header + (repo / "docs" / "DATA.md").read_text()
                                .replace("<org>/<repo>", "stat236/llama31-8b-neuron-attributions"))
print("card configs:", [c.splitlines()[0] for c in cfgs])
