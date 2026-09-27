import json,sys
try: d=json.load(sys.stdin)
except Exception: print("  解析败"); sys.exit()
if "data" not in d: print("  ⚠",str(d)[:80]); sys.exit()
for p in d["data"][:5]:
    doi=(p.get("externalIds") or {}).get("DOI","")
    print("  %s 引%5d %s"%(p.get("year","—"),p.get("citationCount",0),p.get("title","")[:86]))
    if doi: print("        DOI:",doi)
