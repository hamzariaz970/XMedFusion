import json

with open("data/iu_xray/annotation.json") as f:
    data = json.load(f)

all_s = data.get("train", []) + data.get("val", []) + data.get("test", [])

targets = ["CXR3233_IM-1530", "CXR1264_IM-0179", "CXR1161_IM-0107", 
           "CXR3638_IM-1804", "CXR152_IM-0335", "CXR811_IM-2343",
           "CXR1743_IM-0489", "CXR3915_IM-1990", "CXR934_IM-2432",
           "CXR651_IM-2229", "CXR1419_IM-0267"]

for s in all_s:
    if s.get("id") in targets:
        print(f"\n{'='*60}")
        print(f"ID: {s['id']}")
        print(f"Report: {s['report']}")
