import sys
import asyncio
sys.path.append("/home/rapids/Desktop/XMedFusion/XMedFusion/backend")
from synthesis import _infer_ct_report

async def test():
    try:
        report = _infer_ct_report(["/home/rapids/Desktop/XMedFusion/XMedFusion/backend/data/ctscan/8.png"])
        print("SUCCESS! Generated CT Report:")
        print(report)
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    asyncio.run(test())
