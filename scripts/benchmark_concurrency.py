import asyncio
import time
import httpx
import numpy as np

TARGET_URL = "http://127.0.0.1:8000/predict"
CONCURRENCY = 50

def generate_payload():
    tensor=np.random.randn(3,32,32).astype(np.float32).tolist()
    return {"tensor": tensor}

async def send_single_request(client: httpx.AsyncClient, payload: dict):
    start=time.perf_counter()
    try:
        response=await client.post(TARGET_URL, json=payload, timeout=15.0)
        duration=(time.perf_counter()-start)*1000.0
        if response.status_code==200:
            data=response.json()
            return {
                "success": True,
                "latency_ms": duration,
                "batched_size": data.get("batched_size",1),
            }
        return {"success":False,"status":response.status_code, "latency_ms":duration}
    
    except Exception as exc:
        duartion=(time.perf_counter()-start)*1000.0
        return {"success": False, "error": str(exc), "latency_ms":duration}
    

async def main():
    print(f"Preparing {CONCURRENCY} concurrent requests.....")
    payloads=[generate_payload() for _ in range(CONCURRENCY)]
    
    limits=httpx.Limits(max_connections=CONCURRENCY, max_keepalive_connections=CONCURRENCY)
    async with httpx.AsyncClient(limits=limits) as client:
        start_total=time.perf_counter()
        results=await asyncio.gather(*(send_single_request(client,p) for p in payloads))
        total_time=time.perf_counter() - start_total
    
    successes=[r for r in results if r.get("success")]
    failures=[r for r in results if not r.get("success")]
    
    latencies=[r["latency_ms"] for r in successes]
    batch_sizes=[r["batched_size"] for r in successes]
    
    print("\n==========BENCHMARK REPORT===========")
    print(f"Total Requests Dispatched : {CONCURRENCY}")
    print(f"Successful (HTTP 200)     : {len(successes)}")
    print(f"Failed                    : {len(failures)}")
    print(f"Total Test Wall Clock     : {total_time:.3f} s")
    print(f"Throughput                : {len(successes)/total_time:.2f} req/s")
    
    if latencies:
        print("\n--- Latent Percentiles (End-to-End Client RoundTrip) ---")
        print(f"p50 (Median) : {np.percentile(latencies, 50):.2f} ms")
        print(f"p95          : {np.percentile(latencies, 95):.2f} ms")
        print(f"p99          : {np.percentile(latencies, 99):.2f} ms")
        print(f"Min / Max    : {np.min(latencies):.2f} ms / {np.max(latencies):.2f} ms")
        
        print("\n--- Dynamic Batching Efficiency ---")
        print(f"Average Batch Size Formed : {np.mean(batch_sizes):.1f}")
        print(f"Max Batch Size Seen       : {np.max(batch_sizes)}")
    print("==================================================")
    
if __name__ == "__main__":
    asyncio.run(main())