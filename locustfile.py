import random
import numpy as np
from locust import HttpUser, task, between

PAYLOAD_POOL=[
    {
        "tensor": np.random.randn(3,32,32).astype(np.float32).tolist()
    }
        for _ in range(10)

]

class CIFAR10InferenceUser(HttpUser):
    wait_time=between(0.01,0.05)
    
    @task(10)
    def predict_inference(self):
        payload=random.choice(PAYLOAD_POOL)
        headers={"Content-Type": "application/json"}
        
        with self.client.post(
            "/predict",
            json=payload,
            headers=headers,
            catch_response=True,
            name="/predict"
        ) as response:
            if response.status_code==200:
                data=response.json()
                if "class_id" in data and "confidence" in data:
                    response.success()
                else:
                    response.failure(f"Malformed response payload: {data}")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")
                
@task(1)
def check_health(self):
    self.client.get("/healthz", name="/healthz")