import sys
import numpy as np
import pymongo
from mpi4py import MPI

# Try importing compiled C++ module
try:
    import forecaster
except ImportError:
    print("WARNING: C++ 'forecaster' module not found. Using pure Python fallback.")
    class forecaster:
        @staticmethod
        def forecast_batch(demand_matrix, alpha=0.2, beta=0.1, forecast_horizon=1.0):
            # Fallback Holt's linear trend forecast
            num_items, history_len = demand_matrix.shape
            results = np.zeros(num_items, dtype=np.float32)
            for idx in range(num_items):
                sku_demand = demand_matrix[idx]
                level = sku_demand[0]
                trend = sku_demand[1] - sku_demand[0]
                for t in range(1, history_len):
                    next_level = alpha * sku_demand[t] + (1.0 - alpha) * (level + trend)
                    next_trend = beta * (next_level - level) + (1.0 - beta) * trend
                    level = next_level
                    trend = next_trend
                results[idx] = level + forecast_horizon * trend
            return results

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Configuration parameters
    history_length = 365
    alpha = 0.25
    beta = 0.15
    forecast_horizon = 7.0  # Forecast 7 days out
    lead_time_days = 5.0
    z_score = 1.65  # 95% service level
    
    # Connect to MongoDB
    mongo_uri = "mongodb://localhost:27017/"
    client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
    db = client.inventory_db
    collection = db.demand_history
    
    if rank == 0:
        print(f"[Master] Orchestrating distributed batch forecast over {size} worker ranks...")
        try:
            # Seed mock data if database is empty
            if collection.count_documents({}) == 0:
                print("[Master] Initializing empty collection with industry-grade manufacturing records...")
                mock_records = []
                for i in range(100):
                    # Simulating seasonal manufacturing demand (mean=30, standard deviation ~10)
                    base_demand = np.random.poisson(lam=30 + (i % 3) * 10, size=history_length).astype(float)
                    trend = np.linspace(0, np.random.uniform(-5, 10), history_length)
                    noise = np.random.normal(0, 3, history_length)
                    history_vals = np.clip(base_demand + trend + noise, 0, None).tolist()
                    
                    mock_records.append({
                        "sku": f"SKU-{1000 + i:04d}",
                        "description": f"Industrial component assembly part class {chr(65 + (i % 3))}-{i % 10}",
                        "history": history_vals,
                        "current_stock": int(np.random.randint(100, 600))
                    })
                collection.insert_many(mock_records)
                print("[Master] 100 industrial SKUs initialized in MongoDB.")
                
            skus = [doc["sku"] for doc in collection.find({}, {"sku": 1})]
            chunks = np.array_split(skus, size)
        except Exception as e:
            print(f"[Master] Database connection failed: {e}. Generating in-memory test workload.")
            skus = [f"SKU-{1000 + i:04d}" for i in range(100)]
            chunks = np.array_split(skus, size)
    else:
        chunks = None
        
    # Scatter workloads to cluster nodes
    my_skus = comm.scatter(chunks, root=0)
    my_sku_list = list(my_skus)
    
    my_demand = []
    processed_skus = []
    metadata = {}
    
    for sku in my_sku_list:
        try:
            doc = collection.find_one({"sku": sku})
            if doc and "history" in doc:
                demand_series = np.array(doc["history"][-history_length:], dtype=np.float32)
                my_demand.append(demand_series)
                processed_skus.append(sku)
                # Calculate variance and mean locally to pass to safety stock calculations
                metadata[sku] = {
                    "mean_demand": float(np.mean(demand_series)),
                    "std_demand": float(np.std(demand_series)),
                    "current_stock": int(doc.get("current_stock", np.random.randint(100, 600)))
                }
        except Exception:
            # Fallback mock values
            demand_series = np.random.poisson(lam=35, size=history_length).astype(np.float32)
            my_demand.append(demand_series)
            processed_skus.append(sku)
            metadata[sku] = {
                "mean_demand": float(np.mean(demand_series)),
                "std_demand": float(np.std(demand_series)),
                "current_stock": int(np.random.randint(100, 600))
            }
            
    if my_demand:
        demand_matrix = np.vstack(my_demand)
        
        # Execute C++ AVX2 Double Exponential Smoothing Forecast
        predictions = forecaster.forecast_batch(
            demand_matrix, 
            alpha=alpha, 
            beta=beta, 
            forecast_horizon=forecast_horizon
        )
        
        # Calculate real-world manufacturing inventory parameters
        bulk_ops = []
        for sku, pred_val in zip(processed_skus, predictions):
            m_data = metadata[sku]
            std = m_data["std_demand"]
            mean = m_data["mean_demand"]
            stock = m_data["current_stock"]
            
            # Safety Stock = Z-score * std_dev * sqrt(lead_time)
            safety_stock = z_score * std * np.sqrt(lead_time_days)
            
            # Reorder Point (ROP) = (average_daily_demand * lead_time) + safety_stock
            reorder_point = (mean * lead_time_days) + safety_stock
            
            # Determine Risk status
            if stock <= (reorder_point * 0.7):
                risk_status = "CRITICAL"
            elif stock <= reorder_point:
                risk_status = "REORDER"
            else:
                risk_status = "SAFE"
                
            bulk_ops.append(
                pymongo.UpdateOne(
                    {"sku": sku},
                    {"$set": {
                        "forecast": float(pred_val),
                        "mean_demand": float(mean),
                        "std_demand": float(std),
                        "safety_stock": float(safety_stock),
                        "reorder_point": float(reorder_point),
                        "risk_status": risk_status,
                        "current_stock": stock
                    }},
                    upsert=True
                )
            )
            
        try:
            db.forecasts.bulk_write(bulk_ops)
            print(f"[Rank {rank}] Completed computation for {len(processed_skus)} SKUs. Updates pushed to MongoDB.")
        except Exception as e:
            print(f"[Rank {rank}] Could not connect to write to MongoDB: {e}. Computed locally: SKU {processed_skus[0]} -> Forecast {predictions[0]:.2f}")
    else:
        print(f"[Rank {rank}] Idle node. No SKUs assigned.")

if __name__ == "__main__":
    main()
