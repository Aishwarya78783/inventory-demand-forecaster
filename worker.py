import sys
import numpy as np
import pymongo
from mpi4py import MPI

# Try importing the compiled C++ library
try:
    import forecaster
except ImportError:
    # Fallback to pure Python implementation for local testing if not compiled yet
    print("WARNING: C++ 'forecaster' module not found. Using fallback mockup.")
    class forecaster:
        @staticmethod
        def forecast_batch(demand_matrix, weights):
            # Simple weighted moving average fallback using numpy
            return np.dot(demand_matrix, weights)

def main():
    # Initialize MPI environment
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Define parameters
    history_length = 365
    weights = np.linspace(0.1, 1.0, history_length, dtype=np.float32)
    weights /= weights.sum()  # Normalize weights
    
    # Connect to MongoDB cluster
    # On exascale nodes, you'd configure a shared connection string or environment variable
    mongo_uri = "mongodb://localhost:27017/"
    client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
    db = client.inventory_db
    collection = db.demand_history
    
    # Fetch tasks and partition across MPI processes
    if rank == 0:
        print(f"[Master] Initializing distributed run with {size} processes...")
        try:
            # Generate mock database records if collection is empty
            if collection.count_documents({}) == 0:
                print("[Master] MongoDB database empty. Creating mockup demand data...")
                mock_data = []
                for i in range(100):
                    mock_data.append({
                        "sku": f"SKU-{1000 + i}",
                        "description": f"High performance storage component {i}",
                        "history": list(np.random.poisson(lam=15, size=history_length).astype(float))
                    })
                collection.insert_many(mock_data)
                print(f"[Master] Inserted 100 mockup items into database.")
            
            # Fetch all SKUs from DB
            skus = [doc["sku"] for doc in collection.find({}, {"sku": 1})]
            # Partition SKUs evenly across nodes
            chunks = np.array_split(skus, size)
        except Exception as e:
            print(f"[Master] Database connection failed: {e}. Running with mock SKU lists.")
            # Fallback mock SKU list for standalone/unconnected testing
            skus = [f"SKU-{1000 + i}" for i in range(100)]
            chunks = np.array_split(skus, size)
    else:
        chunks = None
        
    # Scatter assigned workloads to each rank
    my_skus = comm.scatter(chunks, root=0)
    my_sku_list = list(my_skus)
    
    # Process local SKUs
    my_demand = []
    processed_skus = []
    
    for sku in my_sku_list:
        try:
            doc = collection.find_one({"sku": sku})
            if doc and "history" in doc:
                demand = np.array(doc["history"][-history_length:], dtype=np.float32)
                my_demand.append(demand)
                processed_skus.append(sku)
        except Exception:
            # Local fallback mock data generation if DB not connected
            demand = np.random.poisson(lam=15, size=history_length).astype(np.float32)
            my_demand.append(demand)
            processed_skus.append(sku)
            
    if my_demand:
        demand_matrix = np.vstack(my_demand)
        
        # Execute vectorized forecasting calculation via pybind11 C++ module
        predictions = forecaster.forecast_batch(demand_matrix, weights)
        
        # Write results back to database
        bulk_operations = []
        for sku, forecast_val in zip(processed_skus, predictions):
            bulk_operations.append(
                pymongo.UpdateOne(
                    {"sku": sku},
                    {"$set": {"forecast": float(forecast_val)}},
                    upsert=True
                )
            )
        try:
            db.forecasts.bulk_write(bulk_operations)
            print(f"[Rank {rank}] Completed and saved {len(processed_skus)} forecasts to MongoDB.")
        except Exception as e:
            print(f"[Rank {rank}] Failed to save forecasts to DB: {e}. Executed locally: {list(zip(processed_skus, predictions[:5]))} ...")
    else:
        print(f"[Rank {rank}] No SKUs allocated.")

if __name__ == "__main__":
    main()
