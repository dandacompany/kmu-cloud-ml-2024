import json
import random
from datetime import datetime, timedelta
import uuid

def generate_sample_data(num_records=1000):
    customers = [f'C{i:04d}' for i in range(1, 101)]  # 100명의 고객
    products = [
        {"id": "P1", "name": "T-shirt", "price": 19.99},
        {"id": "P2", "name": "Jeans", "price": 49.99},
        {"id": "P3", "name": "Sneakers", "price": 79.99},
        {"id": "P4", "name": "Backpack", "price": 59.99},
        {"id": "P5", "name": "Hat", "price": 14.99},
        {"id": "P6", "name": "Dress", "price": 69.99},
        {"id": "P7", "name": "Shorts", "price": 29.99},
        {"id": "P8", "name": "Sweater", "price": 39.99},
        {"id": "P9", "name": "Jacket", "price": 89.99},
        {"id": "P10", "name": "Skirt", "price": 34.99},
    ]
    order_statuses = ["Pending", "Shipped", "Delivered"]

    records = []
    start_date = datetime(2023, 1, 1)

    for _ in range(num_records):
        customer_id = random.choice(customers)
        order_date = start_date + timedelta(days=random.randint(0, 365))
        order_items = random.sample(products, k=random.randint(1, 5))
        total_amount = sum(item["price"] * random.randint(1, 3) for item in order_items)
        
        record = {
            "CustomerId": customer_id,
            "OrderDate": order_date.strftime("%Y-%m-%d"),
            "OrderId": f"O-{uuid.uuid4()}",
            "OrderStatus": random.choice(order_statuses),
            "TotalAmount": round(total_amount, 2),
            "Items": [
                {
                    "id": item["id"],
                    "name": item["name"],
                    "quantity": random.randint(1, 3),
                    "price": item["price"]
                } for item in order_items
            ]
        }
        records.append(record)

    return records

def save_to_json(records, filename="sample_orders.json"):
    with open(filename, 'w') as f:
        json.dump(records, f, indent=2)

if __name__ == "__main__":
    sample_data = generate_sample_data()
    save_to_json(sample_data)
    print(f"Generated {len(sample_data)} sample records and saved to sample_orders.json")