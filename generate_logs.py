import json
import random
import uuid
from datetime import datetime, timedelta

# Configurations
NUM_APP_LOGS = 1000
NUM_TELEMETRY_LOGS = 800
NUM_CICD_LOGS = 50

SERVICE_NAME = "checkout-service"
LEVELS =["INFO", "INFO", "INFO", "INFO", "WARN", "ERROR", "DEBUG"]

# Checkout specific functions/methods
METHODS =[
    "validateCart", 
    "calculateTaxes", 
    "applyDiscount", 
    "processPayment", 
    "updateInventory", 
    "confirmOrder"
]

# Expanded telemetry metrics
METRICS =[
    "cpu_utilization_percent", 
    "memory_usage_mb", 
    "payment_gateway_latency_ms", 
    "db_query_latency_ms",
    "active_checkout_sessions",
    "checkout_error_rate_percent"
]

STAGES =["build", "test", "security_scan", "deploy_staging", "deploy_production"]
STATUSES =["SUCCESS", "SUCCESS", "SUCCESS", "FAILED", "IN_PROGRESS"]

# Start generating logs from 24 hours ago
start_time = datetime.utcnow() - timedelta(days=1)

def random_time(start, end=datetime.utcnow()):
    delta = end - start
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=random_seconds)

def get_message_for_method(method, level):
    if level in ["INFO", "DEBUG"]:
        messages = {
            "validateCart": "Cart validated successfully.",
            "calculateTaxes": "Taxes calculated based on user region.",
            "applyDiscount": "Promo code applied to cart total.",
            "processPayment": "Payment transaction initiated.",
            "updateInventory": "Items reserved in inventory.",
            "confirmOrder": "Order confirmation email queued."
        }
        return messages.get(method, "Operation completed.")
    elif level == "WARN":
        return f"Retrying {method} due to slow response."
    else: # ERROR
        errors = {
            "validateCart": "InvalidItemException: Item no longer available.",
            "calculateTaxes": "TaxProviderUnavailableException: Unable to reach tax API.",
            "applyDiscount": "ExpiredPromoCodeException: Code is no longer valid.",
            "processPayment": "PaymentGatewayTimeoutException: Upstream provider timeout.",
            "updateInventory": "InventorySyncError: Failed to lock database row.",
            "confirmOrder": "NotificationDeliveryFailed: Email provider rejected request."
        }
        return errors.get(method, "Unknown internal server error.")

def generate_app_logs(count):
    logs =[]
    for _ in range(count):
        log_time = random_time(start_time)
        lvl = random.choice(LEVELS)
        method = random.choice(METHODS)
        msg = get_message_for_method(method, lvl)

        logs.append({
            "timestamp": log_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "log_type": "application",
            "trace_id": f"chk-{uuid.uuid4().hex[:8]}",
            "service": SERVICE_NAME,
            "method": method,
            "level": lvl,
            "message": msg
        })
    # Sort chronologically
    return sorted(logs, key=lambda x: x["timestamp"])

def generate_telemetry_logs(count):
    logs =[]
    for _ in range(count):
        log_time = random_time(start_time)
        metric = random.choice(METRICS)
        
        # Assign realistic values based on metric type
        if metric == "cpu_utilization_percent": val = round(random.uniform(10.0, 95.0), 2)
        elif metric == "memory_usage_mb": val = round(random.uniform(512.0, 4096.0), 2)
        elif metric == "payment_gateway_latency_ms": val = round(random.uniform(150.0, 3500.0), 2)
        elif metric == "db_query_latency_ms": val = round(random.uniform(5.0, 300.0), 2)
        elif metric == "active_checkout_sessions": val = random.randint(50, 2000)
        else: val = round(random.uniform(0.0, 5.0), 2) # error rate percent

        logs.append({
            "timestamp": log_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "log_type": "telemetry",
            "service": SERVICE_NAME,
            "metric": metric,
            "value": val
        })
    return sorted(logs, key=lambda x: x["timestamp"])

def generate_cicd_logs(count):
    logs =[]
    for _ in range(count):
        log_time = random_time(start_time)
        logs.append({
            "timestamp": log_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "log_type": "cicd",
            "pipeline_id": f"pipe_chk_{random.randint(100, 999)}",
            "service": SERVICE_NAME,
            "commit_hash": uuid.uuid4().hex[:7],
            "stage": random.choice(STAGES),
            "status": random.choice(STATUSES),
            "duration_sec": random.randint(30, 600)
        })
    return sorted(logs, key=lambda x: x["timestamp"])

if __name__ == "__main__":
    app_logs = generate_app_logs(NUM_APP_LOGS)
    telemetry_logs = generate_telemetry_logs(NUM_TELEMETRY_LOGS)
    cicd_logs = generate_cicd_logs(NUM_CICD_LOGS)

    with open("application_logs.json", "w") as f:
        json.dump(app_logs, f, indent=2)
    
    with open("telemetry_logs.json", "w") as f:
        json.dump(telemetry_logs, f, indent=2)

    with open("cicd_logs.json", "w") as f:
        json.dump(cicd_logs, f, indent=2)

    print(f"Successfully generated 3 files for the Checkout Service:")
    print(f"- application_logs.json ({NUM_APP_LOGS} logs)")
    print(f"- telemetry_logs.json ({NUM_TELEMETRY_LOGS} logs)")
    print(f"- cicd_logs.json ({NUM_CICD_LOGS} logs)")