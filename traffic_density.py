

def calculate_density(vehicle_count):
    if vehicle_count <= 10:
        return "LOW"
    elif vehicle_count <= 25:
        return "MEDIUM"
    else:
        return "HIGH"