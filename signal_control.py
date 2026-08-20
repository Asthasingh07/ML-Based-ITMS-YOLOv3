def get_signal_time(density):

    if density == "LOW":
        return 30

    elif density == "MEDIUM":
        return 60

    elif density == "HIGH":
        return 90

    else:
        return 30


def get_signal_status(density):

    if density == "LOW":
        return "NORMAL TRAFFIC"

    elif density == "MEDIUM":
        return "MODERATE TRAFFIC"

    elif density == "HIGH":
        return "HEAVY TRAFFIC"

    else:
        return "NORMAL TRAFFIC"


def control_signal(density):

    signal_time = get_signal_time(density)

    print("Traffic Density:", density)
    print("Green Signal Time:", signal_time, "seconds")

    return signal_time