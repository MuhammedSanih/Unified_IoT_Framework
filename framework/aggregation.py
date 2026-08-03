from datetime import datetime

def aggregate_prediction(
    input_type,
    selected_model,
    prediction,
    confidence,
    inference_time_ms
):

    if confidence >= 95:
        confidence_level = "High"

    elif confidence >= 80:
        confidence_level = "Medium"

    else:
        confidence_level = "Low"

    if input_type == "network":

        if prediction == "Attack":
            recommended_action = "Raise Alert"
            severity = "Critical"

        else:
            recommended_action = "Allow Traffic"
            severity = "Normal"

    else:

        recommended_action = "Object Classified"
        severity = "N/A"

    return {

        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "input_type": input_type,

        "selected_model": selected_model,

        "prediction": prediction,

        "confidence": round(confidence * 100,2),

        "confidence_level": confidence_level,

        "recommended_action": recommended_action,

        "severity": severity,

        "inference_time_ms": round(inference_time_ms,2)
    }
