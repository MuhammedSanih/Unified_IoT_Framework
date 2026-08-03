import numpy as np

from framework.unified_framework import predict

# Load UNSW test data
X_test = np.load("data/X_test.npy")

# Mixed IoT inputs
samples = [
    {
        "name": "Cat Image",
        "data": "data/cifar10/test/cat/0002.png"
    },
    {
        "name": "UNSW Sample 1",
        "data": X_test[0]
    },
    {
        "name": "Ship Image",
        "data": "data/cifar10/test/ship/0005.png"
    },
    {
        "name": "UNSW Sample 2",
        "data": X_test[1]
    }
]

print("\n==============================================")
print("   Unified Heterogeneous IoT Framework")
print("==============================================")

for i, sample in enumerate(samples, start=1):

    print(f"\nProcessing Input {i}: {sample['name']}")

    try:

        result = predict(sample["data"])

        print("----------------------------------------------")
        print(f"Timestamp          : {result['timestamp']}")
        print(f"Input Type         : {result['input_type'].capitalize()}")
        print(f"Selected Model     : {result['selected_model']}")
        print(f"Prediction         : {result['prediction']}")
        print(f"Confidence         : {result['confidence']:.2f}%")
        print(f"Confidence Level   : {result['confidence_level']}")
        print(f"Recommended Action : {result['recommended_action']}")
        print(f"Severity           : {result['severity']}")
        print(f"Inference Time     : {result['inference_time_ms']:.2f} ms")
        print("----------------------------------------------")

    except Exception as e:

        print("Error:", e)

print("\nAll inputs processed successfully.")
