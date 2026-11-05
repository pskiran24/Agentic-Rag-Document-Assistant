
import os
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

# ============================================================
# 1. LANGSMITH CONFIGURATION
# ============================================================

DATASET_NAME = "attention-rag-evaluation-v1"

client = Client(
    api_key=os.getenv("LANGSMITH_API_KEY")
)

# ============================================================
# 2. EVALUATION EXAMPLES
# ============================================================

examples = [
    {
        "inputs": {
            "question": "What is self-attention?"
        },
        "outputs": {
            "reference": (
                "Self-attention is an attention mechanism that "
                "relates different positions of a single sequence "
                "when computing a representation of that sequence."
            )
        }
    },
    {
        "inputs": {
            "question": "What is the purpose of multi-head attention?"
        },
        "outputs": {
            "reference": (
                "Multi-head attention allows the model to jointly "
                "attend to information from different representation "
                "subspaces at different positions."
            )
        }
    },
    {
        "inputs": {
            "question": "How does scaled dot-product attention work?"
        },
        "outputs": {
            "reference": (
                "Scaled dot-product attention computes the dot "
                "products of queries and keys, divides the results "
                "by the square root of the key dimension, applies "
                "softmax, and uses the resulting weights to combine "
                "the values."
            )
        }
    },
    {
        "inputs": {
            "question": (
                "How many attention heads are used in the "
                "original Transformer?"
            )
        },
        "outputs": {
            "reference": (
                "The original Transformer uses eight attention "
                "heads in its multi-head attention layers."
            )
        }
    },
    {
        "inputs": {
            "question": (
                "Why does scaled dot-product attention divide "
                "by the square root of dk?"
            )
        },
        "outputs": {
            "reference": (
                "When the key dimension is large, dot products "
                "can become large in magnitude. This can push "
                "the softmax function into regions with extremely "
                "small gradients. Dividing by the square root "
                "of dk helps counteract this effect."
            )
        }
    }
]


# ============================================================
# 3. ADD EXAMPLES TO DATASET
# ============================================================

def add_examples():

    if not os.getenv("LANGSMITH_API_KEY"):
        raise ValueError(
            "LANGSMITH_API_KEY is missing from your .env file."
        )

    # Confirm that the dataset exists.
    dataset = client.read_dataset(
        dataset_name=DATASET_NAME
    )

    print(f"Dataset found: {dataset.name}")

    # Avoid accidentally adding duplicates.
    existing_examples = list(
        client.list_examples(
            dataset_id=dataset.id
        )
    )

    if existing_examples:
        print(
            f"Dataset already contains "
            f"{len(existing_examples)} examples."
        )
        print("No new examples were added.")
        return

    # Create examples.
    client.create_examples(
        dataset_id=dataset.id,
        inputs=[
            example["inputs"]
            for example in examples
        ],
        outputs=[
            example["outputs"]
            for example in examples
        ],
    )

    print(
        f"\nSuccessfully added {len(examples)} examples!"
    )

    # Verify the dataset.
    saved_examples = list(
        client.list_examples(
            dataset_id=dataset.id
        )
    )

    print(
        f"Total examples in dataset: "
        f"{len(saved_examples)}"
    )


if __name__ == "__main__":
    add_examples()
