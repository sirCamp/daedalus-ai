"""Minimal SFT with HuggingFace TRL — tiny model for demo purposes."""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--dataset", type=str, default="HuggingFaceTB/smoltalk")
    parser.add_argument("--dataset_config", type=str, default="everyday-conversations")
    parser.add_argument("--dataset_split", type=str, default="train[:1000]")
    parser.add_argument("--eval_split", type=str, default="test[:100]")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./output")
    args = parser.parse_args()

    # Lazy imports — only load heavy libs when actually running
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    logger.info("Loading model: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading dataset: %s/%s [%s]", args.dataset, args.dataset_config, args.dataset_split)
    train_dataset = load_dataset(args.dataset, args.dataset_config, split=args.dataset_split)
    eval_dataset = load_dataset(args.dataset, args.dataset_config, split=args.eval_split)
    logger.info("Train: %d, Eval: %d", len(train_dataset), len(eval_dataset))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_length=args.max_length,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        seed=args.seed,
        bf16=False,  # CPU-safe default
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting training")
    train_result = trainer.train()

    # Final evaluation
    logger.info("Final evaluation")
    eval_results = trainer.evaluate()

    results = {
        "eval": {
            "loss": round(eval_results["eval_loss"], 4),
            "train_loss": round(train_result.training_loss, 4),
            "train_runtime": round(train_result.metrics["train_runtime"], 1),
            "train_samples_per_second": round(train_result.metrics["train_samples_per_second"], 2),
        }
    }

    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    logger.info("Results saved to %s", results_path)
    logger.info("eval_loss=%.4f", eval_results["eval_loss"])
    logger.info("train_loss=%.4f", train_result.training_loss)


if __name__ == "__main__":
    main()
