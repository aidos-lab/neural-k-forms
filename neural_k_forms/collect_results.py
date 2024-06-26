"""Prepare results based on `wandb.ai` output files."""

import argparse
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT", type=str, help="Input file")
    parser.add_argument("-c", "--column", type=str, default="test_accuracy")

    args = parser.parse_args()

    df = pd.read_csv(args.INPUT)

    columns = ["dataset", "model", args.column]
    columns_to_drop = [col for col in df.columns if col not in columns]

    df = df.drop(columns_to_drop, axis="columns")
    df[args.column] *= 100.0

    print(df.groupby(["dataset", "model"]).agg(["mean", "std"]).round(2))
