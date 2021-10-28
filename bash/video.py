#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize training process by creating images, and use ffmpeg to create videos.

ffmpeg must be installed. the shell executed command looks similar to this:
`ffmpeg -framerate 4 -i "embed-%05d.png" -r 200 -pix_fmt yuv420p training.mp4`
"""
import os
import subprocess

import click
import matplotlib.pyplot as plt
import pandas as pd


def create_images(directory, architecture, target):
    epochs = [d for d in os.listdir(directory) if d.startswith("0")]

    epochs.sort()
    print(f"{len(epochs)} epochs")

    path = os.path.join(directory, "img")
    if not os.path.exists(path):
        os.makedirs(os.path.join(directory, "img"))

    for epoch in epochs:
        e = int(epoch)

        path = os.path.join(directory, "img", f"embed-{target}-{e:05}.png")

        if os.path.exists(path):
            continue
        print(f"Creating {path} ...")

        os_meta_path = os.path.join(directory, epoch, "val/metadata.tsv")
        os_embed_path = os.path.join(directory, epoch, "val/tensors.tsv")

        if not os.path.exists(os_meta_path):
            print(f"Skipping {os_meta_path}")
            continue

        # train_meta_path = os.path.join(directory, epoch, f"train/metadata.tsv")
        # train_embed_path = os.path.join(directory, epoch, f"train/tensors.tsv")

        data1_meta = pd.read_csv(os_meta_path, delimiter="\t", header=0)
        data1_df = pd.read_csv(os_embed_path, delimiter="\t", header=None, names=["x", "y"])
        # train_meta_df = pd.read_csv(train_meta_path, delimiter="\t", header=0)
        # train_embed_df = pd.read_csv(train_embed_path, delimiter="\t", header=None, names=["x", "y"])

        # axes = plt.gca()

        # plot training
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 9))
        #
        # labels = train_meta_df["label"].unique()
        # labels.sort()
        # for label in labels:
        #     # print(label)
        #     data = train_embed_df[train_meta_df["label"] == label]
        #     ax1.scatter(data["x"], data["y"], cmap="rainbow", marker=",", s=1, label=label)

        centers_path = os.path.join(directory, epoch, "centers/tensors.tsv")
        if os.path.exists(centers_path):
            centers_df = pd.read_csv(centers_path, delimiter="\t", header=None, names=["x", "y"])
            # print(len(centers_df["x"]))
            ax1.scatter(centers_df["x"], centers_df["y"], c="black", marker="o")

        # ax1.set_ylim([-7, 7])
        # ax1.set_xlim([-7, 7])
        # ax1.legend(bbox_to_anchor=(1.2, 1.00))

        # plot openset stuff
        # ax2 = plt.subplot(122)
        labels = data1_meta["label"].unique()
        labels.sort()
        for label in labels:
            # print(label)
            data = data1_df[data1_meta["label"] == label]
            ax2.scatter(data["x"], data["y"], cmap="rainbow", marker=",", s=1, label=label)

        centers_path = os.path.join(directory, epoch, "centers/tensors.tsv")
        if os.path.exists(centers_path):
            centers_df = pd.read_csv(centers_path, delimiter="\t", header=None, names=["x", "y"])
            # print(len(centers_df["x"]))
            ax2.scatter(centers_df["x"], centers_df["y"], c="black", marker="o")

        ax2.set_ylim([-7, 7])
        ax2.set_xlim([-7, 7])
        ax2.legend(bbox_to_anchor=(1.2, 1.00))

        # ax1.tight_layout()
        # ax2.tight_layout()

        plt.suptitle(f"{architecture} {e}", y=1.05)
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()


@click.command()
@click.argument("directory")
@click.option("--architecture", "-a", type=str, default="unknown")
def main(directory, architecture):
    create_images(directory, architecture, "val")
    out_path = os.path.join(directory, "train-val.mp4")
    subprocess.run(["ffmpeg", "-framerate", "2", "-i", f"{directory}/img/embed-val-%05d.png", "-r", "200", "-vf",
                    "scale=1920:1080", f"{out_path}"])

    create_images(directory, architecture, "train")
    out_path = os.path.join(directory, "train-train.mp4")
    subprocess.run(["ffmpeg", "-framerate", "2", "-i", f"{directory}/img/embed-train-%05d.png", "-r", "200", "-vf",
                    "scale=1920:1080", f"{out_path}"])


if __name__ == '__main__':
    main()
