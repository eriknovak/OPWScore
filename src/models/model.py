# model packages
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer, logging

# visualization packages
import os
import numpy as np
import matplotlib.pyplot as plt

# python types
from typing import List

# distance functions
from src.utils.distances import (
    get_wasserstein_dist,
    get_cls_dist,
    get_max_dist,
    get_mean_dist,
)


class SeqMoverScore(pl.LightningModule):
    def __init__(
        self,
        model: str,
        tokenizer: str,
        dist_type: str = "emd",
        reg: float = 0.1,
        nit: int = 100,
        lr: float = 1e-5,
        eps: float = 1e-5,
        wd: float = 1e-2,
    ):
        super().__init__()

        if dist_type not in ["seq", "emd", "cls", "max", "mean"]:
            raise Exception(f"Unsupported distance type: {dist_type}")

        # save the model hyperparameters
        self.save_hyperparameters(
            "model", "tokenizer", "dist_type", "reg", "nit", "lr", "eps", "wd"
        )
        self.model = AutoModel.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def forward(self, system: str, references: List[str]):
        """Calculate the distance between the system and references
        Args:
            system (str): The system generated text.
            references (List[str]): The list of reference texts.
        Returns:
            (distances, cost_matrix, transport_matrix): Returns the distances,
                cost matrix and transport matrix, all in the form of a torch.Tensor.
        """
        sys_inputs = self.tokenizer(
            system, padding=True, truncation=True, return_tensors="pt"
        )
        ref_inputs = self.tokenizer(
            references, padding=True, truncation=True, return_tensors="pt"
        )
        # get the embeddings of the services
        sys_embed = self.model(**sys_inputs)["last_hidden_state"]
        ref_embed = self.model(**ref_inputs)["last_hidden_state"]

        # duplicate the sys_embeds and sys_attns to match ref_embed batch size
        sys_embed = sys_embed.repeat(ref_embed.shape[0], 1, 1)
        sys_attns = sys_inputs["attention_mask"].repeat(ref_embed.shape[0], 1)
        ref_attns = ref_inputs["attention_mask"]

        if self.hparams.dist_type == "emd":
            return get_wasserstein_dist(
                sys_embed,
                ref_embed,
                sys_attns,
                ref_attns,
                self.hparams.reg,
                self.hparams.nit,
            )
        elif self.hparams.dist_type == "cls":
            return get_cls_dist(sys_embed, ref_embed, sys_attns, ref_attns)
        elif self.hparams.dist_type == "max":
            return get_max_dist(sys_embed, ref_embed, sys_attns, ref_attns)
        elif self.hparams.dist_type == "mean":
            return get_mean_dist(sys_embed, ref_embed, sys_attns, ref_attns)
        else:
            raise Exception(f"Unsupported distance type: {self.hparams.dist_type}")

    def visualize(self, system: str, references: List[str], image_path: str):
        """Visualize the distances between the system and references
        Args:
            system (str): The system generated text.
            references (List[str]): The list of reference texts.
        Returns:
            (None): Returns None.
        """

        # check if image path is provided
        if not image_path:
            raise Exception(f"Image path not provided!")

        # check if visualization is supported
        if self.hparams.dist_type not in ["seq", "emd"]:
            raise Exception(
                f"Unable to visualize for distance type: {self.hparams.dist_type}"
            )

        # get the input ids of the system and references
        sys_inputs = self.tokenizer(
            system, padding=True, truncation=True, return_tensors="pt"
        )["input_ids"]
        ref_inputs = self.tokenizer(
            references, padding=True, truncation=True, return_tensors="pt"
        )["input_ids"]

        # get the distances, cost matrix and transportation matrix
        distances, Cm, Tm = self.forward(system, references)

        # convert to numpy
        Cm_np = Cm.detach().numpy()
        Tm_np = Tm.detach().numpy()

        # get the max sizes (batch, system, references)
        batch_size, sys_size, ref_size = Cm.shape

        # prepare the figure size based on the input size
        x_size = 2 * ref_size
        y_size = 1 * batch_size * sys_size

        # initialize the figure object
        fig, big_axes = plt.subplots(
            nrows=batch_size, ncols=1, figsize=(x_size, y_size)
        )

        # figure preprocessing
        for big_ax in big_axes:
            # Turn off axis lines and ticks of the big subplot
            # obs alpha is 0 in RGBA string!
            big_ax.tick_params(
                labelcolor=(1.0, 1.0, 1.0, 0.0),
                top=False,
                bottom=False,
                left=False,
                right=False,
            )
            # removes the white frame
            big_ax._frameon = False

        for batch_id in range(batch_size):
            ax_distance = fig.add_subplot(batch_size, 2, 2 * batch_id + 1)
            ax_transport = fig.add_subplot(batch_size, 2, 2 * batch_id + 2)

            # system and document tokens
            sys_tokens = self.tokenizer.convert_ids_to_tokens(sys_inputs[0])
            ref_tokens = self.tokenizer.convert_ids_to_tokens(ref_inputs[batch_id])

            # get the special token's ids
            sys_special_ids = [
                idx
                for idx, token in enumerate(sys_tokens)
                if token in self.tokenizer.all_special_tokens
            ]
            ref_special_ids = [
                idx
                for idx, token in enumerate(ref_tokens)
                if token in self.tokenizer.all_special_tokens
            ]

            # cleanup the tokens and matrices
            sys_tokens = np.delete(sys_tokens, sys_special_ids)
            ref_tokens = np.delete(ref_tokens, ref_special_ids)
            Cm_clean = np.delete(
                np.delete(Cm_np[batch_id], sys_special_ids, axis=0),
                ref_special_ids,
                axis=1,
            )
            Tm_clean = np.delete(
                np.delete(Tm_np[batch_id], sys_special_ids, axis=0),
                ref_special_ids,
                axis=1,
            )
            # the cosine distance matrix
            ax_distance.set_title("distance matrix", fontsize="large")
            cmim = ax_distance.imshow(Cm_clean, cmap="PuBu", vmin=0)
            cbar = fig.colorbar(cmim, ax=ax_distance, shrink=0.9)
            cbar.ax.set_ylabel(
                "token distances", rotation=-90, va="bottom", fontsize=14
            )

            # the transport matrix
            ax_transport.set_title("transportation matrix", fontsize="large")
            tmim = ax_transport.imshow(Tm_clean / Tm_clean.max(), cmap="Greens", vmin=0)
            cbar = fig.colorbar(tmim, ax=ax_transport, shrink=1)
            cbar.ax.set_ylabel(
                "token mass transportation", rotation=-90, va="bottom", fontsize=14
            )

            plots = [ax_distance, ax_transport]
            for plot_id in range(len(plots)):
                # set the x and y ticks
                plots[plot_id].set_yticks(np.arange(len(sys_tokens)))
                plots[plot_id].set_xticks(np.arange(len(ref_tokens)))

                # add the x and y labels
                plots[plot_id].set_yticklabels(sys_tokens, fontsize=12)
                plots[plot_id].set_xticklabels(ref_tokens, fontsize=12)

                # rotate the x labels a bit
                plt.setp(
                    plots[plot_id].get_xticklabels(),
                    rotation=45,
                    ha="right",
                    rotation_mode="anchor",
                )

            # assign the document score (lower scores -> greater rank)
            d_score = round(distances[batch_id].item(), 3)
            plots[0].set_ylabel(
                f"Distance: {d_score}",
                rotation=-90,
                va="bottom",
                labelpad=30,
                fontsize=14,
            )

        # make the layout more tight
        plt.tight_layout()

        if image_path:
            # save the plot in a file
            plt.savefig(image_path, dpi=300, transparent=True, bbox_inches="tight")

        return None
