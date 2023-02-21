# model packages
import os
import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer

# visualization packages
import numpy as np
import matplotlib.pyplot as plt

# python types
from typing import List

# distance functions
from src.utils.distances import (
    get_seq_wasserstein_dist,
    get_wasserstein_dist,
    get_cls_dist,
    get_max_dist,
    get_mean_dist,
    get_model,
)
from src.utils.weight_store import WeightStore

# ===================================================================
# Predefined Weight Stores Paths
# ===================================================================

WEIGHT_STORES_PATH = os.path.join("..", "results", "weight_stores")

# ===================================================================
# Model Definitions
# ===================================================================

# enable NaN detection in PyTorch
torch.autograd.set_detect_anomaly(True)


class OPWScore(pl.LightningModule):
    def __init__(
        self,
        distance: str = "seq",
        weight_dist: str = "idf",
        temporal_type: str = "OPW",
        lang: str = "en",
        reg1: float = 0.1,
        reg2: float = 0.1,
        nit: int = 100,
    ):
        super().__init__()

        if distance not in ["seq", "emd", "cls", "max", "mean"]:
            raise Exception(f"Unsupported distance type: {distance}")

        if lang not in ["en", "cs", "de", "fi", "ru", "tr", "et", "zh"]:
            raise Exception(f"Unsupported language type: {lang}")

        if distance == "seq" and temporal_type not in ["TCOT", "OPW"]:
            raise Exception(f"Unsupported temporal type: {temporal_type}")

        # save the model hyperparameters
        self.save_hyperparameters(
            "distance",
            "weight_dist",
            "temporal_type",
            "lang",
            "reg1",
            "reg2",
            "nit",
        )
        if lang == "en":
            # taken from the BERTScore paper
            model = "roberta-large-mnli"
            tokenizer = "roberta-large-mnli"
            num_layers = 19
        else:
            # taken from the BERTScore paper
            model = "bert-base-multilingual-cased"
            tokenizer = "bert-base-multilingual-cased"
            num_layers = 9

        self.model = get_model(model, num_layers)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.ws = (
            None
            if weight_dist != "idf"
            else WeightStore.load(
                os.path.join(
                    WEIGHT_STORES_PATH,
                    f"weight_store.{lang}.wmt17.{tokenizer.replace('/', '_')}.pickle",
                )
            )
        )

    def forward(
        self, predictions: str or List[str], references: List[str], all_layers=False
    ):
        """Calculate the distance between the system and references
        Args:
            predictions (str or List[str]): The system generated text.
            references (List[str]): The list of reference texts.
        Returns:
            (distances, cost_matrix, transport_matrix): Returns the distances,
                cost matrix and transport matrix, all in the form of a torch.Tensor.
        """
        sys_inputs = self.tokenizer(
            predictions,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        ref_inputs = self.tokenizer(
            references,
            padding=True,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        # get the embeddings of the services
        sys_embed = self.model(**sys_inputs, output_hidden_states=all_layers)[
            "last_hidden_state"
        ]
        ref_embed = self.model(**ref_inputs, output_hidden_states=all_layers)[
            "last_hidden_state"
        ]
        sys_input_ids = sys_inputs["input_ids"]
        ref_input_ids = ref_inputs["input_ids"]
        sys_attns = sys_inputs["attention_mask"]
        ref_attns = ref_inputs["attention_mask"]

        if type(predictions) is not list:
            # duplicate the sys_embeds and sys_attns to match ref_embed batch size
            sys_embed = sys_embed.repeat(ref_embed.shape[0], 1, 1)
            sys_attns = sys_attns.repeat(ref_embed.shape[0], 1)
            sys_input_ids = sys_input_ids.repeat(ref_embed.shape[0], 1)

        # TODO: remove special tokens ([CLS], [SEP], <s>, </s>) from tensors

        if self.hparams.distance == "seq":
            return get_seq_wasserstein_dist(
                self.hparams.weight_dist,
                self.ws,
                sys_embed,
                ref_embed,
                sys_input_ids,
                ref_input_ids,
                sys_attns,
                ref_attns,
                self.hparams.reg1,
                self.hparams.reg2,
                self.hparams.nit,
                self.hparams.temporal_type,
            )

        elif self.hparams.distance == "emd":
            return get_wasserstein_dist(
                self.hparams.weight_dist,
                self.ws,
                sys_embed,
                ref_embed,
                sys_input_ids,
                ref_input_ids,
                sys_attns,
                ref_attns,
                self.hparams.reg1,
                self.hparams.nit,
            )
        elif self.hparams.distance == "cls":
            return get_cls_dist(sys_embed, ref_embed, sys_attns, ref_attns)
        elif self.hparams.distance == "max":
            return get_max_dist(sys_embed, ref_embed, sys_attns, ref_attns)
        elif self.hparams.distance == "mean":
            return get_mean_dist(sys_embed, ref_embed, sys_attns, ref_attns)
        else:
            raise Exception(f"Unsupported distance type: {self.hparams.dist_type}")

    def visualize(
        self,
        predictions: str or List[str],
        references: List[str],
        image_path: str = None,
    ):
        """Visualize the distances between the system and references
        Args:
            predictions (str): The system generated text.
            references (List[str]): The list of reference texts.
            image_path (str): The path to where the image is stored (optional).
        Returns:
            (None): Returns None.
        """

        # check if visualization is supported
        if self.hparams.distance not in ["seq", "emd"]:
            raise Exception(
                f"Unable to visualize for distance type: {self.hparams.dist_type}"
            )

        # get the input ids of the system and references
        sys_inputs = self.tokenizer(
            predictions, padding=True, truncation=True, return_tensors="pt"
        )["input_ids"]
        ref_inputs = self.tokenizer(
            references, padding=True, truncation=True, return_tensors="pt"
        )["input_ids"]

        # get the distances, cost matrix and transportation matrix
        distances, Cm, Tm = self.forward(predictions, references)

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
        if batch_size == 1:
            big_axes = [big_axes]

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
            sys_tokens = self.tokenizer.convert_ids_to_tokens(
                sys_inputs[batch_id] if type(predictions) is list else sys_inputs[0]
            )
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
            cmim = ax_distance.imshow(Cm_clean, cmap="Blues", vmin=0)
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
