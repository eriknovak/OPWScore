import math
import torch
import torch.nn.functional as f
from transformers import AutoModel
from src.utils.weight_store import WeightStore


# ===================================================================
# Sinkhorn algorithm and Wasserstein Distance Helper Functions
# ===================================================================


def get_model(model_type, num_layers=-1):
    model = AutoModel.from_pretrained(model_type)
    model.eval()

    if hasattr(model, "decoder") and hasattr(model, "encoder"):
        model = model.encoder

    if num_layers > 0:
        if hasattr(model, "n_layers"):  # xlm
            assert (
                0 <= num_layers <= model.n_layers
            ), f"Invalid num_layers: num_layers should be between 0 and {model.n_layers} for {model_type}"
            model.n_layers = num_layers
        elif hasattr(model, "layer"):  # xlnet
            assert (
                0 <= num_layers <= len(model.layer)
            ), f"Invalid num_layers: num_layers should be between 0 and {len(model.layer)} for {model_type}"
            model.layer = torch.nn.ModuleList(
                [layer for layer in model.layer[:num_layers]]
            )
        elif hasattr(model, "encoder"):  # albert
            if hasattr(model.encoder, "albert_layer_groups"):
                assert (
                    0 <= num_layers <= model.encoder.config.num_hidden_layers
                ), f"Invalid num_layers: num_layers should be between 0 and {model.encoder.config.num_hidden_layers} for {model_type}"
                model.encoder.config.num_hidden_layers = num_layers
            elif hasattr(model.encoder, "block"):  # t5
                assert (
                    0 <= num_layers <= len(model.encoder.block)
                ), f"Invalid num_layers: num_layers should be between 0 and {len(model.encoder.block)} for {model_type}"
                model.encoder.block = torch.nn.ModuleList(
                    [layer for layer in model.encoder.block[:num_layers]]
                )
            else:  # bert, roberta
                assert (
                    0 <= num_layers <= len(model.encoder.layer)
                ), f"Invalid num_layers: num_layers should be between 0 and {len(model.encoder.layer)} for {model_type}"
                model.encoder.layer = torch.nn.ModuleList(
                    [layer for layer in model.encoder.layer[:num_layers]]
                )
        elif hasattr(model, "transformer"):  # bert, roberta
            assert (
                0 <= num_layers <= len(model.transformer.layer)
            ), f"Invalid num_layers: num_layers should be between 0 and {len(model.transformer.layer)} for {model_type}"
            model.transformer.layer = torch.nn.ModuleList(
                [layer for layer in model.transformer.layer[:num_layers]]
            )
        elif hasattr(model, "layers"):  # bart
            assert (
                0 <= num_layers <= len(model.layers)
            ), f"Invalid num_layers: num_layers should be between 0 and {len(model.layers)} for {model_type}"
            model.layers = torch.nn.ModuleList(
                [layer for layer in model.layers[:num_layers]]
            )
        else:
            raise ValueError("Not supported")

    return model


# ===================================================================
# Sinkhorn algorithm and Wasserstein Distance Helper Functions
# ===================================================================


def __replace_zeros(tensor: torch.Tensor, eps: float = 1e-32):
    # we create a copy of the original tensor,
    # because of the way we are replacing them.
    res = tensor.clone()
    res[tensor == 0] = eps
    return res


def get_cost_matrix(source: torch.Tensor, target: torch.Tensor):
    """Calculates the cost matrix between the source and target tensors
    Args:
        source (torch.Tensor): The source tensor.
        target (torch.Tensor): The target tensor.
    Returns:
        torch.Tensor: The cost matrix between the source and target tensors.
    """
    # normalize the embeddings
    source.div_(torch.norm(source, dim=-1).unsqueeze(-1))
    target.div_(torch.norm(target, dim=-1).unsqueeze(-1))
    # calculate and return the cost matrix
    cost_matrix = source.matmul(target.transpose(1, 2))
    cost_matrix = torch.ones_like(cost_matrix) - cost_matrix
    return cost_matrix


def get_weight_dist(
    attn: torch.Tensor,
    input_ids: torch.Tensor,
    weight_dist: str,
    ws: WeightStore,
):
    """Generates the weight distribution tensor
    Args:
        attn (torch.Tensor): The attention tensor.
        input_ids (torch.Tensor): The input IDs tensor.
        weight_dist (str): The weight distribution.
        ws (WeightStore): The weight store.
    Returns:
        torch.Tensor: The weight distribution tensor.
    """
    if weight_dist == "uniform":
        # uniform distribution
        dist = torch.ones(attn.shape) * attn
    elif weight_dist == "idf":
        # IDF weights distribution
        dist = (
            torch.tensor(
                [ws.get_idf(ws.get_word(id.item())) for id in input_ids.flatten()]
            ).reshape(input_ids.shape)
            * attn
        )
    else:
        raise Exception(f"Invalid weight distribution value: {weight_dist}")
    # normalize the distribution
    dist.div_(dist.sum(dim=1).view(-1, 1).repeat(1, attn.shape[1]))
    dist.requires_grad = False
    return dist


# def get_prior_distribution(N: int, M: int, std: float = 1):
#     """The prior distribution used to fit the Wasserstein distance to a particular pattern
#     Args:
#         N (int): The number of tokens in the source tensor.
#         M (int): The number of tokens in the target tensor.
#         std (float): The standard deviation used to define the prior distribution.
#     Returns:
#         torch.Tensor: The prior distribution.

#     """
#     Ni = torch.Tensor([[i / N for i in range(1, N + 1)] for _ in range(M)]).T
#     Mj = torch.Tensor([[j / M for j in range(1, M + 1)] for _ in range(N)])
#     Lij = torch.abs(Ni - Mj).div(math.sqrt(1 / N**2 + 1 / M**2)).pow(2)
#     return Lij / (2 + std**2) + math.log(std * math.sqrt(2 * math.pi))


def get_distance_matrix_from_diagonal_line(N: int, M: int):
    """Get the distance matrix of the (i, j) point from the diagonal line
    Args:
        N (int): The number of tokens in the source tensor.
        M (int): The number of tokens in the target tensor.
    Returns:
        torch.Tensor: The distance matrix.
    """
    Ni = torch.Tensor([[i / N for i in range(1, N + 1)] for _ in range(M)]).T
    Mj = torch.Tensor([[j / M for j in range(1, M + 1)] for _ in range(N)])
    return torch.abs(Ni - Mj) / math.sqrt(1 / N**2 + 1 / M**2)


def get_prior_distribution(N: int, M: int, std: float = 1):
    """The prior distribution used to fit the Wasserstein distance to a particular pattern
    Args:
        N (int): The number of tokens in the source tensor.
        M (int): The number of tokens in the target tensor.
        std (float): The standard deviation used to define the prior distribution.
    Returns:
        torch.Tensor: The prior distribution.

    """
    Lij = get_distance_matrix_from_diagonal_line(N, M)
    exp = torch.exp(-torch.pow(Lij, 2) / (2 * std**2))
    return 1 / (std * math.sqrt(2 * math.pi)) * exp


def inverse_difference_moment_matrix(N: int, M: int):
    """The inverse different moment matrix for measuring local homogeneity of the transport matrix
    Args:
        N (int): The number of tokens in the source tensor.
        M (int): The number of tokens in the target tensor.
    Returns:
        torch.Tensor: The inverse different moment matrix.
    """
    Ni = torch.Tensor([[i / N for i in range(1, N + 1)] for _ in range(M)]).T
    Mj = torch.Tensor([[j / M for j in range(1, M + 1)] for _ in range(N)])
    return 1 / ((Ni - Mj).pow(2) + 1)


def get_temporal_cost_matrix(N: int, M: int):
    """The temporal distance matrix
    Args:
        N (int): The number of tokens in the source tensor.
        M (int): The number of tokens in the target tensor.
    Returns:
        torch.Tensor: The inverse different moment matrix.
    """
    Ni = torch.Tensor([[i / N for i in range(1, N + 1)] for _ in range(M)]).T
    Mj = torch.Tensor([[j / M for j in range(1, M + 1)] for _ in range(N)])
    return (Ni - Mj).abs() + 1


def get_otw_cost_matrix(
    D: torch.Tensor, lambda1: float, lambda2: float, std: float = 1.0
):
    _, N, M = D.shape
    E = inverse_difference_moment_matrix(N, M)
    F = get_distance_matrix_from_diagonal_line(N, M)
    return (
        D
        - lambda1 * E
        + lambda2 * (F / (2 * std**2) + math.log(std * math.sqrt(2 * math.pi)))
    )


# ===================================================================
# Sinkhorn Algorithm and Wasserstein Distance
# ===================================================================


def sinkhorn(
    source_dist: torch.FloatTensor,
    target_dist: torch.FloatTensor,
    cost_matrix: torch.Tensor,
    reg: float,
    nit: int = 100,
    eps: float = 1e-32,
):
    """The sinkhorn algorithm adapted for PyTorch from the
        PythonOT library <https://pythonot.github.io/>.
    Args:
        source_dist (torch.Tensor): The source distribution tensor.
        target_dist (torch.Tensor): The target distribution tensor.
        cost_matrix (torch.Tensor): The cost matrix tensor.
        reg (float): The regularization factor.
        nit (int): Number of maximum iterations.
        eps (float): The zero-value substitute value.
    Returns:
        torch.Tensor: The transportation matrix.
    """
    # assert the tensor dimensions
    assert (
        source_dist.shape[0] == cost_matrix.shape[0]
        and source_dist.shape[1] == cost_matrix.shape[1]
    )
    assert (
        target_dist.shape[0] == cost_matrix.shape[0]
        and target_dist.shape[1] == cost_matrix.shape[2]
    )

    # prepare the initial variables
    K = torch.exp(cost_matrix / -reg)
    Kp = (1 / source_dist).reshape(source_dist.shape[0], -1, 1) * K

    # initialize the u and v tensors
    u = torch.ones_like(source_dist).div(source_dist.shape[1])
    v = torch.ones_like(target_dist).div(target_dist.shape[1])

    istep = 0
    while istep < nit:
        # calculate K.T * u for each example in batch
        KTransposeU = K.transpose(1, 2).bmm(u.unsqueeze(2)).squeeze(2)
        # calculate the v_{i} tensor
        v = target_dist.div(__replace_zeros(KTransposeU, eps))
        # calculate the u_{i} tensor
        u = 1.0 / __replace_zeros(Kp.bmm(v.unsqueeze(2)).squeeze(2), eps)
        # go to next step
        istep = istep + 1
    # calculate the transport matrix
    U = torch.diag_embed(u)
    V = torch.diag_embed(v)
    return U.bmm(K).bmm(V)


def get_wasserstein_dist(
    weight_dist: str,
    weight_store: WeightStore,
    source_embed: torch.Tensor,
    target_embed: torch.Tensor,
    source_input_ids: torch.Tensor,
    target_input_ids: torch.Tensor,
    source_attns: torch.Tensor,
    target_attns: torch.Tensor,
    reg: float,
    nit: int = 100,
):
    """Gets the Wasserstein distances between the source and target embeddings.
    Args:
        weight_dist: The weight distribution option.
        weight_store: The weight distribution store.
        source_embed: The source embeddings.
        target_embed: The target embeddings.
        source_input_ids: The source token ids.
        target_input_ids: The target token ids.
        source_attns: The source attention masks.
        target_attns: The target attention masks.
        reg: The regularization factor.
        nit: The number of iterations.
    Returns:
        torch.Tensor, torch.Tensor, torch.Tensor: The tuple containing
            - the wasserstein distances between the embeddings,
            - the cost matrix, and
            - the transportation matrix.
    """
    # get the cost matrix
    C = get_cost_matrix(source_embed, target_embed)
    # get query and texts distributions
    q_dist = get_weight_dist(source_attns, source_input_ids, weight_dist, weight_store)
    d_dist = get_weight_dist(target_attns, target_input_ids, weight_dist, weight_store)
    # solve the optimal transport problem
    T = sinkhorn(q_dist, d_dist, C, reg, nit)
    # calculate the distances
    distances = (C * T).view(C.shape[0], -1).sum(dim=1)
    # return the loss, transport and cost matrices
    return distances, C, T


def seq_sinkhorn(
    source_dist: torch.FloatTensor,
    target_dist: torch.FloatTensor,
    cost_matrix: torch.Tensor,
    reg1: float,
    reg2: float,
    nit: int = 100,
    eps: float = 1e-32,
):
    """The sequence sinkhorn algorithm adapted for PyTorch from the
        PythonOT library <https://pythonot.github.io/>.
    Args:
        source_dist: The source distribution tensor.
        target_dist: The target distribution tensor.
        cost_matrix: The cost matrix tensor.
        reg: The regularization factor.
        nit: Number of maximum iterations.
    Returns:
        torch.Tensor: The transportation matrix.
    """
    # assert the tensor dimensions
    assert (
        source_dist.shape[0] == cost_matrix.shape[0]
        and source_dist.shape[1] == cost_matrix.shape[1]
    )
    assert (
        target_dist.shape[0] == cost_matrix.shape[0]
        and target_dist.shape[1] == cost_matrix.shape[2]
    )
    # construct the sinkhorn matrix
    _, N, M = cost_matrix.shape
    P_ij = get_prior_distribution(N, M)
    S_ij = inverse_difference_moment_matrix(N, M)
    K = P_ij * torch.exp((reg1 * S_ij - cost_matrix) / reg2)
    Kp = (1 / source_dist).reshape(source_dist.shape[0], -1, 1) * K

    # initialize the u and v tensors
    u = torch.ones_like(source_dist) / source_dist.shape[1]
    v = torch.ones_like(target_dist) / target_dist.shape[1]

    istep = 0
    while istep < nit:
        # calculate K.T * u for each example in batch
        KTransposeU = K.transpose(1, 2).bmm(u.unsqueeze(2)).squeeze(2)
        # calculate the v_{i} tensor
        v = target_dist / (KTransposeU + eps)
        # calculate the u_{i} tensor
        u = 1.0 / (Kp.bmm(v.unsqueeze(2)).squeeze(2) + eps)
        # go to next step
        istep = istep + 1
    # calculate the transport matrix
    U = torch.diag_embed(u)
    V = torch.diag_embed(v)
    return U.bmm(K).bmm(V)


def get_seq_wasserstein_dist(
    weight_dist: str,
    weight_store: WeightStore,
    source_embed: torch.Tensor,
    target_embed: torch.Tensor,
    source_input_ids: torch.Tensor,
    target_input_ids: torch.Tensor,
    source_attns: torch.Tensor,
    target_attns: torch.Tensor,
    reg1: float,
    reg2: float,
    nit: int,
    temporal_type: str,
):
    """Gets the sequence Wasserstein distances between the source and target embeddings.
    Args:
        weight_dist: The weight distribution option.
        weight_store: The weight distribution store.
        source_embed: The source embeddings.
        target_embed: The target embeddings.
        source_input_ids: The source token ids.
        target_input_ids: The target token ids.
        source_attns: The source attention masks.
        target_attns: The target attention masks.
        reg1: The first regularization factor associated with the inverse difference
            moment.
        reg2: The second regularization factor associated with the Kullback-Leibler
            Divergence between the transport matrix and the prior (diagonal)
            distribution.
        nit: The number of iterations.
        type: The type of temporal OT used to calculate. Options:
            - TCOT
            - OPW
    Returns:
        torch.Tensor, torch.Tensor, torch.Tensor: The tuple containing
            - the wasserstein distances between the embeddings,
            - the cost matrix, and
            - the transportation matrix.
    """
    # get query and texts distributions
    q_dist = get_weight_dist(source_attns, source_input_ids, weight_dist, weight_store)
    d_dist = get_weight_dist(target_attns, target_input_ids, weight_dist, weight_store)

    # get the cost matrix
    C = get_cost_matrix(source_embed, target_embed)
    _, N, M = C.shape
    if temporal_type == "TCOT":
        D = get_temporal_cost_matrix(N, M)
        C = C * D
        T = sinkhorn(q_dist, d_dist, C, reg1, nit)
    elif temporal_type == "OPW":
        D = get_otw_cost_matrix(C, reg1, reg2)
        T = sinkhorn(q_dist, d_dist, D, reg2, nit)
    else:
        raise Exception(f"Unrecognized type: {temporal_type}")

    # calculate the distances
    distances = (C * T).view(C.shape[0], -1).sum(dim=1)
    # return the loss, transport and cost matrices
    return distances, C, T


# ===================================================================
# Support for Cosine Distance-Based Similarity Measures
# ===================================================================


def prep_relevant_embeds(embed: torch.Tensor, attns: torch.Tensor):
    """Prepares the token embeddings for mean and max types
    Args:
        embed: The token embeddings.
        attns: The attention mask.
    Returns:
        torch.Tensor: The tensor with padding embeddings set to zero.
    """
    return embed * attns.unsqueeze(2).repeat(1, 1, embed.shape[2])


def __get_cosine_dist(source_embed: torch.Tensor, target_embed: torch.Tensor):
    """Calculate the cosine distance between the source and target embeddings
    Args:
        source_embed: The source embeddings.
        target_embed: The target embeddings.
    Returns:
        torch.Tensor, None, None: The tuple containing the distances between
            the embeddings and two None values.
    """

    # normalize the vectors before calculating
    source_embed = f.normalize(source_embed, p=2, dim=1)
    target_embed = f.normalize(target_embed, p=2, dim=1)

    # calculate the mean distances
    distances = source_embed.matmul(target_embed.T).diagonal()
    distances = torch.ones_like(distances) - distances
    return distances, None, None


def get_cls_dist(
    source_embed: torch.Tensor,
    target_embed: torch.Tensor,
    source_attns: torch.Tensor,
    target_attns: torch.Tensor,
):
    """Gets the cosine distance using the first [CLS] token
    Args:
        source_embed: The source embeddings.
        target_embed: The target embeddings.
        source_attns: The source attention masks. Not used by this method; it is
            here just for having the consistent input schema.
        target_attns: The target attention masks. Not used by this method; it is
            here just for having the consistent input schema.
    Returns:
        torch.Tensor, None, None: The tuple containing the distances between
            the embeddings and two None values.
    """

    # get the first [CLS] token
    source_embed = source_embed[:, 0, :]
    target_embed = target_embed[:, 0, :]
    # return the cosine distance
    return __get_cosine_dist(source_embed, target_embed)


def get_max_dist(
    source_embed: torch.Tensor,
    target_embed: torch.Tensor,
    source_attns: torch.Tensor,
    target_attns: torch.Tensor,
):
    """Get the cosine distance using the maximum values of the embeddings
    Args:
        source_embed: The source embeddings.
        target_embed: The target embeddings.
        source_attns: The source attention masks. Not used by this method; it is
            here just for having the consistent input schema.
        target_attns: The target attention masks. Not used by this method; it is
            here just for having the consistent input schema.
    Returns:
        torch.Tensor, None, None: The tuple containing the distances between
            the embeddings and two None values.
    """

    # get the maximum values of the embeddings
    source_embed, _ = prep_relevant_embeds(source_embed, source_attns).max(dim=1)
    target_embed, _ = prep_relevant_embeds(target_embed, target_attns).max(dim=1)
    # return the cosine distance
    return __get_cosine_dist(source_embed, target_embed)


def get_mean_dist(
    source_embed: torch.Tensor,
    target_embed: torch.Tensor,
    source_attns: torch.Tensor,
    target_attns: torch.Tensor,
):
    """Get the cosine distance using the mean values of the embeddings
    Args:
        source_embed: The source embeddings.
        target_embed: The target embeddings.
        source_attns: The source attention masks. Not used by this method; it is
            here just for having the consistent input schema.
        target_attns: The target attention masks. Not used by this method; it is
            here just for having the consistent input schema.
    Returns:
        torch.Tensor, None, None: The tuple containing the distances between
            the embeddings and two None values.
    """

    # get the maximum values of the embeddings
    source_embed = prep_relevant_embeds(source_embed, source_attns).mean(dim=1)
    target_embed = prep_relevant_embeds(target_embed, target_attns).mean(dim=1)
    # return the cosine distance
    return __get_cosine_dist(source_embed, target_embed)
