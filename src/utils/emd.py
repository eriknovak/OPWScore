import torch
import torch.nn as nn
import torch.nn.functional as f

def get_cost_matrix(source: torch.Tensor, target: torch.Tensor):
    """Calculates the cost matrix between the source and target tensors
    Args:
        source: The source tensor.
        target: The target tensor.
    Returns:
        torch.Tensor: The cost matrix between the source and target tensors.
    """
    # normalize the embeddings
    source = f.normalize(source, p=2, dim=2)
    target = f.normalize(target, p=2, dim=2)
    # calculate and return the cost matrix
    cost_matrix = source.matmul(target.transpose(1, 2))
    cost_matrix = torch.ones_like(cost_matrix) - cost_matrix
    return cost_matrix


def get_weight_dist(attn: torch.FloatTensor):
    """Generates the weight distribution tensor
    Args:
        attn: The attention tensor.
    Returns:
        torch.Tensor: The weight distribution tensor.
    """
    dist = torch.ones_like(attn) * attn
    dist = dist / dist.sum(dim=1).view(-1, 1).repeat(1, attn.shape[1])
    return dist


def prior_distribution(N: int, M: int, std: float):
    """The prior distribution used to fit the Wasserstein distance to a particular pattern

    """
    # TODO: define the prior distribution as in the paper


def inverse_difference_moment_matrix(N: int, M: int):
    """The inverse different moment matrix for measuring local homogeneity of the transport matrix

    """
    # TODO: define the inversse difference moment


def sinkhorn(
    source_dist: torch.FloatTensor,
    target_dist: torch.FloatTensor,
    cost_matrix: torch.Tensor,
    reg: float,
    nit: int = 20,
):
    """The sinkhorn algorithm adapted for PyTorch from the
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
    assert(source_dist.shape[0] == cost_matrix.shape[0] and source_dist.shape[1] == cost_matrix.shape[1])
    assert(target_dist.shape[0] == cost_matrix.shape[0] and target_dist.shape[1] == cost_matrix.shape[2])

    # prepare the initial variables
    K = torch.exp(-cost_matrix / reg)
    Kp = (1 / source_dist).reshape(source_dist.shape[0], -1, 1) * K
    # initialize the u and v tensors
    u = torch.ones_like(source_dist)
    v = torch.ones_like(target_dist)

    istep = 0
    while istep < nit:
        # calculate K.T * u for each example in batch
        KTransposeU = K.transpose(1, 2).bmm(u.unsqueeze(2)).squeeze(2)
        # calculate the v_{i} tensor
        v = target_dist / KTransposeU
        # calculate the u_{i} tensor
        u = 1.0 / Kp.bmm(v.unsqueeze(2)).squeeze(2)
        # go to next step
        istep = istep + 1
    # calculate the transport matrix
    U = torch.diag_embed(u)
    V = torch.diag_embed(v)
    return U.bmm(K).bmm(V)
