"""Boosted trees for a two-sample density ratio on a GPU.

A PyTorch port of the boosting part of the R package BATTS (Awaya, Xu and Ma;
https://github.com/nawaya040/BATTS, version 0.3.0): ``BATTS::boots()`` and
``BATTS::eval_balance_weight()``. The fitted quantity is the balancing weight
w = sqrt(p0 / p1), where p0 is the density of group 0 and p1 that of group 1,
so the log density ratio log(p0 / p1) is 2 log(w).

The algorithm is the one of BATTS (same rescaling, cut grid, split rule, leaf
values, cross-validation and per-tree subsampling). What differs is how it is
computed: a tree is grown one level at a time for all of its nodes at once, and
many fits (the cross-validation folds and the final fit of one call, and the
calls of ``boots_many``) are grown together as one batch.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch

_F = torch.float64
_LARGE = 1e100  # loss given to a cut that is not allowed, as in BATTS
_TIE = 1e-9     # default tie_tolerance


# --------------------------------------------------------------------------------------------
# Random draws: counter-based, so that a seed gives the same folds and the same per-tree
# subsamples on a CPU and on a GPU, alone or inside any batch.
# --------------------------------------------------------------------------------------------

def _signed(x: int) -> int:
    """A Python integer wrapped to the range of a signed 64-bit integer."""
    return (x + (1 << 63)) % (1 << 64) - (1 << 63)


_GOLD = _signed(0x9E3779B97F4A7C15)
_MIX1 = _signed(0xBF58476D1CE4E5B9)
_MIX2 = _signed(0x94D049BB133111EB)


def _shift(z, k):
    """Logical right shift of signed 64-bit integers (tensor or Python integer)."""
    return (z >> k) & ((1 << (64 - k)) - 1)


def _mix(z):
    """The splitmix64 finaliser on a tensor of signed 64-bit integers."""
    z = (z ^ _shift(z, 30)) * _MIX1
    z = (z ^ _shift(z, 27)) * _MIX2
    return z ^ _shift(z, 31)


def _mix_int(z: int) -> int:
    """The same finaliser on a Python integer."""
    z = _signed(z)
    z = _signed((z ^ _shift(z, 30)) * _MIX1)
    z = _signed((z ^ _shift(z, 27)) * _MIX2)
    return _signed(z ^ _shift(z, 31))


def _model_seed(seed: int, model: int) -> int:
    return _mix_int(_mix_int(seed) + _GOLD * (model + 1))


def _draw_folds(group: np.ndarray, K: int, seed: int) -> np.ndarray:
    """Group-stratified fold labels 0..K-1, as BATTS::boots() draws them."""
    rng = np.random.default_rng(seed)
    labels = np.empty(group.size, dtype=np.int64)
    for value in (0, 1):
        members = np.flatnonzero(group == value)
        labels[members] = rng.permutation(np.arange(members.size) % K)
    return labels


def _subsample_size(fraction: float, m: int) -> int:
    return int(math.ceil(fraction * m - math.sqrt(np.finfo(float).eps)))


def _make_sampler(train, g1, seeds, position, fraction):
    """A function tree -> (models, n) mask of the observations the tree is built on.

    As in BATTS (and ada's bag.frac): one training observation from each group, and the rest
    uniformly without replacement from the other training observations.
    """
    if fraction == 1:
        return lambda t: train
    B, n = train.shape
    sizes = torch.tensor([_subsample_size(fraction, int(m)) for m in train.sum(1).tolist()],
                         device=train.device)
    if int(sizes.min()) < 2:
        raise ValueError("subsample_fraction leaves fewer than two training observations per tree")
    in0, in1 = train & ~g1[None, :], train & g1[None, :]
    rows = torch.arange(B, device=train.device)
    index = torch.arange(n, device=train.device).expand(B, n)
    never = 1 << 62

    def draw(t):
        base = _mix(seeds + _signed(_GOLD * (t + 1)))
        h = _mix(base[:, None] + position * _MIX2)
        key = _shift(h, 2)
        forced0 = key.masked_fill(~in0, -1).argmax(1)
        forced1 = key.masked_fill(~in1, -1).argmax(1)
        key = _shift(_mix(h + _GOLD), 2).masked_fill(~train, never)
        key[rows, forced0] = -1
        key[rows, forced1] = -1
        rank = torch.empty_like(key).scatter_(1, key.argsort(1), index)
        return rank < sizes[:, None]

    return draw


# --------------------------------------------------------------------------------------------
# The boosting loop
# --------------------------------------------------------------------------------------------

def _cut_candidates(n_bins: int) -> np.ndarray:
    """The candidate cut locations of BATTS::boots(), relative to a node: 1/n_bins, ..., 1 - 1/n_bins."""
    by, to = 1.0 / n_bins, 1.0 - 1.0 / n_bins
    count = int((to - by) / by + 1e-10)  # as R's seq(from, to, by)
    L = np.minimum(by + np.arange(count + 1) * by, to)
    y = L / (1.0 - L)
    return y / (1.0 + y)


@torch.no_grad()
def _boost(Xb, g1, train, valid, feat_ok, member_fn, keep, *, num_trees, max_resol, learn_rate, cuts,
           n_min, use_gradient, tie_tolerance=_TIE, progress=None, force=None):
    """Boost B models at once on the same n observations.

    Xb        (B, n, d) or (1, n, d): the data, rescaled to the unit box of each model.
    g1        (n,) bool: the observations of group 1.
    train     (B, n) bool: the training observations of each model.
    valid     (B, n) bool: the observations its validation loss is computed on.
    feat_ok   (B, d) bool: the variables each model may split on.
    member_fn tree -> (B, n) bool: the training observations the tree is built on.
    keep      indices of the models whose trees are returned.

    Returns the validation loss after each tree (B, num_trees), the normalising constant after
    each tree (B, num_trees), the trees of the kept models as arrays over the nodes of a complete
    binary tree (node h has children 2h + 1 and 2h + 2): the split variable (-1 in a leaf), the
    cut location relative to the node, the cut point, and the leaf value beta; the final weights
    w (B, n); and the largest violation per model, which is zero unless ``force`` is given.

    Ties. Cuts often have exactly the same loss (in the first tree the loss depends only on the
    group counts of the children; later, whenever two cuts divide the tree's sample in the same
    way), and which of them a floating-point minimum picks is then rounding noise: it differs
    between R, a CPU and a GPU, and between two runs on a GPU. So losses within ``tie_tolerance``
    of the minimum, relative to the loss of the node, count as equal and the first of them in
    the order of BATTS (cuts, then variables) is taken, which is what BATTS's rule gives in exact
    arithmetic; a node is split when that gains more than the same tolerance. With a tolerance
    of zero the rule is the floating-point minimum of BATTS.

    ``force`` is for the tests against R: (variable, cut index, active), the first two (B, trees,
    nodes) and the last (B, trees). Where active, the trees are not searched for but taken from
    there, and the violation says how far they are from what the search allows: the excess of
    the loss of a forced cut over the minimum, the excess of that loss over the loss of the node,
    and the gain of the best cut in a node forced to be a leaf, each relative to the loss of the
    node.
    """
    dev = train.device
    B, n = train.shape
    d = Xb.shape[2]
    Lc = cuts.numel()
    C = Lc + 1                      # cells per variable in a node
    D = max_resol
    M = 2 ** (D + 1) - 1            # nodes of a complete tree of depth D
    T = num_trees

    g1f = g1.to(_F)[None, :]
    g0f = 1.0 - g1f
    tr0, tr1 = train * g0f, train * g1f
    va0, va1 = valid * g0f, valid * g1f
    n0_learn, n1_learn = tr0.sum(1), tr1.sum(1)
    n0_valid, n1_valid = va0.sum(1).clamp_min(1.0), va1.sum(1).clamp_min(1.0)

    w = torch.ones(B, n, dtype=_F, device=dev)
    winv = torch.ones(B, n, dtype=_F, device=dev)
    prod_c = torch.ones(B, dtype=_F, device=dev)
    loss_curve = torch.zeros(B, T, dtype=_F, device=dev)
    c_curve = torch.ones(B, T, dtype=_F, device=dev)
    nk = keep.numel()
    kept_dim = torch.full((nk, T, M), -1, dtype=torch.long, device=dev)
    kept_loc = torch.full((nk, T, M), -1.0, dtype=_F, device=dev)
    kept_thr = torch.zeros((nk, T, M), dtype=_F, device=dev)
    kept_beta = torch.ones((nk, T, M), dtype=_F, device=dev)

    jj = torch.arange(d, device=dev)
    bb = torch.arange(B, device=dev)
    rule = torch.arange(Lc * d, device=dev)
    feat = feat_ok[:, None, :, None]
    # The cells of the root are the same for every tree
    flat_root = ((bb[:, None, None] * d + jj) * C
                 + (Xb / (Xb + (1.0 - Xb)) * C).long().clamp_(0, C - 1)).reshape(-1)
    root_lo = torch.zeros(B, 1, d, dtype=_F, device=dev)
    root_hi = torch.ones(B, 1, d, dtype=_F, device=dev)
    violation = torch.zeros(B, dtype=_F, device=dev)
    started = time.time()

    for t in range(T):
        member = member_fn(t)
        m0, m1 = member * g0f, member * g1f
        n0_tree, n1_tree = m0.sum(1, keepdim=True), m1.sum(1, keepdim=True)
        a0, a1 = m0 * winv, m1 * w          # the two terms of the balancing loss, on the tree's sample
        if use_gradient:
            grad = a1 / n1_tree - a0 / n0_tree
            size = (n0_tree + n1_tree).squeeze(1)
            node_loss = ((grad * grad).sum(1) / size - (grad.sum(1) / size) ** 2)[:, None]
            stats = torch.stack((grad, grad * grad, m0, m1), 2)
        else:
            total0, total1 = a0.sum(1)[:, None, None, None], a1.sum(1)[:, None, None, None]
            node_loss = torch.ones(B, 1, dtype=_F, device=dev)
            stats = torch.stack((a0, a1, m0, m1), 2)

        node = torch.zeros(B, n, dtype=torch.long, device=dev)   # node of every observation
        alive = None                                             # is that node on the current level
        lo, hi = root_lo, root_hi                                # boxes of the nodes of the level
        cur_dim = torch.full((B, M), -1, dtype=torch.long, device=dev)
        cur_loc = torch.full((B, M), -1.0, dtype=_F, device=dev)
        cur_thr = torch.zeros(B, M, dtype=_F, device=dev)

        for k in range(D):
            K = 1 << k
            first = K - 1
            # Sums over the cells of every variable in every node of the level
            if k == 0:
                local, flat, source = node, flat_root, stats
            else:
                local = (node - first).clamp_min_(0)
                which = local[:, :, None].expand(B, n, d)
                a = lo.gather(1, which)
                z = (Xb - a) / (hi.gather(1, which) - a)
                cell = (z / (z + (1.0 - z)) * C).long().clamp_(0, C - 1)
                flat = (((bb[:, None] * K + local)[:, :, None] * d + jj) * C + cell).reshape(-1)
                source = stats * alive[:, :, None]
                del which, a, z, cell
            hist = torch.zeros(B * K * d * C, 4, dtype=_F, device=dev)
            hist.index_add_(0, flat, source[:, :, None, :].expand(B, n, d, 4).reshape(-1, 4))
            hist = hist.view(B, K, d, C, 4)

            # Loss of every cut: cells up to the cut on the left, the others on the right
            left = hist.cumsum(3)[:, :, :, :Lc]
            right = hist.flip(3).cumsum(3).flip(3)[:, :, :, 1:]
            l0, l1, r0, r1 = left[..., 2], left[..., 3], right[..., 2], right[..., 3]
            allowed = (l0 >= n_min) & (r0 >= n_min) & (l1 >= n_min) & (r1 >= n_min) & feat
            if use_gradient:     # variance of the gradient within the children
                ln, rn = l0 + l1, r0 + r1
                child_l = left[..., 1] / ln - (left[..., 0] / ln) ** 2
                child_r = right[..., 1] / rn - (right[..., 0] / rn) ** 2
                both = ln + rn
                loss = ln / both * child_l + rn / both * child_r
            else:                # Hellinger affinity of the two weighted samples
                child_l = torch.sqrt((left[..., 0] / total0) * (left[..., 1] / total1))
                child_r = torch.sqrt((right[..., 0] / total0) * (right[..., 1] / total1))
                loss = child_l + child_r
            # First minimum over cuts, then variables (the order of BATTS); see "Ties" above
            loss = loss.masked_fill(~allowed, _LARGE).permute(0, 1, 3, 2).reshape(B, K, Lc * d)
            best = loss.amin(2)
            tie = tie_tolerance * node_loss.abs()
            pos = rule.expand(B, K, -1).masked_fill(loss > (best + tie)[..., None], Lc * d - 1).amin(2)
            n_node = hist[:, :, 0, :, 2:].sum((2, 3))
            can = (n_node > n_min) & (best < node_loss - tie)
            if force is not None:
                forced_var, forced_cut = force[0][:, t, first:first + K], force[1][:, t, first:first + K]
                splits = forced_var >= 0
                forced_pos = (forced_cut * d + forced_var).clamp_min(0)
                forced_loss = loss.gather(2, forced_pos[..., None]).squeeze(2)
                could = (n_node > n_min) & ~splits
                excess = torch.where(splits, torch.maximum(forced_loss - best, forced_loss - node_loss),
                                     torch.where(could, node_loss - best, torch.zeros_like(best)))
                excess = torch.where(n_node > 0, excess / node_loss.abs(), torch.zeros_like(best)).amax(1)
                excess = excess.clamp_min(0.0)
                violation = torch.where(force[2][:, t], torch.maximum(violation, excess), violation)
                pos, can = torch.where(splits, forced_pos, pos), splits

            jsel = pos % d
            loc = cuts[torch.div(pos, d, rounding_mode="floor")]
            lo_j = lo.gather(2, jsel[..., None]).squeeze(2)
            hi_j = hi.gather(2, jsel[..., None]).squeeze(2)
            thr = lo_j + (hi_j - lo_j) * loc
            cur_dim[:, first:first + K] = jsel.masked_fill(~can, -1)
            cur_loc[:, first:first + K] = loc.masked_fill(~can, -1.0)
            cur_thr[:, first:first + K] = thr

            # Send the observations of the split nodes to the children
            moves = can.gather(1, local)
            if alive is not None:
                moves &= alive
            x = Xb.expand(B, n, d).gather(2, jsel.gather(1, local)[..., None]).squeeze(2)
            go_right = ~(x < thr.gather(1, local))
            node = torch.where(moves, 2 * node + 1 + go_right.long(), node)
            alive = moves
            if k + 1 == D or not bool(can.any()):
                break
            pick = pos[..., None]
            child_l = child_l.permute(0, 1, 3, 2).reshape(B, K, Lc * d).gather(2, pick)
            child_r = child_r.permute(0, 1, 3, 2).reshape(B, K, Lc * d).gather(2, pick)
            node_loss = torch.cat((child_l, child_r), 2).reshape(B, 2 * K)
            lo = lo[:, :, None, :].repeat(1, 1, 2, 1)
            hi = hi[:, :, None, :].repeat(1, 1, 2, 1)
            hi[:, :, 0].scatter_(2, jsel[..., None], thr[..., None])
            lo[:, :, 1].scatter_(2, jsel[..., None], thr[..., None])
            lo, hi = lo.reshape(B, 2 * K, d), hi.reshape(B, 2 * K, d)

        # Leaf values from the tree's sample, then the weights of all observations
        s0 = torch.zeros(B, M, dtype=_F, device=dev).scatter_add_(1, node, a0)
        s1 = torch.zeros(B, M, dtype=_F, device=dev).scatter_add_(1, node, a1)
        beta = ((s0 / n0_tree) / (s1 / n1_tree)) ** learn_rate
        beta.masked_fill_((s0 == 0) | (s1 == 0), 1.0)
        step = beta.sqrt().gather(1, node)
        w = w * step
        winv = winv / step
        if not use_gradient:     # forward-stagewise: normalise on the whole training set
            c = torch.sqrt(((tr0 * winv).sum(1) / n0_learn) / ((tr1 * w).sum(1) / n1_learn))
            prod_c = prod_c * c
            w = c[:, None] * w
            winv = (1.0 / c)[:, None] * winv
        loss_curve[:, t] = (va0 * winv).sum(1) / n0_valid + (va1 * w).sum(1) / n1_valid
        c_curve[:, t] = prod_c
        if nk:
            kept_dim[:, t], kept_loc[:, t] = cur_dim[keep], cur_loc[keep]
            kept_thr[:, t], kept_beta[:, t] = cur_thr[keep], beta[keep]
        if progress is not None and ((t + 1) % 100 == 0 or t + 1 == T):
            progress(t + 1, T, time.time() - started)

    return loss_curve, c_curve, (kept_dim, kept_loc, kept_thr, kept_beta), w, violation


@torch.no_grad()
def _tree_product(tree_dim, tree_thr, tree_beta, Xs, budget=1 << 24):
    """The product over trees of sqrt(beta) of the leaf of each row of Xs (rescaled points)."""
    T, M = tree_dim.shape
    depth = int(round(math.log2(M + 1))) - 1
    out = torch.empty(Xs.shape[0], dtype=_F, device=Xs.device)
    chunk = max(1, budget // max(T, 1))
    for start in range(0, Xs.shape[0], chunk):
        X = Xs[start:start + chunk]
        m = X.shape[0]
        row = torch.arange(m, device=X.device)[None, :]
        node = torch.zeros(T, m, dtype=torch.long, device=X.device)
        for _ in range(depth):
            var = tree_dim.gather(1, node)
            go_right = ~(X[row, var.clamp_min(0)] < tree_thr.gather(1, node))
            node = torch.where(var >= 0, 2 * node + 1 + go_right.long(), node)
        out[start:start + m] = tree_beta.gather(1, node).sqrt().prod(0)
    return out


# --------------------------------------------------------------------------------------------
# Public interface
# --------------------------------------------------------------------------------------------

@dataclass
class Task:
    """One fit of ``boots_many``: ``boots(data[rows][:, cols], group_labels[rows])``.

    rows         indices or a boolean mask of the observations to fit on (None: all).
    cols         indices or a boolean mask of the variables the trees may use (None: all).
    seed         seed of the folds and subsamples (None: the seed of the call plus the task's place).
    fold_labels  one fold label per fitted observation, in the order of the rows (None: drawn).
    """
    rows: Optional[Sequence] = None
    cols: Optional[Sequence] = None
    seed: Optional[int] = None
    fold_labels: Optional[Sequence] = None


class BoostFit:
    """A fitted additive tree estimate of the balancing weight w = sqrt(p0 / p1).

    balance_weight  w at the fitted observations (BATTS: balance_weight_boosting_data).
    num_trees       number of trees (chosen by cross-validation when K_CV > 0).
    loss_cv         held-out balancing loss, folds by number of trees (BATTS: loss_CV_store), or None.
    c               normalising constant (1 for gradient boosting).
    omega           the fitted domain, variables by (lower, upper) (BATTS: Omega); for a variable
                    the fit does not use, (0, 1).
    rows, cols      the observations fitted on and the mask of the variables used.
    fold_labels     the cross-validation fold of each fitted observation, or None.
    seed            the seed of the folds and subsamples.
    """

    def __init__(self, tree_dim, tree_loc, tree_thr, tree_beta, c, omega, cols, rows=None, loss_cv=None,
                 fold_labels=None, seed=None, balance_weight=None):
        self.tree_dim, self.tree_loc, self.tree_thr, self.tree_beta = tree_dim, tree_loc, tree_thr, tree_beta
        self.c = float(c)
        self.omega = np.asarray(omega, dtype=float)
        self.cols = np.asarray(cols, dtype=bool)
        self.rows = rows
        self.loss_cv = loss_cv
        self.fold_labels = fold_labels
        self.seed = seed
        self.balance_weight = balance_weight
        self.violation = None   # only for the tests against R (see boots_many)

    @property
    def num_trees(self) -> int:
        return int(self.tree_dim.shape[0])

    @property
    def device(self):
        return self.tree_dim.device

    def to(self, device) -> "BoostFit":
        """Move the trees to another device (in place) and return the fit."""
        for name in ("tree_dim", "tree_loc", "tree_thr", "tree_beta"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def predict(self, points, outside: str = "error") -> np.ndarray:
        """The balancing weight w = sqrt(p0 / p1) at new points.

        ``points`` has the columns of the data the fit was made from (all of them, also when the
        fit uses only some). Points outside the fitted domain ``omega`` raise an error, as in
        BATTS, or give NaN with ``outside="nan"``.
        """
        if outside not in ("error", "nan"):
            raise ValueError('outside must be "error" or "nan"')
        P = np.asarray(points, dtype=float)
        if P.ndim != 2 or P.shape[0] < 1 or P.shape[1] != self.cols.size or not np.isfinite(P).all():
            raise ValueError("points must be a nonempty finite matrix with the columns of the training data")
        lo, hi = self.omega[:, 0], self.omega[:, 1]
        out = (((P < lo) | (P > hi)) & self.cols).any(1)
        if out.any() and outside == "error":
            raise ValueError("Some points are outside of the sample space")
        Xs = torch.as_tensor((P - lo) / (hi - lo), dtype=_F, device=self.device)
        w = self.c * _tree_product(self.tree_dim, self.tree_thr, self.tree_beta, Xs).cpu().numpy()
        w[out] = np.nan
        return w

    def log_ratio(self, points, outside: str = "error") -> np.ndarray:
        """log(p0 / p1) = 2 log(w) at new points."""
        return 2.0 * np.log(self.predict(points, outside))

    def tree_list(self) -> List[dict]:
        """The trees in the format of BATTS (``fit$tree_list``).

        One dict per tree with the nodes in depth-first order, left child first: ``d`` the split
        variable (0-based among the variables used; -1 in a leaf), ``l`` the cut location relative
        to the node (-1 in a leaf) and ``beta`` the leaf value (-1 in a split node).
        """
        dim, loc, beta = (a.cpu().numpy() for a in (self.tree_dim, self.tree_loc, self.tree_beta))
        used = np.cumsum(self.cols) - 1
        trees = []
        for t in range(dim.shape[0]):
            d_out, l_out, b_out, stack = [], [], [], [0]
            while stack:
                h = stack.pop()
                if dim[t, h] >= 0:
                    d_out.append(int(used[dim[t, h]])), l_out.append(float(loc[t, h])), b_out.append(-1.0)
                    stack += [2 * h + 2, 2 * h + 1]
                else:
                    d_out.append(-1), l_out.append(-1.0), b_out.append(float(beta[t, h]))
            trees.append({"d": d_out, "l": l_out, "beta": b_out})
        return trees


def _as_mask(selection, size, what):
    if selection is None:
        return np.ones(size, dtype=bool)
    selection = np.asarray(selection)
    if selection.dtype == bool:
        if selection.shape != (size,):
            raise ValueError(f"a boolean {what} mask must have length {size}")
        return selection.copy()
    mask = np.zeros(size, dtype=bool)
    mask[selection.astype(np.int64)] = True
    if mask.sum() != selection.size:
        raise ValueError(f"{what} must not repeat an index")
    return mask


def _check_settings(num_trees_max, K_CV, max_resol, learn_rate, n_bins, n_min_obs_per_node, subsample_fraction,
                    tie_tolerance):
    def whole(value):
        return isinstance(value, (int, np.integer)) or (isinstance(value, float) and value == int(value))
    if not whole(num_trees_max) or num_trees_max < 1:
        raise ValueError("num_trees_max must be a positive integer")
    if not whole(K_CV) or K_CV < 0 or K_CV == 1:
        raise ValueError("K_CV must be 0 or an integer between 2 and min(n0, n1)")
    if not whole(max_resol) or max_resol < 0:
        raise ValueError("max_resol must be a nonnegative integer")
    if not whole(n_bins) or n_bins < 2:
        raise ValueError("n_bins must be an integer of at least 2")
    if not learn_rate > 0:
        raise ValueError("learn_rate must be positive")
    if not n_min_obs_per_node >= 1:
        raise ValueError("n_min_obs_per_node must be at least 1")
    if not tie_tolerance >= 0:
        raise ValueError("tie_tolerance must be nonnegative")
    if isinstance(subsample_fraction, bool) or not np.isscalar(subsample_fraction) \
            or not 0 < subsample_fraction <= 1:
        raise ValueError("subsample_fraction must be a single number in (0, 1]")


def boots_many(data, group_labels, tasks, *, num_trees_max=1000, K_CV=5, max_resol=4, learn_rate=0.01,
               n_bins=32, n_min_obs_per_node=1, margin_scale=0.1, use_gradient=True,
               subsample_fraction=0.5, quiet=True, seed=None, device=None, tie_tolerance=_TIE,
               tasks_per_batch=None, _members=None, _force=None):
    """Fit many ``boots`` models on subsets of the rows and columns of one data set, as one batch.

    Task ``i`` gives the fit of ``boots(data[rows][:, cols], group_labels[rows], seed=task.seed)``
    with the settings of this call, and the results are the same as from those separate calls
    (up to rounding). All the models (every task's cross-validation folds and final fit) are
    boosted together, which is what makes small problems fast on a GPU.

    tasks            a sequence of ``Task`` (or of dicts with its fields).
    tasks_per_batch  how many tasks to boost at once (None: as many as the free memory allows).
    The other arguments are those of ``boots``. Returns one ``BoostFit`` per task.

    ``_members`` and ``_force`` are for the tests against R, with a single task. ``_members`` is
    a boolean array (models, trees, n) of the observations each tree is built on, used instead of
    drawn subsamples; the models are the cross-validation folds in the order of their labels,
    then the final fit. ``_force`` is the ``force`` of ``_boost`` (tensors on the device); the
    violations of the models are then in ``fit.violation``.
    """
    _check_settings(num_trees_max, K_CV, max_resol, learn_rate, n_bins, n_min_obs_per_node, subsample_fraction,
                    tie_tolerance)
    T, K, D = int(num_trees_max), int(K_CV), int(max_resol)
    X = np.asarray(data, dtype=float)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError("data must be a finite numeric matrix")
    n, d = X.shape
    g = np.asarray(group_labels)
    if g.shape != (n,) or not np.isin(g, (0, 1)).all():
        raise ValueError("group_labels must have one 0 or 1 per data row, with both groups present")
    g = g.astype(np.int64)
    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (1 << 62))

    # Rows, columns, folds and seed of every task
    prepared = []
    for i, task in enumerate(tasks):
        if isinstance(task, dict):
            task = Task(**task)
        rows, cols = _as_mask(task.rows, n, "rows"), _as_mask(task.cols, d, "cols")
        g_rows = g[rows]
        n0, n1 = int((g_rows == 0).sum()), int((g_rows == 1).sum())
        if n0 == 0 or n1 == 0:
            raise ValueError("group_labels must have one 0 or 1 per data row, with both groups present")
        if not cols.any():
            raise ValueError("a task must use at least one variable")
        if K > min(n0, n1):
            raise ValueError("K_CV must be 0 or an integer between 2 and min(n0, n1)")
        task_seed = int(task.seed) if task.seed is not None else seed + i
        folds = None
        if K > 0:
            if task.fold_labels is None:
                folds = _draw_folds(g_rows, K, task_seed)
            else:
                given = np.asarray(task.fold_labels)
                values, folds = np.unique(given, return_inverse=True)
                if given.shape != (g_rows.size,) or values.size != K:
                    raise ValueError("fold_labels must have one of K_CV labels per fitted observation")
        prepared.append((rows, cols, folds, task_seed))
    if not prepared:
        return []

    models = K + 1 if K > 0 else 1
    if tasks_per_batch is None:
        index = dev.index if dev.index is not None else torch.cuda.current_device() if dev.type == "cuda" else None
        free = torch.cuda.mem_get_info(index)[0] if dev.type == "cuda" else 4 << 30
        tasks_per_batch = max(1, int(0.5 * free / (112.0 * n * d * models)))
    X_dev = torch.as_tensor(X, dtype=_F, device=dev)
    g1 = torch.as_tensor(g == 1, device=dev)
    cuts = torch.as_tensor(_cut_candidates(int(n_bins)), dtype=_F, device=dev)
    fits = []

    for start in range(0, len(prepared), tasks_per_batch):
        batch = prepared[start:start + tasks_per_batch]
        G = len(batch)
        rows = torch.as_tensor(np.stack([b[0] for b in batch]), device=dev)       # (G, n)
        cols = torch.as_tensor(np.stack([b[1] for b in batch]), device=dev)       # (G, d)
        # Rescale to the unit box of each task: the range of its rows plus the margin
        lo = torch.zeros(G, d, dtype=_F, device=dev)
        hi = torch.ones(G, d, dtype=_F, device=dev)
        for i in range(G):
            Xi = X_dev[rows[i]]
            low, high = Xi.amin(0), Xi.amax(0)
            if margin_scale >= 0:
                constant = (low == high) & cols[i]
                if bool(constant.any()):
                    j = int(constant.nonzero()[0])
                    raise ValueError(f"Column {j} is constant; remove constant columns before fitting")
                margin = (high - low) * margin_scale
                lo[i], hi[i] = torch.where(cols[i], low - margin, lo[i]), torch.where(cols[i], high + margin, hi[i])
            elif bool((((low < 0) | (high > 1)) & cols[i]).any()):
                raise ValueError("with a negative margin_scale the data must lie in the unit box")
        Xs = (X_dev[None, :, :] - lo[:, None, :]) / (hi - lo)[:, None, :]          # (G, n, d)

        # The models of a task: one per fold (trained on the other folds), then the fit on all rows
        fold = torch.full((G, n), -1, dtype=torch.long, device=dev)
        position = torch.zeros(G, n, dtype=torch.long, device=dev)
        for i, b in enumerate(batch):
            position[i, rows[i]] = torch.arange(int(rows[i].sum()), device=dev)
            if K > 0:
                fold[i, rows[i]] = torch.as_tensor(b[2], device=dev)
        fold_of_model = torch.arange(models, device=dev)[None, :, None]            # the last is the final fit
        if K > 0:
            is_final = fold_of_model == K
            train = (rows[:, None, :] & ((fold[:, None, :] != fold_of_model) | is_final)).reshape(G * models, n)
            valid = (rows[:, None, :] & (fold[:, None, :] == fold_of_model) & ~is_final).reshape(G * models, n)
        else:
            train, valid = rows.clone(), torch.zeros_like(rows)
        seeds = torch.tensor([_model_seed(b[3], m) for b in batch for m in range(models)], device=dev)
        if _members is None:
            sampler = _make_sampler(train, g1, seeds, position.repeat_interleave(models, 0), subsample_fraction)
        else:
            given = torch.as_tensor(np.asarray(_members, dtype=bool), device=dev)
            if len(prepared) != 1 or given.shape != (models, T, n):
                raise ValueError("_members needs a single task and the shape (models, trees, n)")
            sampler = lambda t: given[:, t]
        keep = torch.arange(G, device=dev) * models + (models - 1)
        progress = None
        if not quiet:
            label = f"tasks {start + 1}-{start + G} of {len(prepared)}"
            progress = lambda t, total, seconds: print(f"{label}: tree {t}/{total}, {seconds:.1f} s", flush=True)
        loss_curve, c_curve, kept, _, violation = _boost(
            Xs if G == 1 else Xs.repeat_interleave(models, 0), g1, train, valid,
            cols.repeat_interleave(models, 0), sampler, keep, num_trees=T, max_resol=D,
            learn_rate=float(learn_rate), cuts=cuts, n_min=float(n_min_obs_per_node),
            use_gradient=bool(use_gradient), tie_tolerance=float(tie_tolerance), progress=progress, force=_force)

        loss_curve = loss_curve.view(G, models, T)
        lo_np, hi_np = lo.cpu().numpy(), hi.cpu().numpy()
        for i, b in enumerate(batch):
            loss_cv, opt = None, T
            if K > 0:   # the number of trees with the smallest mean held-out loss
                loss_cv = loss_curve[i, :K].cpu().numpy()
                opt = int(np.argmin(loss_cv.mean(0))) + 1
            trees = [a[i, :opt].clone() for a in kept]
            fit = BoostFit(*trees, c=c_curve[keep[i], opt - 1], omega=np.stack([lo_np[i], hi_np[i]], 1),
                           cols=b[1], rows=np.flatnonzero(b[0]), loss_cv=loss_cv, fold_labels=b[2], seed=b[3])
            fit.balance_weight = (fit.c * _tree_product(trees[0], trees[2], trees[3], Xs[i][rows[i]])).cpu().numpy()
            fit.violation = violation.view(G, models)[i].cpu().numpy()
            fits.append(fit)
        del Xs, loss_curve, c_curve, kept
    return fits


def boots(data, group_labels, num_trees_max=1000, K_CV=5, max_resol=4, learn_rate=0.01, n_bins=32,
          n_min_obs_per_node=1, margin_scale=0.1, use_gradient=True, subsample_fraction=0.5, quiet=True,
          *, seed=None, fold_labels=None, device=None, tie_tolerance=_TIE) -> BoostFit:
    """Boosting for a two-sample density ratio: ``BATTS::boots()`` on a GPU.

    Fits an additive tree estimate of the balancing weight w = sqrt(p0 / p1); the log density
    ratio log(p0 / p1) is ``2 * log(w)``. The arguments and defaults are those of BATTS 0.3.0
    (the settings of the paper's simulations).

    data                matrix, observations in rows and variables in columns; finite values, and
                        no constant column when ``margin_scale`` is nonnegative.
    group_labels        one 0 or 1 per row. Group 0 has density p0 (the numerator), group 1 p1.
    num_trees_max       maximum number of trees.
    K_CV                number of group-stratified cross-validation folds that choose the number of
                        trees (smallest mean held-out balancing loss), or 0 to fit ``num_trees_max``.
    max_resol           maximum tree depth.
    learn_rate          shrinkage factor.
    n_bins              number of cells of the cut grid in a node.
    n_min_obs_per_node  minimum count from each group in a child.
    margin_scale        fraction of each range added at both ends of the fitted domain; negative
                        keeps the unit box, and the data must lie in it.
    use_gradient        gradient boosting if True, forward-stagewise boosting if False.
    subsample_fraction  fraction of the training observations each tree is built on (1: all).
    quiet               no progress messages.
    seed                seed of the folds and the subsamples. The same seed gives the same draws on
                        a CPU and on a GPU. They are not the draws of R.
    fold_labels         cross-validation fold of each row, to use instead of drawn folds.
    device              ``"cuda"``, ``"cpu"``, ... (default: the GPU if there is one).
    tie_tolerance       cuts whose losses differ by less than this, relative to the loss of the
                        node, count as tied, and the first in the order of BATTS is taken (see
                        ``_boost``). 0 gives the floating-point minimum of BATTS, which rounding
                        noise decides: R's trees on a CPU in most cases, but on a GPU fits that
                        can change from run to run.
    """
    return boots_many(data, group_labels, [Task(seed=seed, fold_labels=fold_labels)], num_trees_max=num_trees_max,
                      K_CV=K_CV, max_resol=max_resol, learn_rate=learn_rate, n_bins=n_bins,
                      n_min_obs_per_node=n_min_obs_per_node, margin_scale=margin_scale, use_gradient=use_gradient,
                      subsample_fraction=subsample_fraction, quiet=quiet, seed=seed, device=device,
                      tie_tolerance=tie_tolerance)[0]


def eval_balance_weight(fit: BoostFit, eval_points, outside: str = "error") -> np.ndarray:
    """``BATTS::eval_balance_weight()``: the weights w = sqrt(p0 / p1) of a fit at new points."""
    return fit.predict(eval_points, outside)
