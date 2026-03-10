from __future__ import annotations

import itertools
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, _tree, export_text


# =============================================================================
# PART 1 — SYMBOLIC LOGIC ENGINE
# =============================================================================
# Implements first-order terms, atoms, clauses,
# substitutions, unification, standardize_apart, and depth-limited SLD proof.

class Term:
    """Abstract base for all logic terms."""


@dataclass(frozen=True)
class Var(Term):
    """A logic variable (e.g. X, Y, A)."""
    name: str
    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Const(Term):
    """A logic constant (e.g. sample identifier s42)."""
    name: str
    def __repr__(self) -> str:
        return self.name


Pred = Union[str, Var]
Subst = Dict[Var, Union[Term, str]]


@dataclass(frozen=True)
class Atom:
    """A predicate applied to a tuple of arguments: pred(arg1, arg2, ...)."""
    pred: Pred
    args: Tuple[Term, ...]

    def __repr__(self) -> str:
        return f"{self.pred}({', '.join(map(str, self.args))})"


@dataclass(frozen=True)
class Clause:
    """A Horn clause: head :- body[0], body[1], ..."""
    head: Atom
    body: Tuple[Atom, ...]

    def __repr__(self) -> str:
        if not self.body:
            return f"{self.head}."
        return f"{self.head} :- {', '.join(map(str, self.body))}."


# ---------------------------------------------------------------------------
# Unification
# ---------------------------------------------------------------------------

def _walk_term(t: Term, s: Subst) -> Term:
    while isinstance(t, Var) and t in s and isinstance(s[t], Term):
        t = s[t]
    return t


def _walk_pred(p: Pred, s: Subst) -> Pred:
    if isinstance(p, Var) and p in s and isinstance(s[p], str):
        return s[p]
    return p

def apply_atom(a: Atom, s: Subst) -> Atom:
    return Atom(
        _walk_pred(a.pred, s),
        tuple(_walk_term(arg, s) for arg in a.args),
    )


def unify_terms(t1: Term, t2: Term, s: Optional[Subst] = None) -> Optional[Subst]:
    if s is None:
        s = {}
    t1 = _walk_term(t1, s)
    t2 = _walk_term(t2, s)
    if t1 == t2:
        return s
    if isinstance(t1, Var):
        s[t1] = t2
        return s
    if isinstance(t2, Var):
        s[t2] = t1
        return s
    return None


def unify_preds(p1: Pred, p2: Pred, s: Optional[Subst] = None) -> Optional[Subst]:
    if s is None:
        s = {}
    p1 = _walk_pred(p1, s)
    p2 = _walk_pred(p2, s)
    if p1 == p2:
        return s
    if isinstance(p1, Var) and isinstance(p2, str):
        s[p1] = p2
        return s
    if isinstance(p2, Var) and isinstance(p1, str):
        s[p2] = p1
        return s
    return None

def unify_atoms(a1: Atom, a2: Atom, s: Optional[Subst] = None) -> Optional[Subst]:
    if s is None:
        s = {}
    s = unify_preds(a1.pred, a2.pred, s)
    if s is None:
        return None
    if len(a1.args) != len(a2.args):
        return None
    for x, y in zip(a1.args, a2.args):
        s = unify_terms(x, y, s)
        if s is None:
            return None
    return s


# ---------------------------------------------------------------------------
# Clause standardization
# ---------------------------------------------------------------------------

_clause_counter = itertools.count(1)


def standardize_apart(clause: Clause) -> Clause:
    """Rename all variables in a clause to fresh names."""
    mapping: Dict[Var, Var] = {}

    def rename(t: Term) -> Term:
        if isinstance(t, Var):
            if t not in mapping:
                mapping[t] = Var(f"{t.name}_{next(_clause_counter)}")
            return mapping[t]
        return t

    def rename_atom(a: Atom) -> Atom:
        return Atom(a.pred, tuple(rename(x) for x in a.args))

    return Clause(rename_atom(clause.head), tuple(rename_atom(b) for b in clause.body))


# ---------------------------------------------------------------------------
# Depth-limited SLD resolution
# ---------------------------------------------------------------------------

def prove_all(
    goals: Tuple[Atom, ...],
    program: List[Clause],
    s: Subst,
    depth: int,
) -> List[Subst]:
    if depth < 0:
        return []
    if not goals:
        return [s]
    first, rest = goals[0], goals[1:]
    first = apply_atom(first, s)
    solutions: List[Subst] = []
    for clause in program:
        c = standardize_apart(clause)
        s2 = unify_atoms(c.head, first, dict(s))
        if s2 is None:
            continue
        new_goals = tuple(apply_atom(b, s2) for b in c.body) + rest
        solutions.extend(prove_all(new_goals, program, s2, depth - 1))
    return solutions


def prove(goal: Atom, program: List[Clause], depth: int = 20) -> bool:
    """Return True iff goal is provable from program within depth."""
    return bool(prove_all((goal,), program, {}, depth))


# =============================================================================
# PART 2 — METARULES
# =============================================================================
# A MetaRule is a second-order clause template whose predicate positions
# contain Vars.  The learner instantiates these to produce first-order clauses.
#
# Correspondence to Formal Metagol pseudocode:
#   "choose metarule m such that mgu(head(m), e) = theta"
#   "for each literal l in body(m): choose predicate p in L"
# These two steps are implemented in generate_candidates().

@dataclass(frozen=True)
class MetaRule:
    """Second-order clause template with named predicate variables."""
    name: str
    head: Atom
    body: Tuple[Atom, ...]

    def __repr__(self) -> str:
        body_str = ", ".join(map(str, self.body))
        return f"[{self.name}] {self.head} :- {body_str}"


def identity_metarule() -> MetaRule:
    """P(A) :- Q(A)   — one condition suffices."""
    P, Q, A = Var("P"), Var("Q"), Var("A")
    return MetaRule("identity", Atom(P, (A,)), (Atom(Q, (A,)),))


def binary_conj_metarule() -> MetaRule:
    """P(A) :- Q(A), R(A)   — two conditions in conjunction."""
    P, Q, R, A = Var("P"), Var("Q"), Var("R"), Var("A")
    return MetaRule("binary_conj", Atom(P, (A,)), (Atom(Q, (A,)), Atom(R, (A,))))


def ternary_conj_metarule() -> MetaRule:
    """P(A) :- Q(A), R(A), S(A)   — three conditions in conjunction."""
    P, Q, R, S, A = Var("P"), Var("Q"), Var("R"), Var("S"), Var("A")
    return MetaRule(
        "ternary_conj",
        Atom(P, (A,)),
        (Atom(Q, (A,)), Atom(R, (A,)), Atom(S, (A,))),
    )


def _pred_vars_in(m: MetaRule) -> List[Var]:
    """Return predicate-position Vars in order of appearance."""
    seen: List[Var] = []
    for atom in (m.head,) + m.body:
        if isinstance(atom.pred, Var) and atom.pred not in seen:
            seen.append(atom.pred)
    return seen


# ---------------------------------------------------------------------------
# Clause normalisation
# ---------------------------------------------------------------------------

def _generalize_clause(clause: Clause) -> Clause:
    """Replace every Const with a fresh Var (Generalize step in pseudocode)."""
    mapping: Dict[Const, Var] = {}
    idx = 0

    def gen(t: Term) -> Term:
        nonlocal idx
        if isinstance(t, Const):
            if t not in mapping:
                mapping[t] = Var(f"_G{idx}")
                idx += 1
            return mapping[t]
        return t

    def gen_atom(a: Atom) -> Atom:
        return Atom(a.pred, tuple(gen(x) for x in a.args))

    return Clause(gen_atom(clause.head), tuple(gen_atom(b) for b in clause.body))


_VAR_NAMES = "XYZUVWABCDEFGHIJKLMNOPQRST"


def _normalize_vars(clause: Clause) -> Clause:
    """Rename all Vars to canonical names X, Y, Z, ... for readability."""
    mapping: Dict[Var, Var] = {}
    idx = 0

    def norm(t: Term) -> Term:
        nonlocal idx
        if isinstance(t, Var):
            if t not in mapping:
                name = _VAR_NAMES[idx] if idx < len(_VAR_NAMES) else f"V{idx + 1}"
                mapping[t] = Var(name)
                idx += 1
            return mapping[t]
        return t

    def norm_atom(a: Atom) -> Atom:
        return Atom(a.pred, tuple(norm(x) for x in a.args))

    return Clause(norm_atom(clause.head), tuple(norm_atom(b) for b in clause.body))


def _canonicalize(clause: Clause) -> Clause:
    """Generalize constants, then normalize variable names."""
    return _normalize_vars(_generalize_clause(clause))


def _clause_key(clause: Clause) -> str:
    """Canonical string representation for deduplication."""
    return repr(_canonicalize(clause))


# =============================================================================
# PART 3 — FORMAL METAGOL-STYLE LEARNER
# =============================================================================
#
# This section implements the core Metagol algorithm as described in the
# formal pseudocode:
#
#   Metagol(B, E+, E-, C):
#     L := { all predicate heads in B }
#     for d = 1 to Dmax:
#       H := {}
#       for e in E+:
#         if (H, B) |= e then continue
#         [choose metarule + predicates] -> add clause to H
#       if all E+ provable and no E- provable: return H
#
# Key design decisions and how they relate to the pseudocode:
#
# (1) ITERATIVE DEEPENING (faithful):
#     We loop d from 1 to Dmax.  At each d we start fresh with H = {}.
#     We only return H if it is COMPLETE: all E+ covered, no E- covered.
#
# (2) RECURSIVE BACKTRACKING SEARCH (faithful):
#     The "choose" operations in the pseudocode imply nondeterminism.
#     We implement this as DFS with backtracking: _backtrack_search() picks
#     the first uncovered positive, tries all candidate clauses, recurses,
#     and backtracks if the subtree fails.
#
# (3) CANDIDATE GENERATION (faithful):
#     generate_candidates() implements the two "choose" steps:
#       - "choose metarule m such that mgu(head(m), e) = theta"
#       - "for each literal l in body(m): choose predicate p in L"
#     followed by the Generalize() step.
#
# (4) COVERAGE PRECOMPUTATION (practical optimization, documented):
#     The formal pseudocode tests coverage via proof search.  Because our
#     BK is purely propositional (unary ground facts), coverage can be
#     checked by frozenset subset tests.  This is O(k) per (clause, sample)
#     instead of O(n_clauses * depth) for SLD proof, making the search
#     tractable.  The prover is still used at test time.
#
# (5) TWO-PHASE LEARNING (documented extension):
#     Metagol formally requires complete coverage.  In our pipeline the BK
#     derives from an imperfect surrogate, so complete coverage may be
#     impossible for some classes (surrogate infidelity samples cannot be
#     covered without violating negative constraints).  We handle this as:
#       - Phase 1 (formal): strict iterative deepening, return H only if complete.
#       - Phase 2 (fallback): greedy partial coverage, with honest reporting.
#
# (6) DOCUMENTED SIMPLIFICATIONS:
#     - No predicate invention: all body predicates drawn from BK.
#     - No recursive metarules: not needed for threshold-based classification.
#     - Duplicate clause prevention by canonical key (same clause cannot
#       appear twice in one hypothesis).


# ---------------------------------------------------------------------------
# Step 3a: Candidate generation with precomputed coverage
# ---------------------------------------------------------------------------

# A CandidateRecord stores a clause together with the indices of the
# positive and negative examples it covers (as frozensets for O(1) ops).
CandidateRecord = Tuple[Clause, FrozenSet[int], FrozenSet[int]]


def generate_candidates(
    target_pred: str,
    metarules: List[MetaRule],
    predicate_symbols: List[str],
    pos_sample_names: List[str],
    neg_sample_names: List[str],
    sample_facts: Dict[str, FrozenSet[str]],
) -> List[CandidateRecord]:
    """
    Generate all candidate clauses derivable by instantiating each metarule
    with the given target predicate and background predicates.

    This implements the two "choose" steps from the Formal Metagol pseudocode:
        "choose metarule m s.t. mgu(head(m), e) = theta"
        "for each literal l in body(m): choose predicate p in L"
    followed by Generalize().

    Coverage is precomputed using propositional frozenset lookup (see note above).

    Parameters
    ----------
    target_pred      : The predicate being learned (e.g. "is_versicolor")
    metarules        : List of MetaRule templates
    predicate_symbols: L in the pseudocode — all predicate names from BK
    pos_sample_names : Sample-identifier strings for positive examples
    neg_sample_names : Sample-identifier strings for negative examples
    sample_facts     : Maps each sample name to its set of true predicates

    Returns
    -------
    List of (clause, pos_covered, neg_covered) triples.
    Clauses are sorted by: most positives covered first, fewest negatives first.
    """
    # Use a template atom Atom(target_pred, (Var("A"),)) to unify with metarule heads.
    # This fixes the head predicate while leaving the argument variable free.
    template = Atom(target_pred, (Var("_A"),))

    seen_keys: Set[str] = set()
    records: List[CandidateRecord] = []

    for mr in metarules:
        # "choose metarule m s.t. mgu(head(m), e) = theta"
        theta = unify_atoms(mr.head, template, {})
        if theta is None:
            continue  # metarule head doesn't unify with target

        pred_vars = _pred_vars_in(mr)

        # Build choices for each predicate-variable slot.
        # The head predicate is already bound by theta; body slots range over L.
        choices: List[List[str]] = []
        for pv in pred_vars:
            if pv in theta and isinstance(theta[pv], str):
                choices.append([theta[pv]])   # already fixed (head predicate)
            else:
                choices.append(predicate_symbols)  # "choose predicate p in L"

        # Enumerate all combinations of predicate assignments
        for assignment in itertools.product(*choices):
            sigma = dict(theta)
            valid = True
            for pv, sym in zip(pred_vars, assignment):
                if not isinstance(sym, str):
                    valid = False
                    break
                sigma[pv] = sym
            if not valid:
                continue

            cand_head = apply_atom(mr.head, sigma)
            cand_body = tuple(apply_atom(b, sigma) for b in mr.body)

            if not isinstance(cand_head.pred, str):
                continue
            if any(not isinstance(b.pred, str) for b in cand_body):
                continue

            # Generalize() + normalize to canonical form
            candidate = _canonicalize(Clause(cand_head, cand_body))
            key = _clause_key(candidate)

            if key in seen_keys:
                continue
            seen_keys.add(key)

            # Precompute coverage via propositional frozenset lookup.
            # A clause covers sample s iff every body predicate is in sample_facts[s].
            body_preds: Set[str] = {
                b.pred for b in candidate.body if isinstance(b.pred, str)
            }
            pos_covered = frozenset(
                i for i, s in enumerate(pos_sample_names)
                if body_preds.issubset(sample_facts.get(s, frozenset()))
            )
            neg_covered = frozenset(
                i for i, s in enumerate(neg_sample_names)
                if body_preds.issubset(sample_facts.get(s, frozenset()))
            )

            records.append((candidate, pos_covered, neg_covered))

    # Sort: prefer clauses that cover more positives and fewer negatives.
    # This heuristic speeds up the search by trying the most useful clauses first.
    records.sort(key=lambda r: (-len(r[1]), len(r[2])))
    return records


# ---------------------------------------------------------------------------
# Step 3b: Iterative deepening + backtracking search
# ---------------------------------------------------------------------------

def _backtrack_search(
    candidates: List[CandidateRecord],
    all_pos_indices: FrozenSet[int],
    covered_pos: FrozenSet[int],
    covered_neg: FrozenSet[int],
    hypothesis_keys: FrozenSet[str],
    depth_limit: int,
) -> Optional[List[Clause]]:
    """
    Recursive DFS over the hypothesis space.

    Corresponds to the inner loop of the Formal Metagol pseudocode:

        for e in E+:
            if (H, B) |= e then continue
            choose metarule m, instantiate, add Generalize(m) to H

    The "choose" is implemented here as deterministic enumeration with
    backtracking: we try each candidate in order, recurse, and backtrack
    (by returning None) if the subtree fails.

    Parameters
    ----------
    candidates       : Precomputed (clause, pos_cov, neg_cov) triples
    all_pos_indices  : Frozenset of all positive example indices (0..n-1)
    covered_pos      : Positive indices already covered by current hypothesis H
    covered_neg      : Negative indices covered by H (must remain empty)
    hypothesis_keys  : Canonical keys of clauses already in H (no duplicates)
    depth_limit      : Maximum number of clauses allowed in this search (= d)

    Returns
    -------
    List of clauses to append to H to complete it, or None if no completion
    exists within the depth limit.
    """
    uncovered = all_pos_indices - covered_pos

    # Base case: all positives are covered
    if not uncovered:
        if not covered_neg:
            return []   # SUCCESS: no negatives violated, hypothesis is complete
        return None     # FAILURE: negatives were covered, this path is invalid

    # Depth bound: hypothesis cannot grow beyond d clauses
    if len(hypothesis_keys) >= depth_limit:
        return None

    # "for e in E+: if not (H,B) |= e" — pick the first uncovered positive
    # Using min() gives a deterministic choice consistent with processing
    # examples in the order they appear in E+.
    e = min(uncovered)

    # Enumerate candidate clauses ("choose metarule m, choose predicates from L")
    for candidate, pos_cov, neg_cov in candidates:
        key = _clause_key(candidate)

        # Skip if this clause is already in the current hypothesis
        if key in hypothesis_keys:
            continue

        # Skip if it does not cover the target uncovered example e
        if e not in pos_cov:
            continue

        # Consistency pruning: if adding this clause would cover any negative,
        # prune this branch immediately (Formal Metagol's constraint check)
        new_neg_covered = covered_neg | neg_cov
        if new_neg_covered:
            continue

        # Recurse with H extended by this candidate
        tail = _backtrack_search(
            candidates,
            all_pos_indices,
            covered_pos | pos_cov,
            new_neg_covered,
            hypothesis_keys | {key},
            depth_limit,
        )

        if tail is not None:
            # Backtracking succeeds: this clause is part of the solution
            return [candidate] + tail

        # Backtrack: this candidate did not lead to a complete hypothesis,
        # so we try the next one in the enumeration.

    return None  # No candidate worked: backtrack to the caller


def _greedy_partial_fallback(
    candidates: List[CandidateRecord],
    all_pos_indices: FrozenSet[int],
) -> Optional[List[Clause]]:
    """
    Greedy fallback when no complete hypothesis exists within the depth bound.

    Iteratively adds the clause that covers the most new positive examples
    without covering any negative.  This is NOT part of formal Metagol; it
    is an acknowledged extension for the case where the BK cannot support
    complete coverage (e.g., due to surrogate infidelity).

    Returns the best partial hypothesis found, or None if no consistent
    clause exists at all.
    """
    covered_pos: FrozenSet[int] = frozenset()
    hypothesis: List[Clause] = []
    used_keys: Set[str] = set()

    # Candidates are already sorted by coverage quality
    for candidate, pos_cov, neg_cov in candidates:
        key = _clause_key(candidate)
        if key in used_keys:
            continue
        new_pos = pos_cov - covered_pos
        if not new_pos:
            continue   # adds nothing new
        if neg_cov:
            continue   # consistency constraint still applies

        hypothesis.append(candidate)
        used_keys.add(key)
        covered_pos = covered_pos | pos_cov

        if covered_pos == all_pos_indices:
            break   # full coverage achieved

    return hypothesis if hypothesis else None


def metagol(
    background: List[Clause],
    positives: List[Atom],
    negatives: List[Atom],
    metarules: List[MetaRule],
    dmax: int = 4,
) -> Tuple[Optional[List[Clause]], bool]:
    """
    Metagol-style inductive learning with iterative deepening and backtracking.

    Implements the Formal Metagol pseudocode:

        Metagol(B, E+, E-, C):
            L := { all predicate heads in B }
            for d = 1 to Dmax:
                H := {}
                [recursive backtracking search for H of size <= d]
                if B ∪ H |= all E+ and B ∪ H ⊬ any E-:
                    return H

    Parameters
    ----------
    background  : Background knowledge (ground facts from symbolic BK)
    positives   : Positive examples E+ (atoms to be proved)
    negatives   : Negative examples E- (atoms that must NOT be proved)
    metarules   : Second-order templates C
    dmax        : Maximum hypothesis size (Dmax in pseudocode)

    Returns
    -------
    (hypothesis, is_complete) where:
      - hypothesis     : List of learned clauses (or None if nothing found)
      - is_complete    : True if all positives are covered (formal Metagol result);
                         False if only partial coverage was achieved via fallback.

    Documented simplifications:
      - No predicate invention (all body predicates from BK).
      - No recursive metarules.
      - Coverage precomputed by frozenset ops instead of proof search
        (valid because BK is propositional unary ground facts).
      - Two-phase: strict search first, greedy fallback second.
    """
    if not positives:
        return [], True

    # L := { all predicate heads in B }
    predicate_symbols: List[str] = sorted(
        {c.head.pred for c in background if isinstance(c.head.pred, str)}
    )

    # Extract sample names from example atoms
    target_pred = positives[0].pred
    pos_names = [a.args[0].name for a in positives]
    neg_names = [a.args[0].name for a in negatives]

    # Build propositional sample-facts index for fast coverage checks
    sample_facts = _build_sample_facts(background)

    # Precompute all candidate clauses and their coverage (once, before search)
    candidates = generate_candidates(
        target_pred, metarules, predicate_symbols,
        pos_names, neg_names, sample_facts,
    )

    all_pos_indices = frozenset(range(len(positives)))

    # -------------------------------------------------------------------------
    # Phase 1 — Formal Metagol: iterative deepening with backtracking
    # -------------------------------------------------------------------------
    # "for d = 1 to Dmax: H := {}; [search]; if complete: return H"
    for d in range(1, dmax + 1):
        result = _backtrack_search(
            candidates,
            all_pos_indices,
            frozenset(),   # initially nothing covered
            frozenset(),   # initially no negatives covered
            frozenset(),   # empty hypothesis
            d,
        )
        if result is not None:
            return result, True   # COMPLETE hypothesis found

    # -------------------------------------------------------------------------
    # Phase 2 — Greedy fallback for partial coverage
    # -------------------------------------------------------------------------
    # This phase is NOT part of formal Metagol.  It handles the case where
    # complete coverage is impossible because some positive examples correspond
    # to surrogate infidelity cases: the black-box predicted class c for sample x,
    # but the surrogate classified x differently, so no combination of
    # surrogate-threshold predicates can cover x as class c without error.
    partial = _greedy_partial_fallback(candidates, all_pos_indices)
    return partial, False   # PARTIAL hypothesis (not all positives covered)


def _build_sample_facts(background: List[Clause]) -> Dict[str, FrozenSet[str]]:
    """
    Build an index: sample_name -> frozenset of true predicate names.

    Used for O(k) propositional coverage checks during learning.
    The prover (prove()) is still used at prediction time.
    """
    index: Dict[str, Set[str]] = defaultdict(set)
    for clause in background:
        if not clause.body:   # ground fact
            pred = clause.head.pred
            if isinstance(pred, str) and clause.head.args:
                arg = clause.head.args[0]
                if isinstance(arg, Const):
                    index[arg.name].add(pred)
    return {name: frozenset(preds) for name, preds in index.items()}


# =============================================================================
# PART 4 — SYMBOLIC BACKGROUND KNOWLEDGE EXTRACTION
# =============================================================================
# Unchanged from v1.

def _sanitize(name: str) -> str:
    return (
        name.replace(" (cm)", "")
            .replace(" ", "_")
            .replace("/", "_")
            .replace("-", "_")
    )


def _pred_name(feature: str, op: str, threshold: float) -> str:
    thr_str = f"{threshold:.2f}".replace(".", "_").replace("-", "m")
    op_tag = "le" if op == "<=" else "gt"
    return f"{_sanitize(feature)}_{op_tag}_{thr_str}"


def extract_thresholds_from_tree(
    tree: DecisionTreeClassifier,
    feature_names: List[str],
) -> Dict[str, Set[float]]:
    tree_ = tree.tree_
    thresholds: Dict[str, Set[float]] = defaultdict(set)
    for node in range(tree_.node_count):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            fname = feature_names[tree_.feature[node]]
            thresholds[fname].add(float(tree_.threshold[node]))
    return dict(thresholds)


def build_background_knowledge(
    X: pd.DataFrame,
    thresholds: Dict[str, Set[float]],
) -> Tuple[List[Clause], List[Const]]:
    """
    Convert a dataset into symbolic background knowledge.

    For every sample x_i and every surrogate threshold, we assert exactly one of:
        feature_le_threshold(s_i).   if x_i[feature] <= threshold
        feature_gt_threshold(s_i).   otherwise
    """
    background: List[Clause] = []
    sample_ids: List[Const] = []

    for i, row in X.iterrows():
        sample = Const(f"s{i}")
        sample_ids.append(sample)

        for feat, thr_set in thresholds.items():
            value = float(row[feat])
            for thr in sorted(thr_set):
                if value <= thr:
                    pred = _pred_name(feat, "<=", thr)
                else:
                    pred = _pred_name(feat, ">", thr)
                background.append(Clause(Atom(pred, (sample,)), ()))

    return background, sample_ids


def build_examples(
    sample_ids: List[Const],
    labels: List[int],
    target_class: int,
    target_pred: str,
) -> Tuple[List[Atom], List[Atom]]:
    positives: List[Atom] = []
    negatives: List[Atom] = []
    for sample, label in zip(sample_ids, labels):
        atom = Atom(target_pred, (sample,))
        if label == target_class:
            positives.append(atom)
        else:
            negatives.append(atom)
    return positives, negatives


# =============================================================================
# PART 5 — SYMBOLIC PREDICTION & EVALUATION
# =============================================================================

def _symbolic_facts_for_row(
    row: pd.Series,
    thresholds: Dict[str, Set[float]],
    sample_name: str = "query",
) -> List[Clause]:
    sample = Const(sample_name)
    facts: List[Clause] = []
    for feat, thr_set in thresholds.items():
        value = float(row[feat])
        for thr in sorted(thr_set):
            if value <= thr:
                pred = _pred_name(feat, "<=", thr)
            else:
                pred = _pred_name(feat, ">", thr)
            facts.append(Clause(Atom(pred, (sample,)), ()))
    return facts


def predict_one(
    row: pd.Series,
    thresholds: Dict[str, Set[float]],
    theories: Dict[str, Optional[List[Clause]]],
    class_names: List[str],
) -> Optional[str]:
    query_facts = _symbolic_facts_for_row(row, thresholds)
    hits: List[str] = []
    for class_name in class_names:
        pred = f"is_{class_name}"
        theory = theories.get(pred)
        if theory is None:
            continue
        if prove(Atom(pred, (Const("query"),)), query_facts + theory):
            hits.append(class_name)
    if len(hits) == 1:
        return hits[0]
    if len(hits) == 0:
        return None
    return hits[0]


def evaluate(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    blackbox,
    surrogate,
    thresholds: Dict[str, Set[float]],
    theories: Dict[str, Optional[List[Clause]]],
    class_names: List[str],
) -> Dict:
    bb_preds = blackbox.predict(X_test)
    surr_preds = surrogate.predict(X_test)
    true_names = [class_names[int(v)] for v in y_test]
    bb_names = [class_names[int(v)] for v in bb_preds]
    surr_names = [class_names[int(v)] for v in surr_preds]

    logic_preds: List[Optional[str]] = [
        predict_one(X_test.iloc[i], thresholds, theories, class_names)
        for i in range(len(X_test))
    ]

    covered = [i for i, p in enumerate(logic_preds) if p is not None]
    n_test = len(X_test)

    bb_accuracy = float(np.mean([bb_names[i] == true_names[i] for i in range(n_test)]))
    surr_fidelity = float(np.mean([surr_names[i] == bb_names[i] for i in range(n_test)]))

    if covered:
        logic_coverage = len(covered) / n_test
        logic_fidelity = float(np.mean([logic_preds[i] == bb_names[i] for i in covered]))
        logic_accuracy = float(np.mean([logic_preds[i] == true_names[i] for i in covered]))
    else:
        logic_coverage = logic_fidelity = logic_accuracy = 0.0

    return {
        "bb_accuracy": bb_accuracy,
        "surr_fidelity": surr_fidelity,
        "logic_coverage": logic_coverage,
        "logic_fidelity": logic_fidelity,
        "logic_accuracy": logic_accuracy,
        "logic_preds": logic_preds,
        "true_names": true_names,
        "bb_names": bb_names,
        "surr_names": surr_names,
        "n_covered": len(covered),
        "n_test": n_test,
    }


# =============================================================================
# PART 6 — FULL PIPELINE
# =============================================================================

def run_pipeline(
    dmax: int = 4,
    surrogate_max_depth: int = 3,
    test_size: float = 0.4,
    random_state: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Execute the complete hybrid interpretable learning pipeline.

    Steps:
      1  Load data
      2  Train black-box (RandomForest)
      3  Train surrogate tree (mimics black-box)
      4  Extract symbolic background knowledge from surrogate
      5  Build positive/negative examples (one-vs-rest per class)
      6  Run Formal Metagol-style learner for each class
      7  Evaluate: bb_accuracy, surr_fidelity, logic_coverage, fidelity, accuracy
    """

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    log("\n" + "=" * 65)
    log("HYBRID INTERPRETABLE LEARNING PIPELINE  (v2 — Formal Metagol)")
    log("=" * 65)

    # -------------------------------------------------------------------------
    # Step 1 — Data
    # -------------------------------------------------------------------------
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)
    class_names: List[str] = list(iris.target_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    log(f"\n[Step 1] Dataset: Iris  —  {len(X_train)} train / {len(X_test)} test")
    log(f"         Classes: {class_names}")

    # -------------------------------------------------------------------------
    # Step 2 — Black-box
    # -------------------------------------------------------------------------
    blackbox = RandomForestClassifier(n_estimators=100, random_state=random_state)
    blackbox.fit(X_train, y_train)
    log(f"\n[Step 2] Black-box (RandomForest, 100 trees)")
    log(f"         Train acc: {blackbox.score(X_train, y_train):.3f}"
        f"   Test acc: {blackbox.score(X_test, y_test):.3f}")

    # -------------------------------------------------------------------------
    # Step 3 — Surrogate tree
    # -------------------------------------------------------------------------
    y_train_bb = pd.Series(blackbox.predict(X_train), index=X_train.index)
    surrogate = DecisionTreeClassifier(
        max_depth=surrogate_max_depth, random_state=random_state
    )
    surrogate.fit(X_train, y_train_bb)

    log(f"\n[Step 3] Surrogate (DecisionTree, max_depth={surrogate_max_depth})")
    log(f"         Fidelity train: {surrogate.score(X_train, y_train_bb):.3f}"
        f"   Fidelity test: {surrogate.score(X_test, blackbox.predict(X_test)):.3f}")
    log(f"\n         Surrogate tree:")
    for line in export_text(surrogate, feature_names=list(X_train.columns)).strip().split("\n"):
        log(f"           {line}")

    # -------------------------------------------------------------------------
    # Step 4 — Symbolic BK
    # -------------------------------------------------------------------------
    thresholds = extract_thresholds_from_tree(surrogate, list(X_train.columns))
    background, sample_ids = build_background_knowledge(X_train, thresholds)
    labels = [int(y_train_bb.loc[idx]) for idx in X_train.index]

    n_preds = len({c.head.pred for c in background})
    log(f"\n[Step 4] Symbolic Background Knowledge")
    log(f"         Surrogate split features: {[f for f, v in thresholds.items() if v]}")
    log(f"         Threshold predicates: {n_preds}   Ground facts: {len(background)}")

    # -------------------------------------------------------------------------
    # Step 5 — Metarules
    # -------------------------------------------------------------------------
    metarules = [
        identity_metarule(),        # P(A) :- Q(A)
        binary_conj_metarule(),     # P(A) :- Q(A), R(A)
        ternary_conj_metarule(),    # P(A) :- Q(A), R(A), S(A)
    ]
    log(f"\n[Step 5] Metarules: {[m.name for m in metarules]}")

    # -------------------------------------------------------------------------
    # Step 6 — Formal Metagol-style learning
    # -------------------------------------------------------------------------
    log(f"\n[Step 6] Formal Metagol-style learning  (dmax={dmax}, one-vs-rest)")
    log(f"         Algorithm: iterative deepening (d=1..{dmax}) + backtracking DFS")

    theories: Dict[str, Optional[List[Clause]]] = {}
    completeness: Dict[str, bool] = {}

    for class_id, class_name in enumerate(class_names):
        target_pred = f"is_{class_name}"
        positives, negatives = build_examples(
            sample_ids, labels, class_id, target_pred
        )
        log(f"\n  ── {target_pred}:  {len(positives)} E+  /  {len(negatives)} E─ ──")

        t0 = time.time()
        H, complete = metagol(background, positives, negatives, metarules, dmax)
        elapsed = time.time() - t0

        theories[target_pred] = H
        completeness[target_pred] = complete

        status = "COMPLETE" if complete else "PARTIAL (fallback)"
        log(f"  Search: {elapsed:.2f}s   Status: {status}")

        if H is None:
            log("  Result: No hypothesis found.")
        else:
            log(f"  Result: {len(H)} clause(s)")
            for cl in H:
                log(f"    {cl}")

    # -------------------------------------------------------------------------
    # Step 7 — Evaluation
    # -------------------------------------------------------------------------
    log(f"\n[Step 7] Evaluation")
    results = evaluate(
        X_test, y_test, blackbox, surrogate, thresholds, theories, class_names
    )

    log(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │  EVALUATION SUMMARY                                         │
  ├─────────────────────────────────────────────────────────────┤
  │  Black-box test accuracy          : {results['bb_accuracy']:.3f}                  │
  │  Surrogate fidelity (test)        : {results['surr_fidelity']:.3f}                  │
  ├─────────────────────────────────────────────────────────────┤
  │  Logic rule coverage              : {results['logic_coverage']:.3f}                  │
  │    ({results['n_covered']}/{results['n_test']} test samples covered by a logic rule)         │
  │  Logic fidelity vs. black-box     : {results['logic_fidelity']:.3f} (on covered)       │
  │  Logic accuracy vs. true labels   : {results['logic_accuracy']:.3f} (on covered)       │
  └─────────────────────────────────────────────────────────────┘
""")

    log("  Completeness per class:")
    for target, H in theories.items():
        c = completeness.get(target, False)
        n_clauses = len(H) if H else 0
        log(f"    {target}: {n_clauses} clause(s)  [{('COMPLETE' if c else 'partial')}]")

    log("\n  Sample predictions (first 8 test examples):")
    log(f"  {'True':>12}  {'BlackBox':>10}  {'Logic':>12}")
    log(f"  {'-'*12}  {'-'*10}  {'-'*12}")
    for i in range(min(8, len(X_test))):
        lp = results["logic_preds"][i]
        log(f"  {results['true_names'][i]:>12}  {results['bb_names'][i]:>10}  {str(lp):>12}")

    return {
        "blackbox": blackbox,
        "surrogate": surrogate,
        "thresholds": thresholds,
        "background": background,
        "theories": theories,
        "completeness": completeness,
        "metarules": metarules,
        "class_names": class_names,
        "results": results,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
    }


if __name__ == "__main__":
    pipeline = run_pipeline(dmax=4, surrogate_max_depth=3, verbose=True)