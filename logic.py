from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable, Union
import itertools

# -----------------------------
# 1) Core representations
# -----------------------------

class Term: pass

@dataclass(frozen=True)
class Var(Term):
    name: str
    def __repr__(self): return self.name

@dataclass(frozen=True)
class Const(Term):
    name: str
    def __repr__(self): return self.name

# Predicate can be a concrete symbol (str) or a meta-predicate variable (P,Q,R)
Pred = Union[str, Var]

@dataclass(frozen=True)
class Atom:
    pred: Pred
    args: Tuple[Term, ...]
    def __repr__(self):
        args_str = ", ".join(map(str, self.args))
        return f"{self.pred}({args_str})"

@dataclass(frozen=True)
class Clause:
    head: Atom
    body: Tuple[Atom, ...]  # empty for facts
    def __repr__(self):
        if not self.body:
            return f"{self.head}."
        return f"{self.head} :- {', '.join(map(str, self.body))}."


# -----------------------------
# 2) Substitution & apply
# -----------------------------

Subst = Dict[Union[Var], Union[Term, str]]  # Var can map to Term (for term vars) or str (for predicate vars)

def apply_term(t: Term, s: Subst) -> Term:
    while isinstance(t, Var) and t in s and isinstance(s[t], Term):
        t = s[t]
    return t

def apply_pred(p: Pred, s: Subst) -> Pred:
    # If predicate is a Var and bound to a str, replace it
    if isinstance(p, Var) and p in s and isinstance(s[p], str):
        return s[p]
    return p

def apply_atom(a: Atom, s: Subst) -> Atom:
    new_pred = apply_pred(a.pred, s)
    new_args = tuple(apply_term(t, s) for t in a.args)
    return Atom(new_pred, new_args)

def apply_clause(c: Clause, s: Subst) -> Clause:
    return Clause(apply_atom(c.head, s), tuple(apply_atom(b, s) for b in c.body))


# -----------------------------
# 3) Unification (MGU)
# -----------------------------

def unify_terms(t1: Term, t2: Term, s: Optional[Subst] = None) -> Optional[Subst]:
    if s is None: s = {}

    t1 = apply_term(t1, s)
    t2 = apply_term(t2, s)

    if t1 == t2:
        return s

    if isinstance(t1, Var):
        s[t1] = t2
        return s
    if isinstance(t2, Var):
        s[t2] = t1
        return s

    if isinstance(t1, Const) and isinstance(t2, Const):
        return None

    return None

def unify_preds(p1: Pred, p2: Pred, s: Optional[Subst] = None) -> Optional[Subst]:
    if s is None: s = {}

    p1 = apply_pred(p1, s)
    p2 = apply_pred(p2, s)

    if p1 == p2:
        return s

    # Allow predicate meta-variables (Var) to bind to concrete predicate symbols (str)
    if isinstance(p1, Var) and isinstance(p2, str):
        s[p1] = p2
        return s
    if isinstance(p2, Var) and isinstance(p1, str):
        s[p2] = p1
        return s

    # Var-Var predicate binding (optional; binds one to the other as symbol, we keep it simple)
    if isinstance(p1, Var) and isinstance(p2, Var):
        # bind p1 to p2 as a "symbol"? easiest: bind p1 to a string later; for now bind p1 -> p2.name as str is WRONG.
        # We'll just bind p1 to p2 by storing a str is not possible; so we avoid Var-Var and let search bind to str.
        return None

    return None

def unify_atoms(a1: Atom, a2: Atom, s: Optional[Subst] = None) -> Optional[Subst]:
    if s is None: s = {}

    s = unify_preds(a1.pred, a2.pred, s)
    if s is None:
        return None

    if len(a1.args) != len(a2.args):
        return None

    for t1, t2 in zip(a1.args, a2.args):
        s = unify_terms(t1, t2, s)
        if s is None:
            return None

    return s


# -----------------------------
# 4) Standardize-apart (rename vars per clause use)
# -----------------------------

_counter = itertools.count(1)

def standardize_apart(clause: Clause) -> Clause:

    mapping: Dict[Var, Var] = {}

    def rename_term(t: Term) -> Term:
        if isinstance(t, Var):
            if t not in mapping:
                mapping[t] = Var(f"{t.name}_{next(_counter)}")
            return mapping[t]
        return t

    def rename_atom(a: Atom) -> Atom:
        pred = a.pred
        args = tuple(rename_term(t) for t in a.args)
        return Atom(pred, args)

    return Clause(rename_atom(clause.head), tuple(rename_atom(b) for b in clause.body))


# -----------------------------
# 5) Prover: SLD resolution (depth-limited)
# -----------------------------

def prove(goal: Atom, program: List[Clause], depth: int = 15) -> bool:
    return len(prove_all((goal,), program, {}, depth)) > 0

def prove_all(goals: Tuple[Atom, ...], program: List[Clause], s: Subst, depth: int) -> List[Subst]:
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


# -----------------------------
# 6) Metarules
# -----------------------------

@dataclass(frozen=True)
class MetaRule:
    head: Atom
    body: Tuple[Atom, ...]

def chain_metarule() -> MetaRule:
    P = Var("P"); Q = Var("Q"); R = Var("R")
    A = Var("A"); B = Var("B"); C = Var("C")
    head = Atom(P, (A, B))
    body = (Atom(Q, (A, C)), Atom(R, (C, B)))
    return MetaRule(head, body)

def identity_metarule() -> MetaRule:
    # P(A,B) :- Q(A,B)
    P = Var("P"); Q = Var("Q")
    A = Var("A"); B = Var("B")
    head = Atom(P, (A, B))
    body = (Atom(Q, (A, B)),)
    return MetaRule(head, body)

def fork_metarule() -> MetaRule:
    # P(A,B) :- Q(C,A), R(C,B)
    P = Var("P"); Q = Var("Q"); R = Var("R")
    A = Var("A"); B = Var("B"); C = Var("C")
    head = Atom(P, (A, B))
    body = (Atom(Q, (C, A)), Atom(R, (C, B)))
    return MetaRule(head, body)

def tailrec_metarule() -> MetaRule:
    # P(A,B) :- Q(A,C), P(C,B)
    P = Var("P"); Q = Var("Q")
    A = Var("A"); B = Var("B"); C = Var("C")
    head = Atom(P, (A, B))
    body = (Atom(Q, (A, C)), Atom(P, (C, B)))
    return MetaRule(head, body)

# -----------------------------
# 7) Generalize + normalize (pretty vars)
# -----------------------------

def generalize_clause(clause: Clause) -> Clause:
    const_to_var: Dict[Const, Var] = {}
    idx = 0

    def gen_term(t: Term) -> Term:
        nonlocal idx
        if isinstance(t, Const):
            if t not in const_to_var:
                idx += 1
                const_to_var[t] = Var(f"X{idx}")
            return const_to_var[t]
        return t

    def gen_atom(a: Atom) -> Atom:
        return Atom(a.pred, tuple(gen_term(t) for t in a.args))

    return Clause(gen_atom(clause.head), tuple(gen_atom(b) for b in clause.body))

def normalize_vars(clause: Clause) -> Clause:
    mapping: Dict[Var, Var] = {}
    letters = "XYZUVWABCDEFGHIJKLMNOPQRST"
    idx = 0

    def norm_term(t: Term) -> Term:
        nonlocal idx
        if isinstance(t, Var):
            if t not in mapping:
                name = letters[idx] if idx < len(letters) else f"V{idx+1}"
                mapping[t] = Var(name)
                idx += 1
            return mapping[t]
        return t

    def norm_atom(a: Atom) -> Atom:
        return Atom(a.pred, tuple(norm_term(t) for t in a.args))

    return Clause(norm_atom(clause.head), tuple(norm_atom(b) for b in clause.body))

def predicate_vars_in_metarule(m: MetaRule) -> List[Var]:
    vars_ = []
    for atom in (m.head,) + m.body:
        if isinstance(atom.pred, Var) and atom.pred not in vars_:
            vars_.append(atom.pred)
    return vars_


# -----------------------------
# 8) The Metagol algorithm (matches pseudocode)
# -----------------------------

def metagol(
    B: List[Clause],
    Epos: List[Atom],
    Eneg: List[Atom],
    C: List[MetaRule],
    max_d: int = 3
) -> Optional[List[Clause]]:
    """
    B: background knowledge as clauses (facts are clauses with empty body)
    Epos: positive examples (atoms)
    Eneg: negative examples (atoms)
    C: metarules
    """
    # predicate symbols available in BK
    L = sorted({cl.head.pred for cl in B if isinstance(cl.head.pred, str)})

    for d in range(1, max_d + 1):  # iterative deepening
        H: List[Clause] = []
        ok = True

        for e in Epos:
            if prove(e, B + H):
                continue

            # try to extend H until e becomes provable (or we run out of clause budget)
            while len(H) <= d and not prove(e, B + H):
                invented = False

                # choose metarule m
                for m in C:
                    theta = unify_atoms(m.head, e, {})
                    if theta is None:
                        continue

                    # pick predicate symbols for all predicate meta-variables in this metarule
                    pred_vars = predicate_vars_in_metarule(m)

                    choices_per_var = []
                    for pv in pred_vars:
                        if pv in theta:
                            choices_per_var.append([theta[pv]])
                        else:
                            choices_per_var.append(L)

                    for assignment in itertools.product(*choices_per_var):
                        sigma = dict(theta)

                        # bind predicate variables pv -> chosen predicate symbol
                        ok_assign = True
                        for pv, sym in zip(pred_vars, assignment):
                            if not isinstance(sym, str):
                                ok_assign = False
                                break
                            sigma[pv] = sym
                        if not ok_assign:
                            continue

                        # instantiate candidate clause
                        cand_head = apply_atom(m.head, sigma)
                        cand_body = tuple(apply_atom(b, sigma) for b in m.body)

                        # must be fully concrete predicates now (no Var left in pred position)
                        if not isinstance(cand_head.pred, str):
                            continue
                        if any(not isinstance(bb.pred, str) for bb in cand_body):
                            continue

                        cand = Clause(
                            Atom(cand_head.pred, cand_head.args),
                            tuple(Atom(bb.pred, bb.args) for bb in cand_body)
                        )

                        # generalize + normalize
                        cand = normalize_vars(generalize_clause(cand))

                        # candidate must help prove current positive example
                        H2 = H + [cand]
                        if not prove(e, B + H2):
                            continue

                        # must not entail any negative
                        if any(prove(en, B + H2) for en in Eneg):
                            continue

                        # accept
                        H = H2
                        invented = True
                        break

                    if invented:
                        break

                if not invented:
                    ok = False
                    break

            if not ok:
                break

        if ok:
            # final check: all positives provable, no negatives provable
            if all(prove(ep, B + H) for ep in Epos) and not any(prove(en, B + H) for en in Eneg):
                return H

    return None


# -----------------------------
# 9) Tests
# -----------------------------

def test_grandparent():
    a = Const("a"); h = Const("h"); b = Const("b"); l = Const("l")

    B = [
        Clause(Atom("parent", (a, h)), ()),
        Clause(Atom("parent", (h, b)), ()),
        Clause(Atom("parent", (h, l)), ()),
    ]

    Epos = [
        Atom("gp", (a, b)),
        Atom("gp", (a, l)),
    ]

    Eneg = [
        Atom("gp", (a, h)),
        Atom("gp", (h, b)),
    ]

    H = metagol(B, Epos, Eneg, [chain_metarule()], max_d=2)

    print("\n--- Grandparent ---")
    print("H =", H)
    if H:
        for cl in H:
            print(cl)


def test_ancestor():
    a = Const("a"); h = Const("h"); b = Const("b"); c = Const("c")

    B = [
        Clause(Atom("parent", (a, h)), ()),
        Clause(Atom("parent", (h, b)), ()),
        Clause(Atom("parent", (b, c)), ()),
    ]

    Epos = [
        Atom("ancestor", (a, h)),
        Atom("ancestor", (a, b)),
        Atom("ancestor", (a, c)),
        Atom("ancestor", (h, b)),
    ]

    Eneg = [
        Atom("ancestor", (h, a)),
        Atom("ancestor", (c, a)),
    ]

    H = metagol(
        B, Epos, Eneg,
        [identity_metarule(), tailrec_metarule()],
        max_d=3
    )

    print("\n--- Ancestor ---")
    print("H =", H)
    if H:
        for cl in H:
            print(cl)


def test_sibling():
    h = Const("h"); b = Const("b"); l = Const("l"); m = Const("m")

    B = [
        Clause(Atom("parent", (h, b)), ()),
        Clause(Atom("parent", (h, l)), ()),
        Clause(Atom("parent", (h, m)), ()),
    ]

    Epos = [
        Atom("sibling", (b, l)),
        Atom("sibling", (l, m)),
        Atom("sibling", (b, m)),
    ]

    Eneg = [
        Atom("sibling", (h, b)),
    ]

    H = metagol(B, Epos, Eneg, [fork_metarule()], max_d=2)

    print("\n--- Sibling ---")
    print("H =", H)
    if H:
        for cl in H:
            print(cl)


def test_failure():
    a = Const("a"); b = Const("b")

    B = [
        Clause(Atom("parent", (a, b)), ()),
    ]

    Epos = [
        Atom("gp", (a, b)),  # impossible with single parent fact
    ]

    Eneg = []

    H = metagol(B, Epos, Eneg, [chain_metarule()], max_d=2)

    print("\n--- Failure Case ---")
    print("H =", H)


if __name__ == "__main__":
    test_grandparent()
    test_ancestor()
    test_sibling()
    test_failure()



